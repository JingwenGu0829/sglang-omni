# SPDX-License-Identifier: Apache-2.0
"""Multi-process pipeline runner.

Spawns each pipeline stage in its own OS process. Main process runs only
the Coordinator. Stages communicate via ZMQ (control plane) and relay
(data plane).
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
from typing import Any

from sglang_omni.config.compiler import (
    _allocate_endpoints,
    _build_relay_config,
    _compile_stage,
    _create_input_handler,
    _detect_same_gpu_targets,
    _wrap_get_next,
    _wire_stream_targets,
)
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.pipeline import Coordinator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _build_stage_process_config(
    *,
    pipeline_config: PipelineConfig,
    stage_name: str,
    stage_endpoints: dict[str, str],
    all_endpoints: dict[str, str],
    name_map: dict[str, str],
) -> dict[str, Any]:
    """Build a picklable config dict for a stage subprocess."""
    return {
        "pipeline_config": pipeline_config.model_dump(),
        "stage_name": stage_name,
        "stage_endpoints": stage_endpoints,
        "all_endpoints": all_endpoints,
        "name_map": name_map,
    }


def _resolve_relay_config(
    stage_cfg: StageConfig, global_cfg: PipelineConfig
) -> dict[str, Any]:
    """Build relay config with gpu_id from gpu_placement.

    The base _build_relay_config uses relay.device to determine gpu_id,
    which defaults to 0 for "cuda". For multi-process deployment, we
    override with the actual gpu_placement value.
    """
    relay_config = _build_relay_config(stage_cfg, global_cfg)
    if stage_cfg.relay.device != "cpu":
        placement_gpu = global_cfg.gpu_placement.get(stage_cfg.name)
        if placement_gpu is not None:
            relay_config["gpu_id"] = placement_gpu
    return relay_config


def _stage_process_entry(
    config_dict: dict[str, Any],
    ready_event: multiprocessing.Event,
) -> None:
    """Subprocess entrypoint: compile and run a single Stage."""
    import inspect
    import logging
    import sys

    from sglang_omni.config.schema import PipelineConfig, StageConfig
    from sglang_omni.pipeline.stage.runtime import Stage
    from sglang_omni.utils import import_string

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    log = logging.getLogger(f"stage.{config_dict['stage_name']}")

    try:
        stage_name = config_dict["stage_name"]
        stage_endpoints = config_dict["stage_endpoints"]
        all_endpoints = config_dict["all_endpoints"]
        name_map = config_dict["name_map"]

        pipeline_config = PipelineConfig(**config_dict["pipeline_config"])
        stages_cfg, fused_name_map, _ = pipeline_config.apply_fusion()
        name_map.update(fused_name_map)

        stage_cfg = next((s for s in stages_cfg if s.name == stage_name), None)
        if stage_cfg is None:
            log.error("Stage %s not found in config", stage_name)
            return

        log.info("Compiling stage %s...", stage_name)

        # --- Build stage components ---
        factory = import_string(stage_cfg.executor.factory)
        get_next = import_string(stage_cfg.get_next)
        get_next = _wrap_get_next(get_next, name_map)
        input_handler = _create_input_handler(stage_cfg.input_handler, name_map=name_map)

        # Inject model_path and gpu_id
        if (
            "model_path" in inspect.signature(factory).parameters
            and "model_path" not in stage_cfg.executor.args
        ):
            stage_cfg.executor.args["model_path"] = pipeline_config.model_path

        if (
            "gpu_id" in inspect.signature(factory).parameters
            and "gpu_id" not in stage_cfg.executor.args
        ):
            gpu_id = pipeline_config.gpu_placement.get(stage_cfg.name, 0)
            stage_cfg.executor.args["gpu_id"] = gpu_id

        # Factory must return a scheduler (with inbox/outbox)
        scheduler = factory(**stage_cfg.executor.args)

        # Create Stage
        stage = Stage(
            name=stage_cfg.name,
            get_next=get_next,
            recv_endpoint=stage_endpoints[stage_cfg.name],
            coordinator_endpoint=all_endpoints["completion"],
            abort_endpoint=all_endpoints["abort"],
            endpoints=stage_endpoints,
            input_handler=input_handler,
            relay_config=_resolve_relay_config(stage_cfg, pipeline_config),
            scheduler=scheduler,
        )

        # Wire stream targets
        from sglang_omni.config.compiler import _detect_same_gpu_targets
        from sglang_omni.pipeline.stage.stream_queue import StreamQueue

        targets = stage_cfg.stream_to
        if targets:
            all_targets = [t.to_stage for t in targets]
            cfg_map = {s.name: s for s in stages_cfg}
            same_gpu = _detect_same_gpu_targets(
                stage_cfg, targets,
                gpu_placement=pipeline_config.gpu_placement, cfg_map=cfg_map,
            )
            stage._stream_targets = all_targets
            stage._same_gpu_targets = same_gpu

        # Check if this stage receives streams
        is_receiver = any(
            any(t.to_stage == stage_name for t in other.stream_to)
            for other in stages_cfg
        )
        if is_receiver:
            stage._stream_queue = StreamQueue(max_pending=4096)

        # Run
        async def _start_and_run():
            await stage.start()
            log.info("Stage %s ready", stage_name)
            ready_event.set()
            await stage.run()

        asyncio.run(_start_and_run())

    except Exception:
        import traceback
        log.error("Stage process failed:\n%s", traceback.format_exc())
        sys.exit(1)


# ---------------------------------------------------------------------------
# MultiProcessPipelineRunner
# ---------------------------------------------------------------------------


class MultiProcessPipelineRunner:
    """Run each pipeline stage in its own OS process.

    Main process runs only the Coordinator.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._coordinator: Coordinator | None = None
        self._processes: list[multiprocessing.Process] = []
        self._completion_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._started = False

    @property
    def coordinator(self) -> Coordinator:
        if self._coordinator is None:
            raise RuntimeError("Runner not started")
        return self._coordinator

    async def start(self, timeout: float = 120.0) -> None:
        if self._started:
            raise RuntimeError("Already started")

        try:
            stages_cfg, name_map, entry_stage = self._config.apply_fusion()
            endpoints = _allocate_endpoints(self._config, stages=stages_cfg)
            stage_endpoints = {s.name: endpoints[f"stage_{s.name}"] for s in stages_cfg}

            self._coordinator = Coordinator(
                completion_endpoint=endpoints["completion"],
                abort_endpoint=endpoints["abort"],
                entry_stage=entry_stage,
                terminal_stages=self._config.terminal_stages or None,
            )
            await self._coordinator.start()
            self._completion_task = asyncio.create_task(
                self._coordinator.run_completion_loop()
            )

            ctx = multiprocessing.get_context("spawn")
            ready_events = []

            for stage_cfg in stages_cfg:
                ready = ctx.Event()
                config_dict = _build_stage_process_config(
                    pipeline_config=self._config,
                    stage_name=stage_cfg.name,
                    stage_endpoints=stage_endpoints,
                    all_endpoints=endpoints,
                    name_map=name_map,
                )
                p = ctx.Process(
                    target=_stage_process_entry,
                    args=(config_dict, ready),
                    name=f"stage-{stage_cfg.name}",
                    daemon=True,
                )
                p.start()
                self._processes.append(p)
                ready_events.append(ready)

            # Wait for all stages
            import time as _time
            loop = asyncio.get_running_loop()
            for i, event in enumerate(ready_events):
                stage_name = stages_cfg[i].name
                p = self._processes[i]
                deadline = _time.monotonic() + timeout
                while not event.is_set():
                    remaining = deadline - _time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Stage {stage_name} did not become ready within {timeout}s"
                        )
                    if not p.is_alive():
                        raise RuntimeError(
                            f"Stage {stage_name} process died during startup "
                            f"(exit code {p.exitcode})"
                        )
                    await loop.run_in_executor(None, event.wait, min(remaining, 1.0))
                logger.info("Stage %s ready", stage_name)

            for i, p in enumerate(self._processes):
                if not p.is_alive() and p.exitcode != 0:
                    raise RuntimeError(
                        f"Stage {stages_cfg[i].name} exited with code {p.exitcode}"
                    )

            for stage_cfg in stages_cfg:
                self._coordinator.register_stage(
                    stage_cfg.name, stage_endpoints[stage_cfg.name]
                )

            self._started = True
            self._monitor_task = asyncio.create_task(self._monitor_children())
            logger.info(
                "MultiProcessPipelineRunner started: %d stages", len(self._processes)
            )

        except Exception:
            for p in self._processes:
                if p.is_alive():
                    p.terminate()
            for p in self._processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)
            self._processes.clear()
            if self._completion_task is not None:
                self._completion_task.cancel()
                try:
                    await self._completion_task
                except asyncio.CancelledError:
                    pass
                self._completion_task = None
            if self._coordinator is not None:
                try:
                    await self._coordinator.stop()
                except Exception:
                    pass
                self._coordinator = None
            raise

    async def _monitor_children(self) -> None:
        while self._started:
            for i, p in enumerate(self._processes):
                if not p.is_alive():
                    logger.error(
                        "Stage process %d (pid=%d) died with exitcode=%s",
                        i, p.pid, p.exitcode,
                    )
                    await self.stop()
                    return
            await asyncio.sleep(5.0)

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False

        if self._monitor_task is not None:
            current = asyncio.current_task()
            if current != self._monitor_task:
                self._monitor_task.cancel()
            self._monitor_task = None

        try:
            await self._coordinator.shutdown_stages()
        except Exception as e:
            logger.warning("shutdown_stages error: %s", e)

        for p in self._processes:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning("Terminating stuck process %s", p.name)
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)

        if self._completion_task is not None:
            self._completion_task.cancel()
            try:
                await self._completion_task
            except asyncio.CancelledError:
                pass

        await self._coordinator.stop()
        self._processes.clear()
