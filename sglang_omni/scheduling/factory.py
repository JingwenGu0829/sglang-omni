# SPDX-License-Identifier: Apache-2.0
"""Factory functions for creating OmniScheduler instances.

Creates the SGLang infrastructure (ModelWorker, PrefillManager, DecodeManager,
tree cache) and assembles it into an OmniScheduler with the appropriate
ModelRunner.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _create_sglang_infrastructure(
    server_args: Any,
    gpu_id: int,
    *,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
    capture_hidden_layers: list[int] | None = None,
):
    """Create SGLang ModelWorker + memory pools + tree cache + prefill/decode managers.

    Returns (model_worker, tree_cache, req_to_token_pool, token_to_kv_pool_allocator,
             prefill_manager, decode_manager, model_config).
    """
    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )
    from sglang_omni.engines.ar.sglang_backend.scheduler.cache import create_tree_cache
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager

    model_worker = ModelWorker(
        config=ModelWorkerConfig(
            model_arch_override=model_arch_override,
            weight_prefix=weight_prefix,
        ),
        server_args=server_args,
        gpu_id=gpu_id,
    )

    if capture_hidden_layers:
        from sglang_omni.model_runner._hidden_capture import (
            install_hidden_capture_hooks,
        )
        model = model_worker.model_runner.model
        install_hidden_capture_hooks(model, capture_hidden_layers)

    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )

    enable_overlap = not getattr(server_args, "disable_overlap_schedule", False)

    prefill_mgr = PrefillManager(
        page_size=server_args.page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
        max_prefill_tokens=server_args.max_prefill_tokens,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_worker.model_config,
        enable_overlap=enable_overlap,
    )

    decode_mgr = DecodeManager(
        server_args=server_args,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        on_retract=lambda req: prefill_mgr.add_one_request(req),
    )

    return (
        model_worker, tree_cache, req_to_token_pool,
        token_to_kv_pool_allocator, prefill_mgr, decode_mgr,
        model_worker.model_config,
    )


def create_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    speech_enabled: bool = False,
) -> "OmniScheduler":
    """Create OmniScheduler for thinker (standard AR + optional hidden state streaming)."""
    from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_components import SGLangOutputProcessor

    capture_hidden_layers = [0, 24] if speech_enabled else None
    capture_hidden = speech_enabled

    (
        model_worker, tree_cache, req_to_token_pool,
        token_to_kv_pool_allocator, prefill_mgr, decode_mgr,
        model_config,
    ) = _create_sglang_infrastructure(
        server_args, gpu_id,
        capture_hidden_layers=capture_hidden_layers,
    )

    output_proc = SGLangOutputProcessor(
        capture_hidden=capture_hidden,
        capture_hidden_layers=capture_hidden_layers,
        model=model_worker.model_runner.model if capture_hidden_layers else None,
    )

    model_runner = ThinkerModelRunner(model_worker, output_proc)

    scheduler = OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        model_runner=model_runner,
    )

    return scheduler


def create_talker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    weight_prefix: str = "talker.",
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
) -> "OmniScheduler":
    """Create OmniScheduler for talker (feedback AR + fused MTP)."""
    from sglang_omni.model_runner.feedback_ar_model_runner import FeedbackARModelRunner
    from sglang_omni.models.qwen3_omni.callbacks import (
        extract_talker_output,
        talker_prefill_forward,
        write_talker_buffers,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_components import SGLangOutputProcessor

    if feedback_enabled:
        server_args.disable_overlap_schedule = True
    server_args.disable_radix_cache = True

    (
        model_worker, tree_cache, req_to_token_pool,
        token_to_kv_pool_allocator, prefill_mgr, decode_mgr,
        model_config,
    ) = _create_sglang_infrastructure(
        server_args, gpu_id,
        model_arch_override="Qwen3OmniTalker",
        weight_prefix=weight_prefix,
    )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )

    scheduler = OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
    )

    model_runner = FeedbackARModelRunner(
        model_worker, output_proc, scheduler.outbox,
        write_buffers_fn=write_talker_buffers,
        extract_output_fn=extract_talker_output,
        prefill_forward_fn=talker_prefill_forward,
    )

    scheduler._model_runner_override = model_runner

    return scheduler
