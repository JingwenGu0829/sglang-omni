# SPDX-License-Identifier: Apache-2.0
"""
OmniScheduler — SGLang Scheduler subclass for sglang-omni pipeline stages.

Inherits SGLang's battle-tested scheduling logic (get_next_batch_to_run,
run_batch, process_batch_result) but replaces IO (inbox/outbox instead of ZMQ).

Does NOT call super().__init__() — skips ZMQ, tokenizer, metrics, distributed
setup. Only initializes the attributes that the inherited methods need.
"""
from __future__ import annotations

import logging
import queue as _queue_mod
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as SGLangScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Messages between Stage and Scheduler
# ---------------------------------------------------------------------------


@dataclass
class IncomingMessage:
    request_id: str
    type: Literal["new_request", "stream_chunk", "stream_done"]
    data: Any = None


@dataclass
class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream"]
    data: Any = None
    target: str | None = None


# ---------------------------------------------------------------------------
# OmniScheduler
# ---------------------------------------------------------------------------


class OmniScheduler(SGLangScheduler):
    """SGLang Scheduler subclass for pipeline stages.

    Inherits: get_next_batch_to_run(), run_batch(), process_batch_result(),
    event_loop_normal(), event_loop_overlap(), and all batch management.

    Overrides: __init__ (skip ZMQ/tokenizer/metrics), recv_requests() (inbox),
    result sending (outbox).
    """

    def __init__(
        self,
        tp_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        *,
        model_runner: Any = None,
        enable_overlap: bool = False,
    ):
        # ── DO NOT call super().__init__() ──
        # SGLang's __init__ sets up ZMQ, tokenizer, metrics, distributed groups.
        # We only initialize the attributes that inherited methods need.

        # Stage communication
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()

        # Server args and config
        self.server_args = server_args
        self.model_config = model_config
        self.gpu_id = tp_worker.gpu_id if hasattr(tp_worker, "gpu_id") else 0
        self.tp_rank = 0
        self.tp_size = server_args.tp_size
        self.pp_rank = 0
        self.pp_size = server_args.pp_size
        self.dp_rank = None
        self.dp_size = 1
        self.moe_ep_rank = 0
        self.moe_ep_size = 1
        self.page_size = server_args.page_size
        self.enable_overlap = enable_overlap

        # Model worker
        self.tp_worker = tp_worker
        self.model_worker = tp_worker

        # Memory management
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

        # Model runner override (for FeedbackModelRunner)
        self._model_runner_override = model_runner

        # Running status (from init_running_status)
        self.waiting_queue = []
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.cur_batch = None
        self.last_batch = None
        self.forward_ct = 0
        self.return_health_check_ct = 0
        self.num_retracted_reqs = 0
        self.num_paused_reqs = 0
        self.sessions = {}
        self.forward_sleep_time = None
        self._engine_paused = False

        # Chunked prefill (from init_chunked_prefill)
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None
            and server_args.enable_mixed_chunk
        )
        self.enable_dynamic_chunking = False

        # Schedule policy (from init_schedule_policy)
        from sglang.srt.managers.schedule_policy import SchedulePolicy
        self.schedule_policy = server_args.schedule_policy
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            server_args.enable_hierarchical_cache,
            server_args.enable_priority_scheduling,
            server_args.schedule_low_priority_values_first,
        )
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.try_preemption = server_args.enable_priority_scheduling
        from sglang.srt.constrained import envs
        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        self.new_token_ratio = self.init_new_token_ratio
        self.prefill_delayer = None

        # Features we don't use but inherited methods may check
        self.enable_lora = False
        self.enable_pdmux = False
        self.enable_metrics = False
        self.enable_trace = False
        self.enable_hierarchical_cache = False
        self.enable_hicache_storage = False
        self.enable_kv_cache_events = False
        self.is_generation = True
        self.skip_tokenizer_init = True
        self.stream_interval = 1
        self.max_recv_per_poll = 64
        self.enable_lora_overlap_loading = False

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        self.spec_algorithm = SpeculativeAlgorithm.NONE

        from sglang.srt.managers.scheduler import DllmStagingReqs
        self.dllm_config = None
        self.dllm_staging_reqs = DllmStagingReqs(dllm_config=None)

        # Stubs for methods that check these
        self.draft_worker = None
        self.watchdog = None
        self.soft_watchdog = None
        self.recv_skipper = None
        self.idle_sleeper = None
        self.grammar_manager = None
        self.require_mlp_sync = False
        self.attn_tp_rank = 0
        self.attn_tp_size = 1
        self.attn_dp_rank = 0
        self.metrics_collector = None
        self.pad_input_ids_func = None

        # Flag
        self._running = False

    # ------------------------------------------------------------------
    # Override: recv_requests — read from inbox instead of ZMQ
    # ------------------------------------------------------------------

    def recv_requests(self):
        """Drain inbox. Route by message type. Returns list of new requests."""
        new_reqs = []
        while True:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break

            if msg.type == "new_request":
                new_reqs.append(msg.data)
            elif msg.type == "stream_chunk":
                self._on_stream_chunk(msg.request_id, msg.data)
            elif msg.type == "stream_done":
                self._on_stream_done(msg.request_id)

        return new_reqs

    def _on_stream_chunk(self, request_id: str, chunk: Any) -> None:
        """Append streaming hidden state to an existing request."""
        for req in self.running_batch.reqs:
            if req.rid == request_id:
                trailing = getattr(req, "trailing_text_hidden", None)
                if isinstance(trailing, list):
                    trailing.append(chunk)
                return

    def _on_stream_done(self, request_id: str) -> None:
        """Mark thinker done for a request."""
        for req in self.running_batch.reqs:
            if req.rid == request_id:
                if hasattr(req, "thinker_chunks_done"):
                    req.thinker_chunks_done = True
                return

    # ------------------------------------------------------------------
    # Override: process_input_requests — add to waiting_queue
    # ------------------------------------------------------------------

    def process_input_requests(self, recv_reqs):
        """Add new requests to waiting queue."""
        for req_data in recv_reqs:
            if hasattr(req_data, "req"):
                # SGLang Req object wrapped in request data
                self.waiting_queue.append(req_data.req)
            else:
                self.waiting_queue.append(req_data)

    # ------------------------------------------------------------------
    # Override: run_batch — use model_runner if set
    # ------------------------------------------------------------------

    def run_batch(self, batch, pp_proxy_tensors=None):
        if self._model_runner_override is not None:
            # FeedbackModelRunner or custom model runner
            return self._model_runner_override.execute(batch)
        return super().run_batch(batch, pp_proxy_tensors)

    # ------------------------------------------------------------------
    # Override: send results to outbox instead of ZMQ
    # ------------------------------------------------------------------

    def send_to_tokenizer(self):
        """Stub — we don't send to tokenizer. Results go through outbox."""
        pass

    # ------------------------------------------------------------------
    # Event loop control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the scheduling loop (blocks the thread)."""
        self._running = True
        if self.enable_overlap:
            self.event_loop_overlap()
        else:
            self.event_loop_normal()

    def stop(self) -> None:
        self._running = False
