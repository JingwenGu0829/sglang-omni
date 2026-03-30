# SPDX-License-Identifier: Apache-2.0
"""Base model runner — shared execute() pipeline for all AR models.

Handles: ForwardBatch construction, forward pass, sampling, logit
post-processing, output extraction. Subclasses override prepare_forward()
and post_forward() hooks for model-specific behavior.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.scheduling.types import ModelRunnerOutput

logger = logging.getLogger(__name__)


class ModelRunner:
    """Base AR model runner.

    Subclassed by ThinkerModelRunner (multimodal inject) and
    TalkerModelRunner (feedback buffers + secondary extraction).
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
        self.tp_worker = tp_worker
        self.output_processor = output_processor
        self.device = torch.device(f"cuda:{tp_worker.gpu_id}")
        self.model = tp_worker.model_runner.model

    def execute(self, scheduler_output: Any) -> ModelRunnerOutput:
        """Full pipeline: build batch → prepare → forward → post → sample → output."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        model_worker_batch = schedule_batch.get_model_worker_batch()

        if self.output_processor._capture_hidden:
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.tp_worker.model_runner
        )

        # Hook: model-specific preparation. Returns batch_result if it ran
        # a custom forward path, or None for standard forward.
        batch_result = self.prepare_forward(
            forward_batch, schedule_batch, scheduler_output.requests
        )

        if batch_result is None:
            # Standard forward path
            batch_result = self.tp_worker.forward_batch_generation(forward_batch)

        # Hook: model-specific post-processing
        self.post_forward(batch_result, forward_batch, schedule_batch, scheduler_output.requests)

        # Sampling + logit processing
        if schedule_batch.is_prefill_only:
            batch_result.next_token_ids = torch.zeros(
                len(model_worker_batch.seq_lens),
                dtype=torch.long,
                device=model_worker_batch.input_ids.device,
            )
        else:
            self._apply_repetition_penalty(
                batch_result.logits_output, scheduler_output.requests
            )
            self._apply_codec_suppress_tokens(
                batch_result.logits_output, scheduler_output.requests
            )
            batch_result.next_token_ids = self.tp_worker.model_runner.sample(
                batch_result.logits_output, forward_batch
            )
        schedule_batch.output_ids = batch_result.next_token_ids

        # Output extraction
        outputs = self.output_processor.process(batch_result, scheduler_output)
        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def prepare_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any | None:
        """Called before forward. Return batch_result if custom forward done,
        or None for standard tp_worker.forward_batch_generation()."""
        return None

    def post_forward(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Called after forward. Extract secondary outputs, etc."""
        pass

    # ------------------------------------------------------------------
    # Shared logit processing
    # ------------------------------------------------------------------

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        logits = getattr(logits_output, "next_token_logits", None)
        if logits is None or logits.ndim != 2:
            return
        for row_idx, sched_req in enumerate(requests):
            data = getattr(sched_req, "data", None)
            req = getattr(data, "req", None)
            if req is None:
                continue
            penalty = getattr(
                getattr(req, "sampling_params", None), "repetition_penalty", 1.0,
            )
            if penalty == 1.0:
                continue
            output_ids = getattr(req, "output_ids", None)
            if not output_ids:
                continue
            token_ids = list(set(output_ids))
            valid = [t for t in token_ids if 0 <= t < logits.shape[1]]
            if not valid:
                continue
            idx = torch.tensor(valid, dtype=torch.long, device=logits.device)
            scores = logits[row_idx, idx]
            scores = torch.where(scores > 0, scores / penalty, scores * penalty)
            logits[row_idx, idx] = scores

    def _apply_codec_suppress_tokens(self, logits_output: Any, requests: list) -> None:
        logits = getattr(logits_output, "next_token_logits", None)
        if logits is None or logits.ndim != 2:
            return
        for row_idx, sched_req in enumerate(requests):
            data = getattr(sched_req, "data", None)
            suppress_tokens = getattr(data, "suppress_tokens", None)
            if not suppress_tokens:
                req = getattr(data, "req", None)
                suppress_tokens = getattr(req, "_codec_suppress_tokens", None)
            if not suppress_tokens:
                continue
            for token_id in suppress_tokens:
                if 0 <= int(token_id) < logits.shape[1]:
                    logits[row_idx, int(token_id)] = float("-inf")
