# SPDX-License-Identifier: Apache-2.0
"""FeedbackARModelRunner — shared model runner for AR+codebook models.

Handles the common pattern:
  1. Write previous step's feedback into model buffers (before forward)
  2. Model runs backbone + secondary head inside forward()
  3. Read new codes + feedback from model buffers (after forward)
  4. Route codes downstream, store feedback for next step

Model-specific behavior is injected via two callbacks:
  - write_buffers_fn: writes feedback into model's pre-allocated buffers
  - extract_output_fn: reads codes/feedback from model after forward

Used by: Qwen3 Talker, Fish Audio TTS, and future AR+codebook models.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.omni_scheduler import OutgoingMessage

logger = logging.getLogger(__name__)


class WriteBuffersFn(Protocol):
    """Write previous step's feedback into model buffers before forward."""
    def __call__(
        self, model: Any, schedule_batch: Any, requests: list,
    ) -> None: ...


class ExtractOutputFn(Protocol):
    """Read codes/feedback from model after forward. Optionally route downstream."""
    def __call__(
        self, model: Any, schedule_batch: Any, requests: list, outbox: Any,
    ) -> None: ...


class PrefillForwardFn(Protocol):
    """Custom forward for prefill (projected input_embeds, VQ injection, etc)."""
    def __call__(
        self, tp_worker: Any, forward_batch: Any, schedule_batch: Any, requests: list,
    ) -> GenerationBatchResult | None: ...


class FeedbackARModelRunner(ModelRunner):
    """Shared model runner for AR models with codebook feedback.

    The model's forward() handles the full decode step internally
    (backbone + secondary head). This runner just writes/reads the
    model's pre-allocated buffers around the forward call.
    """

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        outbox: Any,
        *,
        write_buffers_fn: WriteBuffersFn,
        extract_output_fn: ExtractOutputFn,
        prefill_forward_fn: PrefillForwardFn | None = None,
    ):
        super().__init__(tp_worker, output_processor)
        self._outbox = outbox
        self._write_buffers = write_buffers_fn
        self._extract_output = extract_output_fn
        self._prefill_forward = prefill_forward_fn

    def prepare_forward(self, forward_batch, schedule_batch, requests):
        if not schedule_batch.forward_mode.is_extend():
            # Decode: write feedback into model buffers
            self._write_buffers(self.model, schedule_batch, requests)
            return None

        # Prefill: custom forward if provided
        if self._prefill_forward is not None:
            return self._prefill_forward(
                self.tp_worker, forward_batch, schedule_batch, requests)
        return None

    def post_forward(self, result, forward_batch, schedule_batch, requests):
        if schedule_batch.forward_mode.is_extend():
            return
        self._extract_output(self.model, schedule_batch, requests, self._outbox)
