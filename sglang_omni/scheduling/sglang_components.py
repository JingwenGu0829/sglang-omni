# SPDX-License-Identifier: Apache-2.0
"""SGLang-specific scheduling components.

Batch selection, output processing, iteration control, and model execution
for SGLang-backed AR models. Used by OmniScheduler.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-request data
# ---------------------------------------------------------------------------


@dataclass
class SGLangARRequestData:
    """SGLang AR per-request state. Stored in SchedulerRequest.data."""
    input_ids: Any = None
    attention_mask: Any = None
    model_inputs: dict[str, Any] = field(default_factory=dict)
    output_ids: list[int] = field(default_factory=list)
    extra_model_outputs: dict[str, Any] = field(default_factory=dict)
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    suppress_tokens: list[int] | None = None



# ---------------------------------------------------------------------------
# Output processor
# ---------------------------------------------------------------------------


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def __init__(
        self,
        capture_hidden: bool = False,
        capture_hidden_layers: list[int] | None = None,
        model: Any = None,
    ):
        self._capture_hidden = capture_hidden
        self._capture_hidden_layers = capture_hidden_layers
        self._model = model

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        token_list = (
            model_output.next_token_ids.tolist()
            if model_output.next_token_ids is not None
            else []
        )

        # Extract hidden states if configured and available
        hidden_states_dict = None
        stream_hidden_states = None
        if self._capture_hidden:
            hidden_states_dict = self._extract_hidden_states(model_output)
            stream_hidden_states = self._extract_stream_hidden_states(model_output)

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            extra = None
            if hidden_states_dict is not None:
                if "_single" in hidden_states_dict:
                    extra = {"hidden_states": hidden_states_dict["_single"][i]}
                else:
                    per_req = {}
                    for key, tensor in hidden_states_dict.items():
                        per_req[key] = tensor[i] if tensor.ndim >= 2 else tensor
                    extra = {"hidden_states": per_req}
                    if stream_hidden_states is not None:
                        extra["stream_hidden_states"] = (
                            stream_hidden_states[i]
                            if stream_hidden_states.ndim >= 2
                            else stream_hidden_states
                        )
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
                extra=extra,
            )
        return outputs

    def _extract_hidden_states(
        self, model_output: Any
    ) -> dict[str, torch.Tensor] | None:
        """Extract hidden states from model output or side-channel.

        Priority:
        1. Side-channel (_captured_aux_hidden_states) from hidden capture hooks
        2. logits_output.hidden_states (legacy single-tensor path)
        """
        # Check side-channel first (set by _hidden_capture hooks)
        if self._model is not None and self._capture_hidden_layers:
            aux = getattr(self._model, "_captured_aux_hidden_states", None)
            if aux is not None:
                result = {}
                for layer_id, tensor in zip(self._capture_hidden_layers, aux):
                    key = "embed" if layer_id == 0 else layer_id
                    result[key] = tensor.clone()
                return result

        # Fallback: logits_output.hidden_states
        logits_output = getattr(model_output, "logits_output", None)
        if logits_output is None:
            return None
        raw_hidden = getattr(logits_output, "hidden_states", None)
        if raw_hidden is None:
            return None

        if isinstance(raw_hidden, dict):
            return raw_hidden
        elif isinstance(raw_hidden, torch.Tensor):
            return {"_single": raw_hidden}
        return None

    def _extract_stream_hidden_states(self, model_output: Any) -> torch.Tensor | None:
        logits_output = getattr(model_output, "logits_output", None)
        if logits_output is None:
            return None
        raw_hidden = getattr(logits_output, "hidden_states", None)
        return raw_hidden if isinstance(raw_hidden, torch.Tensor) else None


