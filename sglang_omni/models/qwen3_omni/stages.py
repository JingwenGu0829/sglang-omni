# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Qwen3-Omni pipelines.

Each factory returns either:
- A callable (compute_fn) for simple stages
- An OmniScheduler for AR stages
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer

from sglang_omni.engines.ar.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.payload_types import OmniEvent
from sglang_omni.models.qwen3_omni.request_builders import (
    apply_encoder_result,
    build_encoder_request,
)
from sglang_omni.models.qwen3_omni.merge import decode_events
from sglang_omni.models.qwen3_omni.routing import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.qwen3_omni.payload_types import PipelineState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> PipelineState:
    return PipelineState.from_dict(payload.data)


def store_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


# ---------------------------------------------------------------------------
# Simple stages — return SimpleScheduler
# ---------------------------------------------------------------------------


def create_preprocessing_executor(model_path: str):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    preprocessor = Qwen3OmniPreprocessor(model_path=model_path)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return SimpleScheduler(_preprocess)


def create_aggregate_executor():
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return SimpleScheduler(_identity)


def create_image_encoder_executor(
    model_path: str, *, device: str = "cuda", dtype: str | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    model = Qwen3OmniImageEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)
        if request.get("_skip"):
            result = request.get("_result", {})
        else:
            with torch.no_grad():
                result = model(request)
        apply_encoder_result(state, stage_name=IMAGE_STAGE, result=result)
        return store_state(payload, state)

    return SimpleScheduler(_encode)


def create_audio_encoder_executor(
    model_path: str, *, device: str = "cuda", dtype: str | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    model = Qwen3OmniAudioEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        request = build_encoder_request(state, stage_name=AUDIO_STAGE)
        if request.get("_skip"):
            result = request.get("_result", {})
        else:
            with torch.no_grad():
                result = model(request)
        apply_encoder_result(state, stage_name=AUDIO_STAGE, result=result)
        return store_state(payload, state)

    return SimpleScheduler(_encode)


def create_decode_executor(model_path: str):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def _decode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [], "step": 0, "is_final": True, "extra_model_outputs": {},
            }

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(decode_events(
            thinker_out=thinker_out, state=state,
            tokenizer=tokenizer, eos_token_id=eos_token_id, step=step,
        ))
        event_dicts = [_event_to_dict(event) for event in events]

        result: dict[str, Any] = {"events": event_dicts}
        final_event = next(
            (e for e in reversed(events) if e.is_final or e.type in {"text_final", "final"}),
            None,
        )
        if final_event is not None:
            result.update(final_event.payload)
            result.setdefault("modality", final_event.modality)

        if "text" not in result:
            output_ids = thinker_out.get("output_ids")
            if callable(getattr(tokenizer, "decode", None)) and isinstance(output_ids, list) and output_ids:
                result["text"] = tokenizer.decode(output_ids, skip_special_tokens=True)
                result.setdefault("modality", "text")

        payload.data = result
        return payload

    return SimpleScheduler(_decode)


# ---------------------------------------------------------------------------
# AR stages — return OmniScheduler
# ---------------------------------------------------------------------------


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    thinker_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = False,
):
    """Returns OmniScheduler for thinker."""
    from sglang_omni.scheduling.factory import create_thinker_scheduler

    server_args = build_sglang_server_args(
        model_path, context_length=thinker_max_seq_len, **(server_args_overrides or {})
    )
    return create_thinker_scheduler(
        server_args, gpu_id, speech_enabled=speech_enabled,
    )


def create_talker_ar_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    talker_max_seq_len: int = 4096,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
    weight_prefix: str = "talker.",
):
    """Returns OmniScheduler for talker."""
    from sglang_omni.scheduling.factory import create_talker_scheduler

    server_args = build_sglang_server_args(
        model_path, context_length=talker_max_seq_len, **(server_args_overrides or {})
    )
    return create_talker_scheduler(
        server_args, gpu_id,
        weight_prefix=weight_prefix,
        speech_enabled=speech_enabled,
        feedback_enabled=feedback_enabled,
    )
