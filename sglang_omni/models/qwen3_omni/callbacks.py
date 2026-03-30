# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker callbacks for FeedbackARModelRunner.

write_buffers: writes feedback_embeds + trailing/pad into model._feedback_buffer
extract_output: reads _output_codes → outbox (code2wav), _output_embeds → feedback
prefill_forward: custom forward with projected input_embeds
"""
from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.scheduling.omni_scheduler import OutgoingMessage


def write_talker_buffers(model: Any, schedule_batch: Any, requests: list) -> None:
    """Write feedback + trailing into model._feedback_buffer."""
    if not getattr(model, "_cp_enabled", False):
        return

    buf = model._feedback_buffer
    mask = model._feedback_mask
    device = buf.device
    dtype = buf.dtype

    for i, req in enumerate(schedule_batch.reqs):
        sched_req = _find_scheduler_request(req, requests)
        if sched_req is None:
            continue
        data = sched_req.data

        feedback = getattr(data, "feedback_embeds", None)
        if feedback is None:
            continue

        combined = feedback.to(device=device, dtype=dtype).reshape(-1)

        step = max(int(getattr(data, "generation_steps", 0)) - 1, 0)
        trailing = getattr(data, "trailing_text_hidden", None)
        tts_pad = getattr(data, "tts_pad_embed", None)
        thinker_done = bool(getattr(data, "thinker_chunks_done", True))

        trailing_value = None
        if isinstance(trailing, list) and step < len(trailing):
            trailing_value = trailing[step]
        elif isinstance(trailing, torch.Tensor) and step < trailing.shape[0]:
            trailing_value = trailing[step]

        if trailing_value is not None:
            combined = combined + trailing_value.to(device=device, dtype=dtype).reshape(-1)
        elif thinker_done and tts_pad is not None:
            combined = combined + tts_pad.to(device=device, dtype=dtype).reshape(-1)

        buf[i] = combined
        mask[i] = True
        data.feedback_embeds = None


def extract_talker_output(
    model: Any, schedule_batch: Any, requests: list, outbox: Any,
    code2wav_target: str = "code2wav",
) -> None:
    """Read fused CP codes/embeds. Codes → outbox → code2wav. Embeds → feedback."""
    if not getattr(model, "_cp_enabled", False):
        return

    bs = len(schedule_batch.reqs)
    codes = model._output_codes[:bs].clone()
    embeds = model._output_embeds[:bs].clone()

    for i, req in enumerate(schedule_batch.reqs):
        request_id = req.rid if hasattr(req, "rid") else str(id(req))
        outbox.put(OutgoingMessage(
            request_id=request_id, type="stream",
            data=codes[i], target=code2wav_target,
        ))
        sched_req = _find_scheduler_request(req, requests)
        if sched_req is not None:
            sched_req.data.feedback_embeds = embeds[i]


def talker_prefill_forward(
    tp_worker: Any, forward_batch: Any, schedule_batch: Any, requests: list,
) -> GenerationBatchResult | None:
    """Custom prefill with projected input_embeds."""
    inner_model = tp_worker.model_runner.model
    outer = inner_model.thinker if hasattr(inner_model, "thinker") else inner_model
    if not hasattr(outer, "prepare_input_embeds"):
        return None

    has_projected = (
        forward_batch.input_embeds is not None
        or any(
            bool(getattr(getattr(r, "data", None), "input_embeds_are_projected", False))
            for r in requests
        )
    )
    if not has_projected:
        return None

    input_embeds = forward_batch.input_embeds
    if input_embeds is None:
        rows = []
        for sr in requests:
            data = getattr(sr, "data", None)
            req = getattr(data, "req", None)
            ie = getattr(req, "input_embeds", None)
            if ie:
                rows.extend(ie)
        if not rows:
            return None
        input_embeds = torch.as_tensor(
            rows, device=forward_batch.input_ids.device, dtype=torch.float32)

    model_runner = tp_worker.model_runner
    model_dtype = next(outer.parameters()).dtype
    model_runner.attn_backend.init_forward_metadata(forward_batch)
    positions = forward_batch.positions
    if forward_batch.mrope_positions is not None:
        positions = forward_batch.mrope_positions
    input_embeds = input_embeds.to(device=forward_batch.input_ids.device, dtype=model_dtype)

    logits_output = outer(
        input_ids=forward_batch.input_ids,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        input_embeds_are_projected=True,
    )
    return GenerationBatchResult(logits_output=logits_output, can_run_cuda_graph=False)


def _find_scheduler_request(req: Any, requests: list) -> Any:
    for sr in requests:
        if hasattr(sr, "data") and hasattr(sr.data, "req") and sr.data.req is req:
            return sr
    return None
