# SPDX-License-Identifier: Apache-2.0
"""Fish Audio S2-Pro callbacks for FeedbackARModelRunner.

write_buffers: writes previous step's codebook values into model._vq_codes/_vq_mask
extract_output: reads _output_codes → per-request S2ProStepOutput
prefill_forward: injects VQ embeddings into input_embeds
"""
from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.scheduling.omni_scheduler import OutgoingMessage


def write_fish_buffers(model: Any, schedule_batch: Any, requests: list) -> None:
    """Write previous step's codebook values into model VQ buffers."""
    input_ids = schedule_batch.reqs[0].last_node.input_ids if hasattr(schedule_batch.reqs[0], "last_node") else None
    # The model reads _vq_mask and _vq_codes during forward
    # For decode: set mask based on whether token is semantic, set codes from prev step
    text_model = model
    bs = len(schedule_batch.reqs)

    for i, req in enumerate(schedule_batch.reqs):
        sched_req = _find_scheduler_request(req, requests)
        if sched_req is None:
            continue
        data = sched_req.data
        last_codes = getattr(data, "_last_codebook_values", None)
        if last_codes is not None and hasattr(text_model, "_vq_codes"):
            text_model._vq_codes[i].copy_(last_codes)


def extract_fish_output(
    model: Any, schedule_batch: Any, requests: list, outbox: Any,
) -> None:
    """Read codebook codes from model output buffers."""
    text_model = model
    for i, req in enumerate(schedule_batch.reqs):
        sched_req = _find_scheduler_request(req, requests)
        if sched_req is None:
            continue
        data = sched_req.data

        if data.req.is_chunked > 0:
            continue

        codes = text_model._output_codes[i].unsqueeze(-1).clone()

        # Store codebook values for next step's VQ embedding
        data._last_codebook_values = codes[1:, 0].clone()

        # Store semantic token
        semantic_token = codes[0, -1].item()
        data._previous_semantic_tokens.append(semantic_token)
        data.output_codes.append(codes)
        data.req.output_ids.append(semantic_token)


def fish_prefill_forward(
    tp_worker: Any, forward_batch: Any, schedule_batch: Any, requests: list,
) -> GenerationBatchResult | None:
    """Inject VQ embeddings into input_embeds for prefill."""
    model_worker_batch = schedule_batch.get_model_worker_batch()
    text_model = tp_worker.model_runner.model
    audio_decoder = text_model._audio_decoder
    embed_tokens = text_model.get_embed_tokens()
    device = model_worker_batch.input_ids.device

    input_ids = model_worker_batch.input_ids
    text_embeds = embed_tokens(input_ids)

    offset = 0
    for sched_req in requests:
        data = sched_req.data
        req_len = data.req.extend_input_len

        if (
            getattr(data, "vq_mask_tokens", None) is not None
            and getattr(data, "vq_parts", None) is not None
            and len(data.vq_parts) > 0
        ):
            vq_mask = data.vq_mask_tokens.to(device)
            if vq_mask.dim() == 2:
                vq_mask = vq_mask.squeeze(0)

            prefix_len = len(data.req.prefix_indices)
            mask_slice = vq_mask[prefix_len : prefix_len + req_len]

            parts = [p.to(device).T for p in data.vq_parts if p.dim() == 2]
            vq_parts_flat = torch.cat(parts, dim=0) if parts else None

            if vq_parts_flat is not None and mask_slice.any():
                vq_before = vq_mask[:prefix_len].sum().item() if prefix_len > 0 else 0
                num_vq_in_slice = mask_slice.sum().item()
                vq_slice = vq_parts_flat[vq_before : vq_before + num_vq_in_slice]
                req_embeds = text_embeds[offset : offset + req_len]
                vq_embeds = audio_decoder.embed_text_dim(
                    req_embeds.unsqueeze(0), vq_slice, mask_slice.unsqueeze(0),
                )
                mask_indices = mask_slice.nonzero(as_tuple=True)[0] + offset
                text_embeds[mask_indices] = vq_embeds.to(text_embeds.dtype)

        offset += req_len

    model_worker_batch.input_embeds = text_embeds
    # Return None to use standard forward path (input_embeds now set on batch)
    return None


def _find_scheduler_request(req: Any, requests: list) -> Any:
    for sr in requests:
        if hasattr(sr, "data") and hasattr(sr.data, "req") and sr.data.req is req:
            return sr
    return None
