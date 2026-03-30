# SPDX-License-Identifier: Apache-2.0
"""SimpleScheduler — lightweight scheduler for non-AR stages.

For stages that just run a function (preprocessing, encoders, decode, code2wav).
No KV cache, no batching. Just: inbox.get() → run function → outbox.put().

Same inbox/outbox interface as OmniScheduler so Stage doesn't need branching.
"""
from __future__ import annotations

import asyncio
import logging
import queue as _queue_mod
from typing import Any, Callable

from sglang_omni.scheduling.omni_scheduler import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


class SimpleScheduler:
    """Process requests one at a time via a callable.

    Supports sync and async callables. For streaming consumers (code2wav),
    the callable receives stream chunks via the inbox with type="stream_chunk".
    """

    def __init__(self, compute_fn: Callable):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._fn = compute_fn
        self._running = False

    def start(self) -> None:
        """Run the processing loop (blocks the thread)."""
        self._running = True
        loop = asyncio.new_event_loop()
        try:
            while self._running:
                try:
                    msg = self.inbox.get(timeout=0.1)
                except _queue_mod.Empty:
                    continue

                if msg.type == "new_request":
                    try:
                        result = self._fn(msg.data)
                        if asyncio.iscoroutine(result):
                            result = loop.run_until_complete(result)
                        self.outbox.put(OutgoingMessage(
                            request_id=msg.request_id,
                            type="result",
                            data=result,
                        ))
                    except Exception:
                        logger.exception(
                            "SimpleScheduler: compute_fn failed for %s", msg.request_id
                        )
        finally:
            loop.close()

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        pass  # Simple scheduler doesn't track request state
