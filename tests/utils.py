# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities — model-agnostic helpers for launching and managing servers."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from benchmarks.benchmarker.utils import wait_for_service

STARTUP_TIMEOUT = 600


@contextmanager
def disable_proxy() -> Generator[None, None, None]:
    """Temporarily disable proxy env vars for loopback requests."""
    proxy_vars = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    )
    saved_env = {k: os.environ[k] for k in proxy_vars if k in os.environ}
    for k in proxy_vars:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k in proxy_vars:
            os.environ.pop(k, None)
        os.environ.update(saved_env)


def find_free_port() -> int:
    """Find and return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
        socket_handle.bind(("", 0))
        return socket_handle.getsockname()[1]


def no_proxy_env() -> dict[str, str]:
    """Return a copy of os.environ with proxy variables removed, for subprocess use."""
    proxy_keys = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {k: v for k, v in os.environ.items() if k.lower() not in proxy_keys}


def start_server(
    model_path: str,
    config_path: str,
    log_file: Path,
    port: int,
    timeout: int = STARTUP_TIMEOUT,
) -> subprocess.Popen:
    """Start a server and wait until healthy."""
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        model_path,
        "--config",
        config_path,
        "--port",
        str(port),
    ]
    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    try:
        with disable_proxy():
            wait_for_service(
                f"http://localhost:{port}",
                timeout=timeout,
                server_process=proc,
                server_log_file=log_file,
                health_body_contains="healthy",
            )
    except Exception as exc:
        stop_server(proc)
        log_text = log_file.read_text() if log_file.exists() else ""
        message = str(exc)
        if log_text and log_text not in message:
            message = f"{message}\n{log_text}"
        if isinstance(exc, TimeoutError):
            raise TimeoutError(message) from exc
        if isinstance(exc, RuntimeError):
            raise RuntimeError(message) from exc
        raise
    return proc


def assert_summary_metrics(summary: dict, *, check_tokens: bool = True) -> None:
    """Verify summary-level sanity invariants that must hold for every run."""
    assert (
        summary["failed_requests"] == 0
    ), f"Expected 0 failed requests, got {summary['failed_requests']}"
    assert (
        summary["audio_duration_mean_s"] > 0
    ), f"Expected positive audio duration, got {summary['audio_duration_mean_s']}"
    if check_tokens:
        assert (
            summary.get("gen_tokens_mean", 0) > 0
        ), f"Expected positive gen_tokens_mean, got {summary.get('gen_tokens_mean', 0)}"
        assert (
            summary.get("prompt_tokens_mean", 0) > 0
        ), f"Expected positive prompt_tokens_mean, got {summary.get('prompt_tokens_mean', 0)}"


def assert_per_request_fields(
    per_request: list[dict], *, check_tokens: bool = True
) -> None:
    """Verify every request has valid audio, prompt_tokens, and completion_tokens."""
    for req in per_request:
        rid = req["id"]
        assert req["is_success"], f"Request {rid} failed: {req.get('error')}"
        assert (
            req["audio_duration_s"] is not None and req["audio_duration_s"] > 0
        ), f"Request {rid}: audio_duration_s={req['audio_duration_s']}, expected > 0"
        if check_tokens:
            assert (
                req["prompt_tokens"] is not None and req["prompt_tokens"] > 0
            ), f"Request {rid}: prompt_tokens={req['prompt_tokens']}, expected > 0"
            assert (
                req["completion_tokens"] is not None and req["completion_tokens"] > 0
            ), f"Request {rid}: completion_tokens={req['completion_tokens']}, expected > 0"


def assert_streaming_consistency(
    non_stream_requests: list[dict],
    stream_requests: list[dict],
    *,
    completion_token_rtol: float = 0.10,
    audio_duration_rtol: float = 0.12,
) -> None:
    """Assert per-request metrics are close between streaming and non-streaming."""
    ns_by_id = {r["id"]: r for r in non_stream_requests}
    st_by_id = {r["id"]: r for r in stream_requests}
    assert set(ns_by_id) == set(st_by_id), (
        f"Request ID mismatch: "
        f"non_stream={sorted(ns_by_id)}, stream={sorted(st_by_id)}"
    )
    for rid in sorted(ns_by_id):
        ns, st = ns_by_id[rid], st_by_id[rid]

        ns_ct, st_ct = ns["completion_tokens"], st["completion_tokens"]
        max_ct = max(ns_ct, st_ct)
        assert abs(ns_ct - st_ct) <= completion_token_rtol * max_ct, (
            f"Request {rid}: completion_tokens differ too much — "
            f"non_stream={ns_ct}, stream={st_ct} "
            f"(rtol={completion_token_rtol})"
        )

        assert ns["prompt_tokens"] == st["prompt_tokens"], (
            f"Request {rid}: prompt_tokens mismatch — "
            f"non_stream={ns['prompt_tokens']}, stream={st['prompt_tokens']}"
        )

        ns_ad, st_ad = ns["audio_duration_s"], st["audio_duration_s"]
        max_ad = max(ns_ad, st_ad)
        assert abs(ns_ad - st_ad) <= audio_duration_rtol * max_ad, (
            f"Request {rid}: audio_duration_s differ too much — "
            f"non_stream={ns_ad}, stream={st_ad} "
            f"(rtol={audio_duration_rtol})"
        )


def assert_wer_results(
    results: dict,
    max_corpus_wer: float,
    max_per_sample_wer: float,
) -> None:
    """Verify WER results are within thresholds."""
    summary = results["summary"]
    per_sample = results["per_sample"]

    failed_details = [
        f"  sample {s['id']}: {s.get('error')}"
        for s in per_sample
        if not s.get("is_success", True)
    ]
    assert summary["evaluated"] == summary["total_samples"], (
        f"Only {summary['evaluated']}/{summary['total_samples']} samples evaluated, "
        f"{summary['skipped']} skipped.\n"
        f"Per-sample errors:\n" + "\n".join(failed_details)
    )

    assert summary["wer_corpus"] <= max_corpus_wer, (
        f"Corpus WER {summary['wer_corpus']:.4f} ({summary['wer_corpus'] * 100:.2f}%) "
        f"> threshold {max_corpus_wer} ({max_corpus_wer * 100:.0f}%)"
    )

    assert summary["n_above_50_pct_wer"] == 0, (
        f"{summary['n_above_50_pct_wer']} samples have >50% WER — "
        f"expected 0 catastrophic failures"
    )

    for sample in per_sample:
        assert sample[
            "is_success"
        ], f"Sample {sample['id']} failed: {sample.get('error')}"
        if sample["wer"] is not None:
            assert sample["wer"] <= max_per_sample_wer, (
                f"Sample {sample['id']} WER {sample['wer']:.4f} "
                f"> {max_per_sample_wer}"
            )


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server process group, tolerating already-dead processes."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, ChildProcessError):
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except (ProcessLookupError, ChildProcessError):
            # Process already exited — nothing left to kill.
            return
