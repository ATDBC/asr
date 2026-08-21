from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from funasr_transcriber.firered_benchmark import (
    choose_batch_candidate,
    stable_response_fingerprint,
    summarize_stage_calls,
)


def _response(*, token_end: int = 900) -> dict[str, object]:
    return {
        "text": "你好。",
        "tokens": [
            {"text": "你好。", "start_ms": 100, "end_ms": token_end, "confidence": 0.9}
        ],
        "sentences": [
            {"text": "你好。", "start_ms": 100, "end_ms": token_end, "asr_confidence": 0.9}
        ],
        "vad_segments_ms": [[100, token_end]],
        "duration_ms": 1_000,
    }


def test_stable_response_fingerprint_includes_timestamped_output():
    assert stable_response_fingerprint(_response()) != stable_response_fingerprint(
        _response(token_end=901)
    )


def test_summarize_stage_calls_reports_median_and_interpolated_p95():
    summary = summarize_stage_calls({"vad": [1.0, 3.0, 2.0, 4.0]})

    assert summary == {
        "vad": {
            "calls": 4,
            "total_seconds": 10.0,
            "median_call_seconds": 2.5,
            "p95_call_seconds": 3.85,
        }
    }


def test_choose_batch_candidate_requires_matching_successful_repetitions():
    candidates = [
        {
            "batch_size": 1,
            "baseline_fingerprint": "base",
            "fingerprints": ["base", "base", "base"],
            "end_to_end_seconds": [10.0, 9.0, 11.0],
            "peak_process_mib": 100,
            "maximum_process_mib": 200,
        },
        {
            "batch_size": 2,
            "baseline_fingerprint": "base",
            "fingerprints": ["base", "wrong", "base"],
            "end_to_end_seconds": [5.0, 5.0, 5.0],
            "peak_process_mib": 100,
            "maximum_process_mib": 200,
        },
        {
            "batch_size": 4,
            "baseline_fingerprint": "base",
            "fingerprints": ["base", "base", "base"],
            "end_to_end_seconds": [4.0, 4.0, 4.0],
            "peak_process_mib": 250,
            "maximum_process_mib": 200,
        },
        {
            "batch_size": 8,
            "baseline_fingerprint": "base",
            "fingerprints": ["base", "base", "base"],
            "end_to_end_seconds": [3.0, 3.0, 3.0],
            "peak_process_mib": 100,
            "maximum_process_mib": 200,
            "failed": True,
        },
        {
            "batch_size": 16,
            "baseline_fingerprint": "base",
            "fingerprints": ["base", "base", "base"],
            "end_to_end_seconds": [2.0, 2.0, 2.0],
            "peak_process_mib": 100,
            "maximum_process_mib": 200,
            "timeout": True,
        },
    ]

    assert choose_batch_candidate(candidates) == 1


def test_choose_batch_candidate_breaks_equal_medians_by_lower_batch_and_falls_back():
    candidates = [
        {
            "batch_size": 4,
            "baseline_fingerprint": "base",
            "fingerprints": ["base", "base", "base"],
            "end_to_end_seconds": [2.0, 2.0, 2.0],
            "peak_process_mib": 100,
            "maximum_process_mib": 200,
        },
        {
            "batch_size": 2,
            "baseline_fingerprint": "base",
            "fingerprints": ["base", "base", "base"],
            "end_to_end_seconds": [2.0, 2.0, 2.0],
            "peak_process_mib": 100,
            "maximum_process_mib": 200,
        },
    ]

    assert choose_batch_candidate(candidates) == 2
    assert choose_batch_candidate([]) == 1


def test_calibration_cli_rejects_non_physical_gpu_zero_before_loading_models():
    root = Path(__file__).parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "benchmark_firered_batches.py"),
            "--audio",
            str(root / "missing.wav"),
            "--model-dir",
            str(root / "models"),
            "--output",
            str(root / "outputs" / "measurements" / "ignored.json"),
            "--device",
            "cuda:1",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 2
    assert "cuda:0" in completed.stderr
