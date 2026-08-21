"""Pure helpers shared by the FireRed segment-batch calibration command."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any


def stable_response_fingerprint(response: Mapping[str, object]) -> str:
    """Hash only output fields whose equality defines a stable ASR result."""

    payload = {
        "text": response["text"],
        "tokens": response["tokens"],
        "sentences": response["sentences"],
        "vad_segments_ms": response["vad_segments_ms"],
        "duration_ms": response["duration_ms"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one timing value is required")
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_stage_calls(
    calls: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float | int]]:
    """Return deterministic timing statistics for each instrumented stage."""

    summary: dict[str, dict[str, float | int]] = {}
    for name, values in calls.items():
        if not values:
            raise ValueError(f"stage {name!r} has no calls")
        numeric = [float(value) for value in values]
        summary[str(name)] = {
            "calls": len(numeric),
            "total_seconds": sum(numeric),
            "median_call_seconds": float(median(numeric)),
            "p95_call_seconds": round(_p95(numeric), 6),
        }
    return summary


def _candidate_passes(candidate: Mapping[str, object]) -> bool:
    if any(
        candidate.get(name) is True
        for name in (
            "failed",
            "oom",
            "out_of_memory",
            "timeout",
            "timed_out",
        )
    ):
        return False
    baseline = candidate.get("baseline_fingerprint")
    fingerprints = candidate.get("fingerprints")
    if not isinstance(baseline, str) or not baseline:
        return False
    if (
        not isinstance(fingerprints, Sequence)
        or isinstance(fingerprints, (str, bytes))
        or not fingerprints
        or any(value != baseline for value in fingerprints)
    ):
        return False
    repetitions = candidate.get("repeats")
    successful = candidate.get("successful_repetitions")
    if repetitions is not None and successful != repetitions:
        return False
    try:
        peak = float(candidate["peak_process_mib"])
        maximum = float(candidate["maximum_process_mib"])
        timings = [float(value) for value in candidate["end_to_end_seconds"]]  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(timings) and peak <= maximum


def choose_batch_candidate(candidates: Sequence[Mapping[str, object]]) -> int:
    """Choose the fastest valid batch, preferring smaller batches on ties."""

    passing: list[tuple[float, int]] = []
    for candidate in candidates:
        if not _candidate_passes(candidate):
            continue
        try:
            batch_size = int(candidate["batch_size"])
            timings = [float(value) for value in candidate["end_to_end_seconds"]]  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            continue
        if batch_size <= 0:
            continue
        passing.append((float(median(timings)), batch_size))
    if not passing:
        return 1
    return min(passing)[1]
