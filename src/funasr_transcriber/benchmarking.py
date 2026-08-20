"""Deterministic manifests used by release-only benchmarks."""

from __future__ import annotations

import random
from typing import Any

from .metrics import cpcer


def build_overlap_manifest(
    sources: list[dict[str, Any]],
    count: int = 50,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    speakers = {str(item["speaker"]) for item in sources}
    if len(speakers) < 2:
        raise ValueError("overlap benchmark requires at least two speakers")
    randomizer = random.Random(seed)
    output = []
    snr_values = (-5, 0, 5)
    for index in range(count):
        first = randomizer.choice(sources)
        alternatives = [
            item
            for item in sources
            if str(item["speaker"]) != str(first["speaker"])
        ]
        second = randomizer.choice(alternatives)
        maximum = min(
            8,
            int(float(first["duration_seconds"])),
            int(float(second["duration_seconds"])),
        )
        if maximum < 1:
            raise ValueError("source utterances must be at least one second")
        output.append(
            {
                "id": f"overlap-{index + 1:04d}",
                "first": dict(first),
                "second": dict(second),
                "snr_db": snr_values[index % len(snr_values)],
                "overlap_seconds": 1 + (index % maximum),
            }
        )
    return output


def stress_durations_minutes() -> tuple[int, int, int]:
    return 30, 60, 120


def input_format_matrix() -> list[dict[str, int | str]]:
    return [
        {
            "extension": extension,
            "sample_rate": sample_rate,
            "channels": channels,
        }
        for extension in (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")
        for sample_rate in (8000, 16000, 44100, 48000)
        for channels in (1, 2)
    ]


def _text_by_speaker(turns: list[dict[str, Any]]) -> dict[Any, str]:
    output: dict[Any, str] = {}
    for turn in turns:
        speaker = turn.get("speaker") or turn.get("speaker_id")
        if speaker is None:
            continue
        output[speaker] = output.get(speaker, "") + str(turn.get("text") or "")
    return output


def score_overlap_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    synthetic = [item for item in cases if item.get("kind") == "synthetic"]
    real = [item for item in cases if item.get("kind") == "real_overlap"]
    negative = [
        item
        for item in cases
        if item.get("kind") == "speaker_change_negative"
    ]
    recovered = 0
    for item in synthetic:
        references = _text_by_speaker(item.get("reference_turns") or [])
        hypotheses = _text_by_speaker(item.get("hypothesis_turns") or [])
        if (
            len(references) == 2
            and len(hypotheses) >= 2
            and cpcer(references, hypotheses) <= 0.5
        ):
            recovered += 1
    synthetic_rate = recovered / len(synthetic) if synthetic else 0.0
    false_positives = sum(
        any(bool(turn.get("overlap")) for turn in item.get("hypothesis_turns") or [])
        for item in negative
    )
    false_positive_rate = false_positives / len(negative) if negative else 0.0
    synthetic_passed = bool(synthetic) and synthetic_rate >= 0.60
    negative_passed = bool(negative) and false_positive_rate <= 0.02
    return {
        "synthetic": {
            "total": len(synthetic),
            "recovered": recovered,
            "recovery_rate": synthetic_rate,
            "passed": synthetic_passed,
        },
        "real_overlap": {
            "total": len(real),
            "overlap_marked": sum(
                any(
                    bool(turn.get("overlap"))
                    for turn in item.get("hypothesis_turns") or []
                )
                for item in real
            ),
            "blocking": False,
        },
        "speaker_change_negative": {
            "total": len(negative),
            "false_positives": false_positives,
            "false_positive_rate": false_positive_rate,
            "passed": negative_passed,
        },
        "passed": synthetic_passed and negative_passed,
    }
