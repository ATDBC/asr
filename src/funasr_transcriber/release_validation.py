"""Completeness contract for Linux release-validation summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import string
from typing import Any


EVIDENCE_FIELDS = (
    "firered_modules",
    "lid_coverage",
    "timestamp_integrity",
    "gpu_stages",
    "deterministic_hashes",
)
FIRERED_MODULES = ("vad", "lid", "aed", "punc")
STRESS_DURATIONS = ("30", "60", "120")


def _valid_modules(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        value.get(name) is True for name in FIRERED_MODULES
    )


def _valid_lid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    sentences = value.get("sentences")
    covered = value.get("covered")
    ratio = value.get("ratio")
    return (
        isinstance(sentences, int)
        and not isinstance(sentences, bool)
        and sentences > 0
        and covered == sentences
        and ratio == 1.0
    )


def _valid_timestamps(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    checked = value.get("checked")
    return (
        isinstance(checked, int)
        and not isinstance(checked, bool)
        and checked > 0
        and value.get("valid") == checked
        and value.get("passed") is True
    )


def _valid_gpu_stages(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for record in value.values():
        if not isinstance(record, Mapping):
            return False
        if not all(
            isinstance(record.get(name), int)
            and not isinstance(record.get(name), bool)
            for name in ("before_free_mib", "after_free_mib")
        ):
            return False
        if record.get("released") is not True:
            return False
    return True


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _valid_hashes(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    for duration in STRESS_DURATIONS:
        hashes = value.get(duration)
        if (
            not isinstance(hashes, Sequence)
            or isinstance(hashes, (str, bytes))
            or len(hashes) != 2
            or not all(_valid_digest(item) for item in hashes)
            or hashes[0] != hashes[1]
        ):
            return False
    return True


def missing_release_evidence(evidence: Mapping[str, Any]) -> list[str]:
    validators = {
        "firered_modules": _valid_modules,
        "lid_coverage": _valid_lid,
        "timestamp_integrity": _valid_timestamps,
        "gpu_stages": _valid_gpu_stages,
        "deterministic_hashes": _valid_hashes,
    }
    return [
        name
        for name in EVIDENCE_FIELDS
        if not validators[name](evidence.get(name))
    ]


def build_release_summary(
    *,
    exit_code: int,
    stress_skipped: bool,
    last_stage: str,
    stage_results: Mapping[str, Any],
    release_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    missing = missing_release_evidence(release_evidence)
    successful = (
        exit_code == 0
        and not stress_skipped
        and last_stage == "complete"
        and not missing
    )
    return {
        "passed": successful,
        "complete": successful,
        "stress_skipped": stress_skipped,
        "exit_code": exit_code,
        "last_stage": last_stage,
        "missing_evidence": missing,
        "release_evidence": dict(release_evidence),
        "stage_results": dict(stage_results),
    }
