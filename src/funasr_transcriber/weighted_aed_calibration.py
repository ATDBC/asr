"""Device-specific weights for deterministic weighted AED scheduling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_KIND = "weighted_aed_device_calibration"
FORMULA_ID = (
    "max_duration_seconds*(longest_weight+padded_weight*item_count)"
)
TARGET_ITEM_COUNTS = (1, 2, 4, 8, 16, 32)
MAX_BATCH_DURATION_MS = 200_000
STRICT_RUNTIME_KEYS = (
    "device",
    "dtype",
    "gpu_name",
    "compute_capability",
    "firered_source_commit",
    "aed_checkpoint_sha256",
)
WARNING_RUNTIME_KEYS = ("driver_version", "cuda_version", "torch_version")


@dataclass(frozen=True, slots=True)
class WeightedAedWeights:
    longest_weight: float
    padded_weight: float

    def __post_init__(self) -> None:
        values = (float(self.longest_weight), float(self.padded_weight))
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "weighted AED coefficients must be finite and positive"
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "longest_weight": float(self.longest_weight),
            "padded_weight": float(self.padded_weight),
        }


DEFAULT_WEIGHTED_AED_WEIGHTS = WeightedAedWeights(
    longest_weight=0.12854663332839417,
    padded_weight=0.008409640427755745,
)


@dataclass(frozen=True, slots=True)
class CalibrationSource:
    source_recording_group: str
    audio_path: Path
    source_checksum: str
    segments_checksum: str
    segments: tuple[dict[str, int], ...]


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    case_id: str
    source_recording_group: str
    audio_path: Path
    segments: tuple[dict[str, int], ...]
    item_count: int
    max_duration_seconds: float
    total_duration_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "source_recording_group": self.source_recording_group,
            "audio_path": str(self.audio_path),
            "segments": list(self.segments),
            "item_count": self.item_count,
            "max_duration_seconds": self.max_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class WeightSelection:
    weights: WeightedAedWeights
    source: str
    artifact_checksum: str | None
    warnings: tuple[str, ...]


def canonical_sha256(payload: object) -> str:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a SHA-256 checksum")
    return text


def _normalize_segments(raw_segments: object) -> tuple[dict[str, int], ...]:
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("source segments must be a non-empty list")
    normalized: list[dict[str, int]] = []
    indexes: set[int] = set()
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            raise ValueError("source segment must be an object")
        segment = {
            "index": int(raw["index"]),
            "start_ms": int(raw["start_ms"]),
            "end_ms": int(raw["end_ms"]),
        }
        if segment["index"] in indexes:
            raise ValueError("source segment indexes must be unique")
        if segment["start_ms"] < 0 or segment["end_ms"] <= segment["start_ms"]:
            raise ValueError("source segment bounds are invalid")
        if segment["end_ms"] - segment["start_ms"] > MAX_BATCH_DURATION_MS:
            raise ValueError("source segment exceeds 200 seconds")
        indexes.add(segment["index"])
        normalized.append(segment)
    return tuple(normalized)


def load_calibration_sources(
    manifest_path: Path,
    audio_root: Path | None = None,
) -> tuple[CalibrationSource, ...]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    inventory = metadata.get("source_inventory") if isinstance(metadata, Mapping) else None
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("manifest metadata.source_inventory must be non-empty")
    root = None if audio_root is None else Path(audio_root).expanduser().resolve()
    sources: list[CalibrationSource] = []
    groups: set[str] = set()
    for raw in inventory:
        if not isinstance(raw, Mapping):
            raise ValueError("source inventory entry must be an object")
        group = str(raw.get("source_recording_group") or "").strip()
        if not group or group in groups:
            raise ValueError("source_recording_group must be non-empty and unique")
        audio_path = Path(str(raw.get("audio_path") or "")).expanduser()
        if not audio_path.is_absolute():
            if root is None:
                raise ValueError("--audio-root is required for relative audio paths")
            audio_path = root / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise ValueError(f"calibration audio does not exist: {audio_path}")
        source_checksum = _sha256(raw.get("source_checksum"), "source checksum")
        if sha256_file(audio_path) != source_checksum:
            raise ValueError(f"source checksum mismatch: {group}")
        segments = _normalize_segments(raw.get("segments"))
        segments_checksum = _sha256(
            raw.get("segments_checksum"), "segments checksum"
        )
        if canonical_sha256(list(segments)) != segments_checksum:
            raise ValueError(f"segments checksum mismatch: {group}")
        groups.add(group)
        sources.append(
            CalibrationSource(
                source_recording_group=group,
                audio_path=audio_path,
                source_checksum=source_checksum,
                segments_checksum=segments_checksum,
                segments=segments,
            )
        )
    return tuple(sorted(sources, key=lambda item: item.source_recording_group))


def build_calibration_cases(
    sources: Sequence[CalibrationSource],
) -> tuple[CalibrationCase, ...]:
    cases: list[CalibrationCase] = []
    for source in sorted(sources, key=lambda item: item.source_recording_group):
        ordered = sorted(
            source.segments,
            key=lambda item: (item["end_ms"] - item["start_ms"], item["index"]),
        )
        for count in TARGET_ITEM_COUNTS:
            if len(ordered) < count:
                continue
            offsets = (0, (len(ordered) - count) // 2, len(ordered) - count)
            seen: set[tuple[int, ...]] = set()
            for offset in offsets:
                selected = tuple(dict(item) for item in ordered[offset:offset + count])
                signature = tuple(item["index"] for item in selected)
                if signature in seen:
                    continue
                seen.add(signature)
                durations = [item["end_ms"] - item["start_ms"] for item in selected]
                total_ms = sum(durations)
                if total_ms > MAX_BATCH_DURATION_MS:
                    continue
                case_id = canonical_sha256(
                    {
                        "source_recording_group": source.source_recording_group,
                        "segments": list(selected),
                    }
                )
                cases.append(
                    CalibrationCase(
                        case_id=case_id,
                        source_recording_group=source.source_recording_group,
                        audio_path=source.audio_path,
                        segments=selected,
                        item_count=count,
                        max_duration_seconds=max(durations) / 1000.0,
                        total_duration_seconds=total_ms / 1000.0,
                    )
                )
    return tuple(
        sorted(
            cases,
            key=lambda item: (
                item.source_recording_group,
                item.item_count,
                item.max_duration_seconds,
                item.case_id,
            ),
        )
    )


def validate_calibration_coverage(
    cases: Sequence[CalibrationCase],
) -> dict[str, object]:
    groups = {case.source_recording_group for case in cases}
    counts = {case.item_count for case in cases}
    if len(groups) < 10:
        raise ValueError("calibration requires at least 10 source groups")
    if len(counts) < 4 or 1 not in counts:
        raise ValueError("calibration requires at least four item counts including 1")
    if max(counts, default=0) < 16:
        raise ValueError("calibration requires an item count of at least 16")
    matrix = np.asarray(
        [
            [case.max_duration_seconds, case.max_duration_seconds * case.item_count]
            for case in cases
        ],
        dtype=np.float64,
    )
    if np.linalg.matrix_rank(matrix) != 2:
        raise ValueError("weighted AED calibration feature matrix must have rank 2")
    return {
        "source_groups": len(groups),
        "item_counts": sorted(counts),
        "batch_cases": len(cases),
        "matrix_rank": 2,
    }


def assign_grouped_folds(
    groups: Sequence[str],
    fold_count: int = 5,
) -> dict[str, int]:
    unique = sorted(
        {str(group) for group in groups},
        key=lambda group: (hashlib.sha256(group.encode()).hexdigest(), group),
    )
    if fold_count < 2 or len(unique) < fold_count:
        raise ValueError("grouped folds require at least one source per fold")
    return {group: index % fold_count for index, group in enumerate(unique)}


def _fit(rows: Sequence[Mapping[str, object]]) -> WeightedAedWeights:
    matrix = np.asarray(
        [
            [
                float(row["max_duration_seconds"]),
                float(row["max_duration_seconds"]) * int(row["item_count"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    labels = np.asarray(
        [float(row["median_actual_seconds"]) for row in rows],
        dtype=np.float64,
    )
    if np.linalg.matrix_rank(matrix) != 2:
        raise ValueError("weighted AED calibration feature matrix must have rank 2")
    coefficients, _, _, _ = np.linalg.lstsq(matrix, labels, rcond=None)
    return WeightedAedWeights(*map(float, coefficients))


def _error_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    absolute = np.abs(actual - predicted)
    percentages = absolute / actual * 100.0
    return {
        "mae_seconds": float(np.mean(absolute)),
        "median_ape_pct": float(np.median(percentages)),
        "p95_ape_pct": float(np.percentile(percentages, 95)),
    }


def evaluate_calibration(
    records: Sequence[Mapping[str, object]],
) -> tuple[WeightedAedWeights, dict[str, object]]:
    if not records or any(record.get("status") != "complete" for record in records):
        raise ValueError("calibration records must all be complete")
    rows: list[dict[str, object]] = []
    deviations: list[float] = []
    for record in records:
        fingerprints = [str(item) for item in record.get("output_fingerprints", [])]
        if fingerprints and len(set(fingerprints)) != 1:
            raise ValueError("calibration output fingerprints do not match")
        repeats = np.asarray(record.get("repeat_seconds"), dtype=np.float64)
        if repeats.size < 2 or not np.all(np.isfinite(repeats)) or np.any(repeats <= 0):
            raise ValueError("calibration repeat timings are invalid")
        median = float(np.median(repeats))
        deviations.extend((np.abs(repeats - median) / median * 100.0).tolist())
        rows.append({**record, "median_actual_seconds": median})
    groups = {str(row["source_recording_group"]) for row in rows}
    counts = {int(row["item_count"]) for row in rows}
    if len(groups) < 10:
        raise ValueError("calibration requires at least 10 source groups")
    if len(counts) < 4 or 1 not in counts or max(counts) < 16:
        raise ValueError("calibration item count coverage is insufficient")

    fold_count = 5
    folds = assign_grouped_folds([str(row["source_recording_group"]) for row in rows])
    actual_values: list[float] = []
    predicted_values: list[float] = []
    by_count: dict[int, list[tuple[float, float]]] = {}
    for fold in range(fold_count):
        training = [row for row in rows if folds[str(row["source_recording_group"])] != fold]
        held_out = [row for row in rows if folds[str(row["source_recording_group"])] == fold]
        weights = _fit(training)
        for row in held_out:
            actual = float(row["median_actual_seconds"])
            predicted = float(row["max_duration_seconds"]) * (
                weights.longest_weight + weights.padded_weight * int(row["item_count"])
            )
            actual_values.append(actual)
            predicted_values.append(predicted)
            by_count.setdefault(int(row["item_count"]), []).append((actual, predicted))
    actual = np.asarray(actual_values, dtype=np.float64)
    predicted = np.asarray(predicted_values, dtype=np.float64)
    noise_median = float(np.median(deviations))
    noise_p95 = float(np.percentile(deviations, 95))
    stable = noise_median <= 5.0 and noise_p95 <= 20.0
    final_weights = _fit(rows)
    report: dict[str, object] = {
        "status": "ready" if stable else "measurement_unstable",
        "weights": final_weights.as_dict(),
        "grouped_folds": fold_count,
        "repeat_noise": {
            "median_deviation_pct": noise_median,
            "p95_deviation_pct": noise_p95,
            "thresholds": {"median_pct": 5.0, "p95_pct": 20.0},
            "status": "stable" if stable else "measurement_unstable",
        },
        "oof_error": _error_summary(actual, predicted),
        "oof_by_item_count": {
            str(count): _error_summary(
                np.asarray([item[0] for item in values]),
                np.asarray([item[1] for item in values]),
            )
            for count, values in sorted(by_count.items())
        },
        "feature_support": {
            "item_count": [min(counts), max(counts)],
            "max_duration_seconds": [
                min(float(row["max_duration_seconds"]) for row in rows),
                max(float(row["max_duration_seconds"]) for row in rows),
            ],
        },
        "counts": {
            "source_groups": len(groups),
            "batch_cases": len(rows),
            "repeats": len(rows[0]["repeat_seconds"]),
        },
    }
    return final_weights, report


def build_ready_artifact(
    *,
    weights: WeightedAedWeights,
    report: Mapping[str, object],
    runtime_identity: Mapping[str, object],
    manifest_checksum: str,
    plan_checksum: str,
    records_checksum: str,
    gpu_start: Mapping[str, object],
    gpu_end: Mapping[str, object],
) -> dict[str, object]:
    if report.get("status") != "ready":
        raise ValueError("calibration report must be ready")
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("calibration report counts are missing")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "status": "ready",
        "formula": FORMULA_ID,
        "weights": weights.as_dict(),
        "runtime_identity": dict(runtime_identity),
        "evidence": {
            "manifest_checksum": _sha256(manifest_checksum, "manifest checksum"),
            "plan_checksum": _sha256(plan_checksum, "plan checksum"),
            "records_checksum": _sha256(records_checksum, "records checksum"),
            "source_groups": int(counts["source_groups"]),
            "batch_cases": int(counts["batch_cases"]),
            "repeats": int(counts["repeats"]),
        },
        "fit": dict(report),
        "gpu0": {"start": dict(gpu_start), "end": dict(gpu_end)},
    }


def _fallback(message: str) -> WeightSelection:
    return WeightSelection(
        weights=DEFAULT_WEIGHTED_AED_WEIGHTS,
        source="builtin_fallback",
        artifact_checksum=None,
        warnings=(message,),
    )


def load_weight_artifact(
    path: Path,
    expected_runtime_identity: Mapping[str, object],
) -> WeightSelection:
    try:
        path = Path(path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("weighted AED artifact must be an object")
        if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("weighted AED artifact schema_version mismatch")
        if payload.get("kind") != ARTIFACT_KIND or payload.get("status") != "ready":
            raise ValueError("weighted AED artifact kind/status mismatch")
        if payload.get("formula") != FORMULA_ID:
            raise ValueError("weighted AED artifact formula mismatch")
        raw_weights = payload.get("weights")
        if not isinstance(raw_weights, Mapping):
            raise ValueError("weighted AED artifact weights are missing")
        weights = WeightedAedWeights(
            float(raw_weights["longest_weight"]),
            float(raw_weights["padded_weight"]),
        )
        runtime = payload.get("runtime_identity")
        if not isinstance(runtime, Mapping):
            raise ValueError("weighted AED artifact runtime identity is missing")
        for key in STRICT_RUNTIME_KEYS:
            if runtime.get(key) != expected_runtime_identity.get(key):
                raise ValueError(f"weighted AED artifact {key} mismatch")
        warnings = tuple(
            f"weighted AED artifact {key} differs: "
            f"artifact={runtime.get(key)!r}, runtime={expected_runtime_identity.get(key)!r}"
            for key in WARNING_RUNTIME_KEYS
            if runtime.get(key) != expected_runtime_identity.get(key)
        )
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("weighted AED artifact evidence is missing")
        for key in ("manifest_checksum", "plan_checksum", "records_checksum"):
            _sha256(evidence.get(key), key)
        for key in ("source_groups", "batch_cases", "repeats"):
            if int(evidence.get(key) or 0) <= 0:
                raise ValueError(f"weighted AED artifact {key} is invalid")
        fit = payload.get("fit")
        if not isinstance(fit, Mapping) or not isinstance(
            fit.get("feature_support"), Mapping
        ):
            raise ValueError("weighted AED artifact feature support is missing")
        return WeightSelection(
            weights=weights,
            source="artifact",
            artifact_checksum=sha256_file(path),
            warnings=warnings,
        )
    except Exception as exc:
        return _fallback(f"{type(exc).__name__}: {exc}")


def _source_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "FireRed source git query failed")
    return completed.stdout.strip()


def _torch_runtime() -> dict[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "compute_capability": f"{major}.{minor}",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }


def build_expected_runtime_identity(
    model_dir: Path,
    *,
    gpu_name: str,
    driver_version: str,
    use_half: bool,
) -> dict[str, object]:
    root = Path(model_dir).expanduser().resolve() / "fireredasr2s"
    runtime = _torch_runtime()
    return {
        "device": "cuda:0",
        "dtype": "torch.bfloat16" if use_half else "torch.float32",
        "gpu_name": str(gpu_name),
        "compute_capability": runtime["compute_capability"],
        "driver_version": str(driver_version),
        "cuda_version": runtime["cuda_version"],
        "torch_version": runtime["torch_version"],
        "firered_source_commit": _source_head(root / "source"),
        "aed_checkpoint_sha256": sha256_file(
            root / "FireRedASR2-AED" / "model.pth.tar"
        ),
    }
