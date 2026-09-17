from __future__ import annotations

import json
from pathlib import Path

import pytest

import funasr_transcriber.weighted_aed_calibration as calibration


def _segments(count: int = 32) -> list[dict[str, int]]:
    return [
        {
            "index": index,
            "start_ms": index * 10_000,
            "end_ms": index * 10_000 + (index + 1) * 100,
        }
        for index in range(count)
    ]


def _source(tmp_path: Path, index: int, count: int = 32):
    audio = tmp_path / f"source-{index}.wav"
    audio.write_bytes(f"audio-{index}".encode())
    segments = _segments(count)
    return calibration.CalibrationSource(
        source_recording_group=f"source-{index}",
        audio_path=audio,
        source_checksum=calibration.sha256_file(audio),
        segments_checksum=calibration.canonical_sha256(segments),
        segments=tuple(segments),
    )


def _records() -> list[dict[str, object]]:
    records = []
    for group_index in range(10):
        for count in (1, 2, 4, 8, 16, 32):
            longest = 1.0 + group_index / 10
            actual = longest * (0.2 + 0.01 * count)
            records.append(
                {
                    "status": "complete",
                    "source_recording_group": f"source-{group_index}",
                    "item_count": count,
                    "max_duration_seconds": longest,
                    "repeat_seconds": [actual, actual, actual],
                    "median_actual_seconds": actual,
                    "output_fingerprints": ["same", "same", "same"],
                }
            )
    return records


def _runtime() -> dict[str, object]:
    return {
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "gpu_name": "NVIDIA Test GPU",
        "compute_capability": "8.6",
        "driver_version": "580.1",
        "cuda_version": "12.8",
        "torch_version": "2.8",
        "firered_source_commit": "f" * 40,
        "aed_checkpoint_sha256": "a" * 64,
    }


def test_load_sources_resolves_relative_audio_and_checks_checksums(tmp_path: Path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio = audio_root / "sample.wav"
    audio.write_bytes(b"sample")
    segments = _segments(2)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "metadata": {
                    "source_inventory": [
                        {
                            "audio_path": "sample.wav",
                            "source_recording_group": "source-a",
                            "source_checksum": calibration.sha256_file(audio),
                            "segments_checksum": calibration.canonical_sha256(
                                segments
                            ),
                            "segments": segments,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    sources = calibration.load_calibration_sources(manifest, audio_root)

    assert sources[0].audio_path == audio.resolve()
    assert sources[0].segments == tuple(segments)


def test_build_cases_uses_source_local_short_middle_long_windows(tmp_path: Path):
    source = _source(tmp_path, 0)

    cases = calibration.build_calibration_cases((source,))

    assert {case.item_count for case in cases} == {1, 2, 4, 8, 16, 32}
    assert all(case.source_recording_group == "source-0" for case in cases)
    assert all(case.total_duration_seconds <= 200.0 for case in cases)


def test_validate_coverage_requires_large_batches_and_ten_sources(tmp_path: Path):
    too_small = tuple(_source(tmp_path, index, count=8) for index in range(10))

    with pytest.raises(ValueError, match="16"):
        calibration.validate_calibration_coverage(
            calibration.build_calibration_cases(too_small)
        )


def test_assign_grouped_folds_keeps_each_source_in_one_fold():
    groups = [f"source-{index}" for index in range(10) for _ in range(3)]

    folds = calibration.assign_grouped_folds(groups)

    assert set(folds) == set(groups)
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_evaluate_calibration_recovers_two_coefficients():
    weights, report = calibration.evaluate_calibration(_records())

    assert weights.longest_weight == pytest.approx(0.2)
    assert weights.padded_weight == pytest.approx(0.01)
    assert report["repeat_noise"]["status"] == "stable"
    assert report["grouped_folds"] == 5
    assert report["status"] == "ready"


def test_evaluate_calibration_separates_repeat_noise_from_fit():
    records = _records()
    for record in records:
        actual = float(record["median_actual_seconds"])
        record["repeat_seconds"] = [actual, actual, actual * 2]

    _, report = calibration.evaluate_calibration(records)

    assert report["repeat_noise"]["status"] == "measurement_unstable"
    assert report["status"] == "measurement_unstable"


def test_evaluate_calibration_rejects_non_positive_coefficients():
    records = _records()
    for record in records:
        longest = float(record["max_duration_seconds"])
        count = int(record["item_count"])
        actual = longest * (1.0 - 0.01 * count)
        record["repeat_seconds"] = [actual, actual, actual]
        record["median_actual_seconds"] = actual

    with pytest.raises(ValueError, match="positive"):
        calibration.evaluate_calibration(records)


def test_artifact_loader_warns_for_versions_and_falls_back_for_gpu(tmp_path: Path):
    runtime = _runtime()
    payload = {
        "schema_version": 1,
        "kind": "weighted_aed_device_calibration",
        "status": "ready",
        "formula": (
            "max_duration_seconds*(longest_weight+padded_weight*item_count)"
        ),
        "weights": {"longest_weight": 0.2, "padded_weight": 0.01},
        "runtime_identity": {**runtime, "driver_version": "older"},
        "evidence": {
            "manifest_checksum": "1" * 64,
            "plan_checksum": "2" * 64,
            "records_checksum": "3" * 64,
            "source_groups": 10,
            "batch_cases": 60,
            "repeats": 3,
        },
        "fit": {"feature_support": {"item_count": [1, 32]}},
    }
    artifact = tmp_path / "weights.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    selected = calibration.load_weight_artifact(artifact, runtime)
    assert selected.source == "artifact"
    assert selected.weights == calibration.WeightedAedWeights(0.2, 0.01)
    assert any("driver_version" in warning for warning in selected.warnings)

    mismatched = {**runtime, "gpu_name": "Different GPU"}
    fallback = calibration.load_weight_artifact(artifact, mismatched)
    assert fallback.source == "builtin_fallback"
    assert fallback.weights == calibration.DEFAULT_WEIGHTED_AED_WEIGHTS
    assert any("gpu_name" in warning for warning in fallback.warnings)


def test_build_ready_artifact_rejects_unstable_report():
    with pytest.raises(ValueError, match="ready"):
        calibration.build_ready_artifact(
            weights=calibration.WeightedAedWeights(0.2, 0.01),
            report={"status": "measurement_unstable"},
            runtime_identity=_runtime(),
            manifest_checksum="1" * 64,
            plan_checksum="2" * 64,
            records_checksum="3" * 64,
            gpu_start={"free_mib": 20_000},
            gpu_end={"free_mib": 20_000},
        )


def test_build_expected_runtime_identity_hashes_target_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "models" / "fireredasr2s"
    checkpoint = root / "FireRedASR2-AED" / "model.pth.tar"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (root / "source").mkdir()
    monkeypatch.setattr(calibration, "_source_head", lambda path: "f" * 40)
    monkeypatch.setattr(
        calibration,
        "_torch_runtime",
        lambda: {
            "compute_capability": "8.6",
            "torch_version": "2.8",
            "cuda_version": "12.8",
        },
    )

    identity = calibration.build_expected_runtime_identity(
        tmp_path / "models",
        gpu_name="NVIDIA Test GPU",
        driver_version="580.1",
        use_half=True,
    )

    assert identity["device"] == "cuda:0"
    assert identity["dtype"] == "torch.bfloat16"
    assert identity["aed_checkpoint_sha256"] == calibration.sha256_file(
        checkpoint
    )
