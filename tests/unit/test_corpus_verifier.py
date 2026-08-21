import json
import hashlib
from pathlib import Path
import wave

from funasr_transcriber.corpus_verifier import (
    RELEASE_COUNTS,
    verify_corpus,
    write_checksums,
)


def test_release_verifier_requires_firered_evidence_contract(tmp_path: Path):
    write_minimal_corpus(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=RELEASE_COUNTS)

    assert any("release evidence contract" in error for error in result.errors)
    assert any("runner lacks FireRed evidence gates" in error for error in result.errors)


def write_valid_wav(path: Path, duration_ms: int = 1_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * int(duration_ms * 16))


def write_minimal_corpus(root: Path):
    for name in ("format_fixture", "stress_source", "doctor_smoke"):
        write_valid_wav(root / "seeds" / f"{name}.wav")
    (root / "ramc_core" / "WAV").mkdir(parents=True)
    (root / "ramc_core" / "TXT").mkdir(parents=True)
    (root / "ramc_core" / "test.tsv").write_text("", encoding="utf-8")
    (root / "overlap").mkdir(parents=True)
    (root / "overlap" / "manifest.json").write_text("[]\n", encoding="utf-8")
    (root / "SOURCE_NOTICE.txt").write_text("RAMC excerpts\n", encoding="utf-8")
    (root / "run_validation_linux.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "builder_version": "1.0",
        "source_dataset": "MagicData-RAMC/MDT2021S003",
        "seed": 20260820,
        "audio": {"sample_rate": 16000, "channels": 1, "sample_width": 2},
        "counts": {
            "core": 0,
            "synthetic": 0,
            "real_overlap": 0,
            "speaker_change_negative": 0,
            "seeds": 3,
        },
        "total_audio_duration_ms": 3_000,
        "checksum_file": "checksums.sha256",
        "core_test_tsv": "ramc_core/test.tsv",
        "core_items": [],
        "overlap_manifest": "overlap/manifest.json",
        "seeds": {
            name: {
                "audio_path": f"seeds/{name}.wav",
                "sha256": hashlib.sha256(
                    (root / "seeds" / f"{name}.wav").read_bytes()
                ).hexdigest(),
                "duration_ms": 1_000,
                "source": {"wav_name": "source.wav", "start_ms": 0, "end_ms": 1_000},
                "reference_turns": [],
            }
            for name in ("format_fixture", "stress_source", "doctor_smoke")
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_checksums(root)


def test_verify_corpus_accepts_complete_minimal_artifact(tmp_path: Path):
    write_minimal_corpus(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert result.ok is True
    assert result.errors == ()


def test_verify_corpus_detects_checksum_corruption(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    (tmp_path / "SOURCE_NOTICE.txt").write_text("changed", encoding="utf-8")

    result = verify_corpus(tmp_path, expected_counts=None)

    assert result.ok is False
    assert any("SOURCE_NOTICE.txt" in error for error in result.errors)


def test_verify_corpus_ignores_runtime_results_and_service_work(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    write_valid_wav(tmp_path / "validation_results" / "generated.wav")
    (tmp_path / "validation_results" / "summary.json").write_text(
        '{"HF_TOKEN": "runtime-only"}\n', encoding="utf-8"
    )
    (tmp_path / ".service-work" / "service" / "service.log").parent.mkdir(
        parents=True
    )
    (tmp_path / ".service-work" / "service" / "service.log").write_text(
        "TOKEN=runtime-only\n", encoding="utf-8"
    )

    result = verify_corpus(tmp_path, expected_counts=None)

    assert result.ok is True
    assert result.errors == ()


def test_verify_corpus_rejects_absolute_source_path(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text("utf-8"))
    payload["source_path"] = "G:\\MagicData-RAMC"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    write_checksums(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert result.ok is False
    assert any("absolute path" in error for error in result.errors)


def test_verify_corpus_rejects_absolute_path_in_tsv(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    (tmp_path / "ramc_core" / "test.tsv").write_text(
        "G:\\private\\sample.wav\n",
        encoding="utf-8",
    )
    write_checksums(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert result.ok is False
    assert any("ramc_core/test.tsv" in error for error in result.errors)


def test_verify_corpus_rejects_overlapping_negative_reference(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    write_valid_wav(tmp_path / "overlap" / "audio" / "negative.wav", 2_000)
    overlap = [
        {
            "id": "negative-1",
            "kind": "speaker_change_negative",
            "audio_path": "overlap/audio/negative.wav",
            "reference_turns": [
                {"speaker": "S1", "start_ms": 0, "end_ms": 1_500, "text": "甲"},
                {"speaker": "S2", "start_ms": 1_000, "end_ms": 2_000, "text": "乙"},
            ],
        }
    ]
    (tmp_path / "overlap" / "manifest.json").write_text(
        json.dumps(overlap, ensure_ascii=False), encoding="utf-8"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["counts"]["speaker_change_negative"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    write_checksums(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert result.ok is False
    assert any("negative reference overlaps" in error for error in result.errors)


def test_release_verifier_rejects_non_release_counts(tmp_path: Path):
    write_minimal_corpus(tmp_path)

    result = verify_corpus(tmp_path)

    assert result.ok is False
    assert any("release count core" in error for error in result.errors)


def test_verify_corpus_requires_420_second_core_audio(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    write_valid_wav(tmp_path / "ramc_core" / "WAV" / "short.wav", 1_000)
    (tmp_path / "ramc_core" / "TXT").mkdir(exist_ok=True)
    (tmp_path / "ramc_core" / "TXT" / "short.txt").write_text(
        "[0.000,1.000]\tS1\t女,普通话\t短音频\n",
        encoding="utf-8",
    )
    (tmp_path / "ramc_core" / "test.tsv").write_text(
        "WAV/short.wav\tshort.wav\tS1\tS2\t测试\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["counts"]["core"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    write_checksums(tmp_path)

    result = verify_corpus(
        tmp_path,
        expected_counts={
            "core": 20,
            "synthetic": 30,
            "real_overlap": 10,
            "speaker_change_negative": 10,
            "seeds": 3,
        },
    )

    assert result.ok is False
    assert any("core duration" in error for error in result.errors)


def test_verify_corpus_requires_linux_runner(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    (tmp_path / "run_validation_linux.sh").unlink()
    write_checksums(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert any("run_validation_linux.sh" in error for error in result.errors)


def test_verify_corpus_requires_tsv_wav_txt_one_to_one(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    write_valid_wav(tmp_path / "ramc_core" / "WAV" / "extra.wav")
    (tmp_path / "ramc_core" / "TXT" / "extra.txt").write_text(
        "[0.000,1.000]\tS1\tF,Mandarin\ttext\n", encoding="utf-8"
    )
    write_checksums(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert any("core TSV/WAV/TXT mismatch" in error for error in result.errors)


def test_verify_corpus_rejects_nonmonotonic_core_transcript(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    write_valid_wav(tmp_path / "ramc_core" / "WAV" / "sample.wav", 2_000)
    (tmp_path / "ramc_core" / "TXT" / "sample.txt").write_text(
        "[1.000,2.000]\tS1\tF,Mandarin\tlater\n"
        "[0.000,1.000]\tS2\tM,Mandarin\tearlier\n",
        encoding="utf-8",
    )
    (tmp_path / "ramc_core" / "test.tsv").write_text(
        "WAV/sample.wav\tsample.wav\tS1\tS2\ttopic\n", encoding="utf-8"
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    manifest["counts"]["core"] = 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_checksums(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert any("not monotonic" in error for error in result.errors)


def test_verify_corpus_requires_real_marker_and_two_effective_speakers(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    write_valid_wav(tmp_path / "overlap" / "audio" / "real.wav", 2_000)
    overlap = [{
        "id": "real-1",
        "kind": "real_overlap",
        "audio_path": "overlap/audio/real.wav",
        "reference_turns": [
            {"speaker": "S1", "start_ms": 0, "end_ms": 1_000, "text": "one"}
        ],
    }]
    (tmp_path / "overlap" / "manifest.json").write_text(json.dumps(overlap), encoding="utf-8")
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    manifest["counts"]["real_overlap"] = 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_checksums(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert any("real overlap" in error for error in result.errors)


def test_verify_corpus_requires_two_distinct_synthetic_speakers(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    write_valid_wav(tmp_path / "overlap" / "audio" / "synthetic.wav", 2_000)
    overlap = [{
        "id": "synthetic-1",
        "kind": "synthetic",
        "audio_path": "overlap/audio/synthetic.wav",
        "reference_turns": [
            {"speaker": "S1", "start_ms": 0, "end_ms": 1_000, "text": "one"},
            {"speaker": "S1", "start_ms": 500, "end_ms": 1_500, "text": "two"},
        ],
    }]
    (tmp_path / "overlap" / "manifest.json").write_text(json.dumps(overlap), encoding="utf-8")
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    manifest["counts"]["synthetic"] = 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_checksums(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert any("synthetic speakers" in error for error in result.errors)


def test_verify_corpus_rejects_wrong_json_top_level_types(tmp_path: Path):
    write_minimal_corpus(tmp_path)
    (tmp_path / "manifest.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "overlap" / "manifest.json").write_text("{}\n", encoding="utf-8")
    write_checksums(tmp_path)

    result = verify_corpus(tmp_path, expected_counts=None)

    assert result.ok is False
    assert any("top level" in error for error in result.errors)
