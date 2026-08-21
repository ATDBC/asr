from pathlib import Path
import subprocess
import sys

from funasr_transcriber.release_validation import build_release_summary


ROOT = Path(__file__).parents[2]


def test_build_cli_exposes_required_arguments():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_release_validation_corpus.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0
    for option in (
        "--source-root",
        "--output-dir",
        "--seed",
        "--overwrite",
        "--ffmpeg",
        "--ffprobe",
        "--refresh-runtime-metadata",
    ):
        assert option in completed.stdout


def test_build_cli_rejects_missing_source(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_release_validation_corpus.py"),
            "--source-root",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode != 0
    assert "source root" in completed.stderr.lower()


def test_linux_runner_contains_all_release_stages():
    script = (ROOT / "scripts" / "run_validation_linux.sh").read_text("utf-8")

    assert "set -euo pipefail" in script
    assert "sha256sum -c checksums.sha256" in script
    assert "models verify" in script
    assert "doctor" in script
    assert "doctor_smoke.wav" in script
    assert "test_ramc_gate.py" in script
    assert "test_overlap_gate.py" in script
    assert "test_input_matrix.py" in script
    assert "test_long_audio.py" in script
    assert 'LATEST_SUMMARY="$RESULTS_ROOT/summary.json"' in script
    assert 'import pytest' in script
    assert '--basetemp "$RESULTS/pytest-ramc"' in script
    assert 'stage_results' in script
    assert 'release_evidence' in script
    assert 'FIRERED_MODULES' in script
    assert '.venv-asr/bin/python' in script
    assert 'FireRedVAD' in script
    assert 'FireRedLID' in script
    assert 'FireRedASR2-AED' in script
    assert 'FireRedPunc' in script
    assert 'summary.txt' in script
    assert 'RUN_ID="$(date -u' in script
    assert "G:\\" not in script


def test_linux_runner_records_gpu_release_at_service_shutdown():
    script = (ROOT / "scripts" / "run_validation_linux.sh").read_text("utf-8")

    assert "SERVICE_GPU_BEFORE" in script
    assert "service-release-summary.json" in script
    assert 'service_release.get("gpu_stages")' in script


def _complete_evidence():
    digest = "a" * 64
    return {
        "firered_modules": {
            "vad": True,
            "lid": True,
            "aed": True,
            "punc": True,
        },
        "lid_coverage": {
            "sentences": 20,
            "covered": 20,
            "ratio": 1.0,
        },
        "timestamp_integrity": {
            "checked": 20,
            "valid": 20,
            "passed": True,
        },
        "gpu_stages": {
            "doctor_primary_asr": {
                "before_free_mib": 22_000,
                "after_free_mib": 21_900,
                "released": True,
            },
        },
        "deterministic_hashes": {
            "30": [digest, digest],
            "60": [digest, digest],
            "120": [digest, digest],
        },
    }


def test_release_summary_rejects_missing_evidence_despite_zero_exit():
    summary = build_release_summary(
        exit_code=0,
        stress_skipped=False,
        last_stage="complete",
        stage_results={},
        release_evidence={},
    )

    assert summary["passed"] is False
    assert summary["complete"] is False
    assert set(summary["missing_evidence"]) == {
        "firered_modules",
        "lid_coverage",
        "timestamp_integrity",
        "gpu_stages",
        "deterministic_hashes",
    }


def test_release_summary_accepts_complete_evidence():
    summary = build_release_summary(
        exit_code=0,
        stress_skipped=False,
        last_stage="complete",
        stage_results={"doctor-summary.json": {"passed": True}},
        release_evidence=_complete_evidence(),
    )

    assert summary["passed"] is True
    assert summary["complete"] is True
    assert summary["missing_evidence"] == []
