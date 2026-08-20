from pathlib import Path
import subprocess
import sys


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
    assert 'summary.txt' in script
    assert 'RUN_ID="$(date -u' in script
    assert "G:\\" not in script
