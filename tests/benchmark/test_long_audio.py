import os
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_long_audio_tool_emits_release_evidence():
    source = (
        Path(__file__).parents[2] / "tools" / "stress_long_audio.py"
    ).read_text(encoding="utf-8")

    for field in (
        '"release_evidence"',
        '"lid_coverage"',
        '"timestamp_integrity"',
        '"gpu_stages"',
        '"deterministic_hashes"',
    ):
        assert field in source


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_BENCHMARKS") != "1",
    reason="release benchmarks are disabled",
)
def test_long_audio_gate_requires_source(tmp_path: Path):
    tool = Path(__file__).parents[2] / "tools" / "stress_long_audio.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--source", os.environ["FUNASR_TRANSCRIBER_STRESS_SOURCE"],
            "--config", os.environ["FUNASR_TRANSCRIBER_CONFIG"],
            "--output-dir", str(tmp_path / "stress"),
        ],
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(
        (tmp_path / "stress" / "stress-summary.json").read_text("utf-8")
    )
    evidence = payload["release_evidence"]
    assert evidence["lid_coverage"]["ratio"] == 1.0
    assert evidence["timestamp_integrity"]["passed"] is True
    assert evidence["gpu_stages"]
    assert set(evidence["deterministic_hashes"]) == {"30", "60", "120"}
