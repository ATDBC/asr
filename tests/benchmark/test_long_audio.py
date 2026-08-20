import os
from pathlib import Path
import subprocess
import sys

import pytest


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
