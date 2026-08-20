import os
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_BENCHMARKS") != "1",
    reason="release benchmarks are disabled",
)
def test_ramc_release_gate_is_run_explicitly(tmp_path: Path):
    root = Path(os.environ["FUNASR_TRANSCRIBER_RAMC_ROOT"])
    tsv_value = os.getenv("FUNASR_TRANSCRIBER_RAMC_TSV")
    candidates = list(root.rglob("*test*.tsv")) if not tsv_value else []
    test_tsv = Path(tsv_value) if tsv_value else candidates[0]
    output = tmp_path / "ramc-summary.json"
    command = [
        sys.executable,
        str(Path(__file__).parents[2] / "tools" / "benchmark_ramc_e2e.py"),
        "--dataset", str(root),
        "--test-tsv", str(test_tsv),
        "--output", str(output),
        "--count", "20",
        "--preclipped",
    ]
    if os.getenv("FUNASR_TRANSCRIBER_CONFIG"):
        command.extend(["--config", os.environ["FUNASR_TRANSCRIBER_CONFIG"]])

    completed = subprocess.run(command, check=False)

    assert completed.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
