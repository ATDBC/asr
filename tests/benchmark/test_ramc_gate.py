import os
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_ramc_tool_emits_persistent_service_evidence():
    source = (
        Path(__file__).parents[2] / "tools" / "benchmark_ramc_e2e.py"
    ).read_text(encoding="utf-8")

    for field in (
        '"release_evidence"',
        '"lid_coverage"',
        '"timestamp_integrity"',
        '"resource_profiles"',
        '"result_hashes"',
    ):
        assert field in source


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
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    evidence = payload["release_evidence"]
    assert evidence["lid_coverage"]["ratio"] == 1.0
    assert evidence["timestamp_integrity"]["passed"] is True
    assert evidence["resource_profiles"] == {
        "requested": 20,
        "completed": 20,
    }
    assert len(evidence["result_hashes"]) == 20
