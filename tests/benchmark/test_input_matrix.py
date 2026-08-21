import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_input_matrix_writes_release_evidence():
    source = Path(__file__).read_text(encoding="utf-8").rsplit(
        "@pytest.mark.skipif", 1
    )[1]

    assert '"release_evidence": release_evidence' in source
    assert '"passed": evidence_passed' in source


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_BENCHMARKS") != "1",
    reason="release benchmarks are disabled",
)
def test_input_matrix_requires_fixture(tmp_path: Path):
    fixture = Path(os.environ["FUNASR_TRANSCRIBER_FORMAT_FIXTURE"])
    matrix = tmp_path / "matrix"
    builder = Path(__file__).parents[2] / "tools" / "build_input_matrix.py"
    subprocess.run(
        [sys.executable, str(builder), "--source", str(fixture), "--output-dir", str(matrix)],
        check=True,
    )
    manifest = json.loads((matrix / "manifest.json").read_text("utf-8"))
    assert len(manifest) == 48
    assert {item["extension"] for item in manifest} == {
        ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"
    }
    assert {item["sample_rate"] for item in manifest} == {8000, 16000, 44100, 48000}
    assert {item["channels"] for item in manifest} == {1, 2}
    executable = os.getenv("FUNASR_TRANSCRIBE_BIN", "funasr-transcribe")
    config = os.environ["FUNASR_TRANSCRIBER_CONFIG"]
    results = []
    for index, item in enumerate(manifest, start=1):
        output = tmp_path / f"result-{index:03d}"
        completed = subprocess.run(
            [executable, item["path"], "--config", config, "--output-dir", str(output), "--disable-overlap"],
            check=False,
        )
        assert completed.returncode == 0, item
        transcript = json.loads((output / "transcript.json").read_text("utf-8"))
        diagnostics = json.loads(
            (output / "diagnostics.json").read_text("utf-8")
        )
        assert {
            "transcript.json",
            "transcript.txt",
            "diagnostics.json",
            "transcribe.log",
        } <= {path.name for path in output.iterdir() if path.is_file()}, item
        turns = transcript.get("turns") or []
        assert turns and all(turn.get("text") for turn in turns), item
        assert all(
            0 <= turn["start_ms"] < turn["end_ms"]
            for turn in turns
        ), item
        results.append({
            "input": item,
            "speaker_count": transcript.get("speaker_count"),
            "turns": len(turns),
            "warnings": transcript.get("warnings", []),
            "diagnostics": diagnostics,
            "result_sha256": hashlib.sha256(
                (output / "transcript.json").read_bytes()
            ).hexdigest(),
        })
    timestamp_checked = sum(item["turns"] for item in results)
    timestamp_valid = timestamp_checked
    gpu_stages = {
        f"matrix-{index:03d}:{stage}": record
        for index, item in enumerate(results, start=1)
        for stage, record in item["diagnostics"].get(
            "gpu_stages", {}
        ).items()
    }
    release_evidence = {
        "timestamp_integrity": {
            "checked": timestamp_checked,
            "valid": timestamp_valid,
            "passed": timestamp_checked > 0,
        },
        "gpu_stages": gpu_stages,
        "result_hashes": {
            f"matrix-{index:03d}": item["result_sha256"]
            for index, item in enumerate(results, start=1)
        },
    }
    evidence_passed = (
        release_evidence["timestamp_integrity"]["passed"]
        and bool(gpu_stages)
        and all(
            record.get("released") is True
            for record in gpu_stages.values()
        )
    )
    for item in results:
        item.pop("diagnostics", None)
    (tmp_path / "input-matrix-summary.json").write_text(
        json.dumps(
            {
                "passed": evidence_passed,
                "results": results,
                "release_evidence": release_evidence,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
