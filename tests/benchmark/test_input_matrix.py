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
        })
    (tmp_path / "input-matrix-summary.json").write_text(
        json.dumps({"passed": True, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
