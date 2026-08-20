import os
import json
from pathlib import Path
import subprocess

from funasr_transcriber.benchmarking import score_overlap_cases

import pytest


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_BENCHMARKS") != "1",
    reason="release benchmarks are disabled",
)
def test_overlap_gate_requires_manifest(tmp_path: Path):
    manifest_path = Path(os.environ["FUNASR_TRANSCRIBER_OVERLAP_MANIFEST"])
    manifest = json.loads(manifest_path.read_text("utf-8"))
    corpus_root = manifest_path.parents[1]
    executable = os.getenv("FUNASR_TRANSCRIBE_BIN", "funasr-transcribe")
    config = os.environ["FUNASR_TRANSCRIBER_CONFIG"]
    evaluated = []
    for index, item in enumerate(manifest, start=1):
        output = tmp_path / f"overlap-{index:03d}"
        audio_path = corpus_root / item["audio_path"]
        completed = subprocess.run(
            [
                executable,
                str(audio_path),
                "--config",
                config,
                "--output-dir",
                str(output),
            ],
            check=False,
        )
        assert completed.returncode == 0, item.get("id")
        payload = json.loads((output / "transcript.json").read_text("utf-8"))
        evaluated.append({
            **item,
            "hypothesis_turns": payload["turns"],
            "warnings": payload.get("warnings", []),
        })
    summary = score_overlap_cases(evaluated)
    summary["case_observations"] = [
        {
            "id": item.get("id"),
            "kind": item.get("kind"),
            "warnings": item.get("warnings", []),
        }
        for item in evaluated
    ]
    (tmp_path / "overlap-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert summary["passed"] is True, summary
