import os
import hashlib
import json
from pathlib import Path
import subprocess

from funasr_transcriber.benchmarking import score_overlap_cases

import pytest


def test_overlap_gate_writes_release_evidence():
    source = Path(__file__).read_text(encoding="utf-8").rsplit(
        "@pytest.mark.skipif", 1
    )[1]

    assert 'summary["release_evidence"] = release_evidence' in source
    assert 'summary["passed"] = summary["passed"] and evidence_passed' in source


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
                "--enable-overlap",
            ],
            check=False,
        )
        assert completed.returncode == 0, item.get("id")
        payload = json.loads((output / "transcript.json").read_text("utf-8"))
        diagnostics = json.loads(
            (output / "diagnostics.json").read_text("utf-8")
        )
        evaluated.append({
            **item,
            "hypothesis_turns": payload["turns"],
            "warnings": payload.get("warnings", []),
            "diagnostics": diagnostics,
            "result_sha256": hashlib.sha256(
                (output / "transcript.json").read_bytes()
            ).hexdigest(),
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
    sentences = [
        sentence
        for item in evaluated
        for sentence in (item["diagnostics"].get("asr") or {}).get(
            "sentences", []
        )
    ]
    turns = [
        turn for item in evaluated for turn in item["hypothesis_turns"]
    ]
    covered = sum(
        bool(item.get("language"))
        and item.get("language_confidence") is not None
        for item in sentences
    )
    valid_timestamps = sum(
        0 <= int(item.get("start_ms", -1)) < int(item.get("end_ms", -1))
        for item in turns
    )
    gpu_stages = {
        f"{item['id']}:{stage}": record
        for item in evaluated
        for stage, record in item["diagnostics"].get(
            "gpu_stages", {}
        ).items()
    }
    release_evidence = {
        "lid_coverage": {
            "sentences": len(sentences),
            "covered": covered,
            "ratio": covered / len(sentences) if sentences else 0.0,
        },
        "timestamp_integrity": {
            "checked": len(turns),
            "valid": valid_timestamps,
            "passed": bool(turns) and valid_timestamps == len(turns),
        },
        "gpu_stages": gpu_stages,
        "result_hashes": {
            item["id"]: item["result_sha256"] for item in evaluated
        },
    }
    evidence_passed = (
        release_evidence["lid_coverage"]["ratio"] == 1.0
        and release_evidence["timestamp_integrity"]["passed"]
        and bool(gpu_stages)
        and all(
            record.get("released") is True
            for record in gpu_stages.values()
        )
    )
    summary["release_evidence"] = release_evidence
    summary["passed"] = summary["passed"] and evidence_passed
    (tmp_path / "overlap-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert summary["passed"] is True, summary
