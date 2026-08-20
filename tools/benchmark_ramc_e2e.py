#!/usr/bin/env python
"""Run the final non-overlap pipeline on the 20-recording RAMC gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import tempfile
import wave

from funasr_transcriber.cli import (
    execute_model_phases,
    require_physical_gpu_zero,
)
from funasr_transcriber.config import load_config
from funasr_transcriber.firered_client import FireRedWorkerClient
from funasr_transcriber.gpu import (
    build_gpu_stage_record,
    query_gpu,
    require_gpu_capacity,
)
from funasr_transcriber.metrics import (
    cer,
    cpcer,
    evaluate_speaker_tracks,
    speaker_count_metrics,
)
from funasr_transcriber.pipeline import normalize_for_integrity
from funasr_transcriber.ramc_corpus import strip_control_markers
from funasr_transcriber.schemas import AlignmentConfidence
from funasr_transcriber.speaker_diarization import SpeakerDiarizer


TRANSCRIPT_PATTERN = re.compile(
    r"^\[(?P<start>[\d.]+),(?P<end>[\d.]+)\]\s+"
    r"(?P<speaker>\w+)\s+(?P<meta>[^\t]*)\t(?P<text>.*)$"
)
EVAL_START = 60.0
EVAL_SECONDS = 420.0


def partition_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        first = line.strip().split("\t")[0]
        if first.endswith(".wav"):
            names.append(Path(first).name)
    return names


def reference_for(dataset: Path, stem: str, preclipped: bool = False):
    intervals = []
    text_by_speaker = defaultdict(str)
    ordered_text = []
    path = dataset / "TXT" / f"{stem}.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TRANSCRIPT_PATTERN.match(line)
        if not match or match.group("speaker") == "G00000000":
            continue
        start, end = float(match.group("start")), float(match.group("end"))
        eval_start = 0.0 if preclipped else EVAL_START
        clipped_start = max(start, eval_start)
        clipped_end = min(end, eval_start + EVAL_SECONDS)
        if clipped_end <= clipped_start:
            continue
        speaker = match.group("speaker")
        text = strip_control_markers(match.group("text"))
        if not text:
            continue
        intervals.append(
            {
                "start": clipped_start - eval_start,
                "end": clipped_end - eval_start,
                "speaker": speaker,
            }
        )
        text_by_speaker[speaker] += text
        ordered_text.append(text)
    return intervals, dict(text_by_speaker), "".join(ordered_text)


def clip_wav(source: Path, destination: Path) -> None:
    with wave.open(str(source), "rb") as input_stream:
        rate = input_stream.getframerate()
        input_stream.setpos(int(EVAL_START * rate))
        frames = input_stream.readframes(int(EVAL_SECONDS * rate))
        channels = input_stream.getnchannels()
        width = input_stream.getsampwidth()
    with wave.open(str(destination), "wb") as output_stream:
        output_stream.setnchannels(channels)
        output_stream.setsampwidth(width)
        output_stream.setframerate(rate)
        output_stream.writeframes(frames)


def record_gpu_stage(stage, before, owner, diagnostics) -> None:
    response = getattr(owner, "last_response", {}) or {}
    metrics = response.get("metrics") or {}
    diagnostics.gpu_stages[stage] = build_gpu_stage_record(
        before,
        query_gpu("cuda:0"),
        peak_gpu_bytes=metrics.get("peak_gpu_bytes"),
        load_seconds=metrics.get("load_seconds"),
        inference_seconds=metrics.get("inference_seconds"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-tsv", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--preclipped", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    require_physical_gpu_zero(config.general.device)
    per_recording = []
    with tempfile.TemporaryDirectory(prefix="ramc-e2e-") as temporary:
        temporary_path = Path(temporary)
        for wav_name in partition_names(args.test_tsv):
            if len(per_recording) >= args.count:
                break
            source = args.dataset / "WAV" / wav_name
            transcript_path = args.dataset / "TXT" / f"{source.stem}.txt"
            if not source.is_file() or not transcript_path.is_file():
                continue
            reference, ref_by_speaker, ref_text = reference_for(
                args.dataset,
                source.stem,
                preclipped=args.preclipped,
            )
            if args.preclipped:
                clip = source
            else:
                clip = temporary_path / f"{source.stem}.wav"
                clip_wav(source, clip)
            result = execute_model_phases(
                clip.resolve(),
                config,
                overlap_enabled=False,
                asr_client_factory=lambda stage: FireRedWorkerClient(
                    worker_python=config.asr.worker_python,
                    model_root=config.paths.model_dir,
                    device="cuda:0",
                    timeout_seconds=config.asr.timeout_seconds,
                    stderr_log_path=(
                        temporary_path / f"{source.stem}-{stage}.log"
                    ),
                ),
                diarizer_factory=lambda: SpeakerDiarizer.from_config(config),
                overlap_preparer_factory=lambda diarizer: None,
                gpu_check=lambda stage: require_gpu_capacity("cuda:0"),
                gpu_after=record_gpu_stage,
            )
            hypothesis = [
                {
                    "start": turn.start_ms / 1000,
                    "end": turn.end_ms / 1000,
                    "speaker": turn.speaker_id,
                }
                for turn in result.transcript.turns
            ]
            hyp_by_speaker = defaultdict(str)
            low_chars = total_chars = 0
            for turn in result.transcript.turns:
                hyp_by_speaker[turn.speaker_id] += turn.text
                total_chars += len(turn.text)
                if turn.confidence is AlignmentConfidence.LOW:
                    low_chars += len(turn.text)
            tracks = evaluate_speaker_tracks(reference, hypothesis)
            sentences = result.diagnostics.asr.get("sentences") or []
            lid_covered = sum(
                bool(item.get("language"))
                and item.get("language_confidence") is not None
                for item in sentences
            )
            valid_timestamps = sum(
                0 <= turn.start_ms < turn.end_ms
                for turn in result.transcript.turns
            )
            transcript_digest = hashlib.sha256(
                json.dumps(
                    [
                        {
                            "speaker_id": turn.speaker_id,
                            "start_ms": turn.start_ms,
                            "end_ms": turn.end_ms,
                            "text": turn.text,
                        }
                        for turn in result.transcript.turns
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            per_recording.append(
                {
                    "name": source.stem,
                    "cer": cer(ref_text, result.transcript.full_text),
                    "cpcer": cpcer(ref_by_speaker, dict(hyp_by_speaker)),
                    "purity": tracks["weighted_purity"],
                    "der_like": tracks["der_like"],
                    "reference_speakers": len(ref_by_speaker),
                    "hypothesis_speakers": result.transcript.speaker_count,
                    "low_confidence_ratio": (
                        low_chars / total_chars if total_chars else 0.0
                    ),
                    "text_retained": (
                        normalize_for_integrity(result.asr_text)
                        == normalize_for_integrity(result.transcript.full_text)
                    ),
                    "lid_sentences": len(sentences),
                    "lid_covered": lid_covered,
                    "timestamp_checked": len(result.transcript.turns),
                    "timestamp_valid": valid_timestamps,
                    "gpu_stages": result.diagnostics.gpu_stages,
                    "result_sha256": transcript_digest,
                }
            )

    if len(per_recording) != args.count:
        raise RuntimeError(
            f"requested {args.count} recordings, prepared {len(per_recording)}"
        )
    counts = speaker_count_metrics(
        [item["reference_speakers"] for item in per_recording],
        [item["hypothesis_speakers"] for item in per_recording],
    )
    summary = {
        "recordings": len(per_recording),
        "cer": statistics.mean(item["cer"] for item in per_recording),
        "cpcer": statistics.mean(item["cpcer"] for item in per_recording),
        "purity": statistics.mean(item["purity"] for item in per_recording),
        "der_like": statistics.mean(item["der_like"] for item in per_recording),
        "low_confidence_ratio": statistics.mean(
            item["low_confidence_ratio"] for item in per_recording
        ),
        "speaker_count": counts,
        "text_retention_rate": statistics.mean(
            item["text_retained"] for item in per_recording
        ),
    }
    gates = {
        "purity": summary["purity"] >= 0.92,
        "der_like": summary["der_like"] <= 0.22,
        "speaker_count_mae": counts["mae"] <= 0.25,
        "speaker_count_accuracy": counts["exact_accuracy"] >= 0.80,
        "cpcer_delta": summary["cpcer"] - summary["cer"] <= 0.05,
        "low_confidence": summary["low_confidence_ratio"] <= 0.05,
        "text_retention": summary["text_retention_rate"] == 1.0,
    }
    lid_sentences = sum(item["lid_sentences"] for item in per_recording)
    lid_covered = sum(item["lid_covered"] for item in per_recording)
    timestamp_checked = sum(
        item["timestamp_checked"] for item in per_recording
    )
    timestamp_valid = sum(item["timestamp_valid"] for item in per_recording)
    gpu_stages = {
        f"{item['name']}:{stage}": record
        for item in per_recording
        for stage, record in item["gpu_stages"].items()
    }
    release_evidence = {
        "lid_coverage": {
            "sentences": lid_sentences,
            "covered": lid_covered,
            "ratio": lid_covered / lid_sentences if lid_sentences else 0.0,
        },
        "timestamp_integrity": {
            "checked": timestamp_checked,
            "valid": timestamp_valid,
            "passed": (
                timestamp_checked > 0 and timestamp_valid == timestamp_checked
            ),
        },
        "gpu_stages": gpu_stages,
        "result_hashes": {
            item["name"]: item["result_sha256"] for item in per_recording
        },
    }
    gates.update(
        {
            "lid_coverage": release_evidence["lid_coverage"]["ratio"] == 1.0,
            "timestamp_integrity": release_evidence["timestamp_integrity"][
                "passed"
            ],
            "gpu_evidence": bool(gpu_stages)
            and all(item.get("released") is True for item in gpu_stages.values()),
        }
    )
    payload = {
        "summary": summary,
        "gates": gates,
        "passed": all(gates.values()),
        "release_evidence": release_evidence,
        "recording_results": per_recording,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
