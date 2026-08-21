#!/usr/bin/env python
"""Run the final non-overlap pipeline on the 20-recording RAMC gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import tempfile
from time import time
from types import SimpleNamespace
import wave

from funasr_transcriber.cli import submit_service_job
from funasr_transcriber.config import load_config
from funasr_transcriber.service_protocol import ServiceProtocolError
from funasr_transcriber.metrics import (
    cer,
    cpcer,
    evaluate_speaker_tracks,
    speaker_count_metrics,
)
from funasr_transcriber.pipeline import normalize_for_integrity
from funasr_transcriber.ramc_corpus import strip_control_markers
from funasr_transcriber.schemas import AlignmentConfidence


TRANSCRIPT_PATTERN = re.compile(
    r"^\[(?P<start>[\d.]+),(?P<end>[\d.]+)\]\s+"
    r"(?P<speaker>\w+)\s+(?P<meta>[^\t]*)\t(?P<text>.*)$"
)
EVAL_START = 60.0
EVAL_SECONDS = 420.0


@dataclass(frozen=True, slots=True)
class PreparedRecording:
    name: str
    clip: Path
    reference: list[dict[str, object]]
    ref_by_speaker: dict[str, str]
    ref_text: str
    output_dir: Path


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Publish one JSON snapshot without leaving a partial document."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


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


def transcribe_with_service(
    audio: Path,
    output_dir: Path,
    config,
    config_path: Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run a RAMC item through the resident service and read its exports."""

    client, request, accepted = submit_service_job(
        SimpleNamespace(
            input_audio=audio,
            output_dir=output_dir,
            overwrite=False,
            keep_intermediate=False,
            enable_overlap=False,
            disable_overlap=True,
            profile_resources=True,
            config=config_path,
        ),
        config,
    )
    job_id = accepted.get("job_id")
    if not isinstance(job_id, str):
        raise RuntimeError("service did not return an accepted job id")
    response = client.wait(job_id, f"wait-{request.identifier}")
    result = response.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("output_dir"), str):
        raise RuntimeError("service did not return an output directory")
    published = Path(result["output_dir"]).resolve()
    expected = output_dir.resolve()
    if published != expected:
        raise RuntimeError(
            f"service published unexpected output directory: {published} != {expected}"
        )
    try:
        transcript = json.loads(
            (published / "transcript.json").read_text(encoding="utf-8")
        )
        diagnostics = json.loads(
            (published / "diagnostics.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"service output is missing a valid transcript or diagnostics: {published}"
        ) from exc
    if not isinstance(transcript, dict) or not isinstance(diagnostics, dict):
        raise RuntimeError("service output payload must be a JSON object")
    return transcript, diagnostics


def asr_text_from_diagnostics(diagnostics: dict[str, object]) -> str:
    """Reconstruct the base ASR text from the service diagnostics contract."""

    asr = diagnostics.get("asr")
    if not isinstance(asr, dict):
        raise RuntimeError("service diagnostics are missing ASR metadata")
    sentences = asr.get("sentences")
    if not isinstance(sentences, list):
        raise RuntimeError("service diagnostics are missing ASR sentences")
    try:
        return "".join(str(sentence["text"]) for sentence in sentences)
    except (KeyError, TypeError) as exc:
        raise RuntimeError("service diagnostics contain an invalid ASR sentence") from exc


def _read_published_exports(
    response: dict[str, object],
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    result = response.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("output_dir"), str):
        raise RuntimeError("service did not return an output directory")
    published = Path(result["output_dir"]).resolve()
    expected = Path(output_dir).resolve()
    if published != expected:
        raise RuntimeError(
            f"service published unexpected output directory: {published} != {expected}"
        )
    try:
        transcript = json.loads(
            (published / "transcript.json").read_text(encoding="utf-8")
        )
        diagnostics = json.loads(
            (published / "diagnostics.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"service output is missing a valid transcript or diagnostics: {published}"
        ) from exc
    if not isinstance(transcript, dict) or not isinstance(diagnostics, dict):
        raise RuntimeError("service output payload must be a JSON object")
    return transcript, diagnostics


def _prepare_recordings(
    dataset: Path,
    test_tsv: Path,
    count: int,
    preclipped: bool,
    run_root: Path,
    temporary_path: Path,
) -> list[PreparedRecording]:
    prepared: list[PreparedRecording] = []
    for wav_name in partition_names(test_tsv):
        if len(prepared) >= count:
            break
        source = dataset / "WAV" / wav_name
        transcript_path = dataset / "TXT" / f"{source.stem}.txt"
        if not source.is_file() or not transcript_path.is_file():
            continue
        reference, ref_by_speaker, ref_text = reference_for(
            dataset,
            source.stem,
            preclipped=preclipped,
        )
        clip = source if preclipped else temporary_path / f"{source.stem}.wav"
        if not preclipped:
            clip_wav(source, clip)
        prepared.append(
            PreparedRecording(
                name=source.stem,
                clip=clip.resolve(),
                reference=reference,
                ref_by_speaker=ref_by_speaker,
                ref_text=ref_text,
                output_dir=(run_root / "ramc" / source.stem).resolve(),
            )
        )
    if len(prepared) != count:
        raise RuntimeError(
            f"requested {count} recordings, prepared {len(prepared)}"
        )
    if len({recording.name for recording in prepared}) != len(prepared):
        raise RuntimeError("RAMC input contains duplicate recording stems")
    return prepared


def _recording_result(
    recording: PreparedRecording,
    transcript: dict[str, object],
    diagnostics: dict[str, object],
) -> dict[str, object]:
    turns = transcript.get("turns", [])
    if not isinstance(turns, list):
        raise RuntimeError("service transcript turns must be a list")
    hypothesis = [
        {
            "start": int(turn["start_ms"]) / 1000,
            "end": int(turn["end_ms"]) / 1000,
            "speaker": turn["speaker"],
        }
        for turn in turns
    ]
    hyp_by_speaker = defaultdict(str)
    low_chars = total_chars = 0
    for turn in turns:
        text = str(turn["text"])
        hyp_by_speaker[turn["speaker"]] += text
        total_chars += len(text)
        if turn.get("alignment_confidence") == AlignmentConfidence.LOW.value:
            low_chars += len(text)
    tracks = evaluate_speaker_tracks(recording.reference, hypothesis)
    asr = diagnostics.get("asr")
    sentences = asr.get("sentences") if isinstance(asr, dict) else []
    if not isinstance(sentences, list):
        raise RuntimeError("service diagnostics are missing ASR sentences")
    asr_text = asr_text_from_diagnostics(diagnostics)
    valid_timestamps = sum(
        0 <= int(turn["start_ms"]) < int(turn["end_ms"]) for turn in turns
    )
    transcript_digest = hashlib.sha256(
        json.dumps(
            [
                {
                    "speaker_id": turn["speaker"],
                    "start_ms": turn["start_ms"],
                    "end_ms": turn["end_ms"],
                    "text": turn["text"],
                }
                for turn in turns
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    profile = diagnostics.get("resource_profile")
    return {
        "name": recording.name,
        "cer": cer(recording.ref_text, str(transcript.get("full_text", ""))),
        "cpcer": cpcer(recording.ref_by_speaker, dict(hyp_by_speaker)),
        "purity": tracks["weighted_purity"],
        "der_like": tracks["der_like"],
        "reference_speakers": len(recording.ref_by_speaker),
        "hypothesis_speakers": int(transcript["speaker_count"]),
        "low_confidence_ratio": low_chars / total_chars if total_chars else 0.0,
        "text_retained": (
            normalize_for_integrity(asr_text)
            == normalize_for_integrity(str(transcript.get("full_text", "")))
        ),
        "timestamp_checked": len(turns),
        "timestamp_valid": valid_timestamps,
        "resource_profile": profile if isinstance(profile, dict) else {},
        "result_sha256": transcript_digest,
    }


def _progress_payload(run_root: Path, records: list[dict[str, object]]) -> dict[str, object]:
    pending = sum(
        record["status"] in {"prepared", "accepted"} for record in records
    )
    return {
        "run_id": run_root.name,
        "complete": pending == 0,
        "pending": pending,
        "recordings": records,
    }


def _summary_payload(
    run_root: Path,
    progress: list[dict[str, object]],
    results: list[dict[str, object]],
    expected_count: int,
) -> dict[str, object]:
    failures = [
        {"name": str(record["name"]), "error": str(record["error"])}
        for record in progress
        if record["status"] in {"failed", "not_submitted"}
    ]
    pending = sum(
        record["status"] in {"prepared", "accepted"} for record in progress
    )
    complete = pending == 0
    if results:
        counts = speaker_count_metrics(
            [int(item["reference_speakers"]) for item in results],
            [int(item["hypothesis_speakers"]) for item in results],
        )
        summary: dict[str, object] = {
            "recordings": len(results),
            "cer": statistics.mean(float(item["cer"]) for item in results),
            "cpcer": statistics.mean(float(item["cpcer"]) for item in results),
            "purity": statistics.mean(float(item["purity"]) for item in results),
            "der_like": statistics.mean(float(item["der_like"]) for item in results),
            "low_confidence_ratio": statistics.mean(
                float(item["low_confidence_ratio"]) for item in results
            ),
            "speaker_count": counts,
            "text_retention_rate": statistics.mean(
                bool(item["text_retained"]) for item in results
            ),
        }
    else:
        counts = {"signed_mean": None, "mae": None, "exact_accuracy": None}
        summary = {
            "recordings": 0,
            "cer": None,
            "cpcer": None,
            "purity": None,
            "der_like": None,
            "low_confidence_ratio": None,
            "speaker_count": counts,
            "text_retention_rate": None,
        }
    timestamp_checked = sum(int(item["timestamp_checked"]) for item in results)
    timestamp_valid = sum(int(item["timestamp_valid"]) for item in results)
    profiled_recordings = sum(
        item["resource_profile"].get("enabled") is True for item in results
    )
    release_evidence = {
        "timestamp_integrity": {
            "checked": timestamp_checked,
            "valid": timestamp_valid,
            "passed": timestamp_checked > 0 and timestamp_valid == timestamp_checked,
        },
        "resource_profiles": {
            "requested": expected_count,
            "completed": profiled_recordings,
        },
        "result_hashes": {
            str(item["name"]): item["result_sha256"] for item in results
        },
    }
    ready = complete and not failures and len(results) == expected_count
    gates = {
        "purity": ready and float(summary["purity"]) >= 0.92,
        "der_like": ready and float(summary["der_like"]) <= 0.22,
        "speaker_count_mae": ready and float(counts["mae"]) <= 0.25,
        "speaker_count_accuracy": ready
        and float(counts["exact_accuracy"]) >= 0.80,
        "cpcer_delta": ready
        and float(summary["cpcer"]) - float(summary["cer"]) <= 0.05,
        "low_confidence": ready
        and float(summary["low_confidence_ratio"]) <= 0.05,
        "text_retention": ready and float(summary["text_retention_rate"]) == 1.0,
        "timestamp_integrity": ready
        and release_evidence["timestamp_integrity"]["passed"],
        "resource_profiles": ready and profiled_recordings == expected_count,
    }
    return {
        "run_id": run_root.name,
        "complete": complete,
        "pending": pending,
        "failures": failures,
        "summary": summary,
        "gates": gates,
        "passed": complete and not failures and all(gates.values()),
        "release_evidence": release_evidence,
        "recording_results": results,
    }


def _service_args(recording: PreparedRecording, config_path: Path | None):
    return SimpleNamespace(
        input_audio=recording.clip,
        output_dir=recording.output_dir,
        overwrite=False,
        keep_intermediate=False,
        enable_overlap=False,
        disable_overlap=True,
        profile_resources=True,
        config=config_path,
    )


def run_benchmark(
    *,
    dataset: Path,
    test_tsv: Path,
    output: Path,
    count: int,
    preclipped: bool,
    config,
    config_path: Path | None,
    submit_job=submit_service_job,
) -> int:
    """Enqueue the complete RAMC batch before waiting for its FIFO results."""

    run_root = Path(output).expanduser().resolve().parent
    summary_path = Path(output).expanduser().resolve()
    progress_path = run_root / "progress.json"
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="ramc-e2e-") as temporary:
        prepared = _prepare_recordings(
            Path(dataset),
            Path(test_tsv),
            count,
            preclipped,
            run_root,
            Path(temporary),
        )
        progress: list[dict[str, object]] = [
            {
                "name": recording.name,
                "job_id": None,
                "queue_depth": None,
                "status": "prepared",
                "output_dir": str(recording.output_dir),
                "submitted_at": None,
                "completed_at": None,
            }
            for recording in prepared
        ]

        def persist() -> dict[str, object]:
            atomic_write_json(progress_path, _progress_payload(run_root, progress))
            payload = _summary_payload(run_root, progress, results, count)
            atomic_write_json(summary_path, payload)
            return payload

        payload = persist()
        accepted: list[tuple[PreparedRecording, dict[str, object], object, str]] = []
        submit_error: str | None = None
        for index, recording in enumerate(prepared):
            entry = progress[index]
            if submit_error is not None:
                entry["status"] = "not_submitted"
                entry["error"] = submit_error
                entry["completed_at"] = time()
                payload = persist()
                continue
            try:
                client, request, acknowledgement = submit_job(
                    _service_args(recording, config_path),
                    config,
                )
                job_id = acknowledgement.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    raise ServiceProtocolError("service did not return an accepted job id")
                queue_depth = acknowledgement.get("queue_depth")
                if not isinstance(queue_depth, int) or isinstance(queue_depth, bool):
                    raise ServiceProtocolError(
                        "service did not return an accepted queue depth"
                    )
                entry["job_id"] = job_id
                entry["queue_depth"] = queue_depth
                entry["status"] = "accepted"
                entry["submitted_at"] = time()
                accepted.append((recording, entry, client, job_id))
            except Exception as exc:
                submit_error = str(exc)
                entry["status"] = "failed"
                entry["error"] = submit_error
                entry["completed_at"] = time()
            finally:
                persist()

        for recording, entry, client, job_id in accepted:
            try:
                response = client.wait(job_id, f"ramc-wait-{job_id}")
                transcript, diagnostics = _read_published_exports(
                    response,
                    recording.output_dir,
                )
                results.append(_recording_result(recording, transcript, diagnostics))
                entry["status"] = "completed"
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)
            finally:
                entry["completed_at"] = time()
                payload = persist()
    return 0 if payload["passed"] is True else 1


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
    return run_benchmark(
        dataset=args.dataset,
        test_tsv=args.test_tsv,
        output=args.output,
        count=args.count,
        preclipped=args.preclipped,
        config=config,
        config_path=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
