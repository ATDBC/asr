"""Deterministic, atomic transcript exporters."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

from .audio import AudioInfo
from .schemas import Transcript, Turn


def format_timestamp(milliseconds: int) -> str:
    total_seconds = max(0, milliseconds // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _turn_payload(turn: Turn) -> dict[str, Any]:
    return {
        "index": turn.index,
        "speaker": turn.speaker_id,
        "start_ms": turn.start_ms,
        "end_ms": turn.end_ms,
        "text": turn.text,
        "source": turn.source,
        "alignment_confidence": turn.confidence.value,
        "overlap": turn.overlap,
    }


def transcript_payload(
    transcript: Transcript,
    input_info: AudioInfo | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "status": "success",
        "language": transcript.language,
        "speaker_count": transcript.speaker_count,
        "full_text": transcript.full_text,
        "turns": [_turn_payload(turn) for turn in transcript.turns],
        "warnings": list(transcript.warnings),
    }
    if input_info is not None:
        payload["input"] = {
            "path": str(input_info.path),
            "format": input_info.codec,
            "duration_ms": input_info.duration_ms,
            "sample_rate": input_info.sample_rate,
            "channels": input_info.channels,
        }
    return payload


def _write_durable(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def export_transcript(
    transcript: Transcript,
    output_dir: Path,
    overwrite: bool,
    input_info: AudioInfo | None = None,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    text_path = directory / "transcript.txt"
    json_path = directory / "transcript.json"
    if not overwrite:
        existing = [path for path in (text_path, json_path) if path.exists()]
        if existing:
            raise FileExistsError(f"transcript output already exists: {existing[0]}")

    directory.mkdir(parents=True, exist_ok=True)
    text_tmp = directory / ".transcript.txt.tmp"
    json_tmp = directory / ".transcript.json.tmp"
    text_content = "".join(
        (
            f"[{format_timestamp(turn.start_ms)}] "
            f"说话人{turn.speaker_id}：{turn.text}\n"
        )
        for turn in transcript.turns
    )
    json_content = json.dumps(
        transcript_payload(transcript, input_info=input_info),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    try:
        _write_durable(text_tmp, text_content)
        _write_durable(json_tmp, json_content)
        text_tmp.replace(text_path)
        json_tmp.replace(json_path)
    finally:
        text_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)
    return text_path, json_path


def publish_output_directory(
    staging_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    staging = Path(staging_dir)
    target = Path(output_dir)
    if not staging.is_dir():
        raise FileNotFoundError(f"staging output does not exist: {staging}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        staging.replace(target)
        return target

    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    target.replace(backup)
    try:
        staging.replace(target)
    except Exception:
        backup.replace(target)
        raise
    else:
        try:
            shutil.rmtree(backup)
        except OSError:
            pass
    return target
