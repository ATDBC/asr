"""Audio probing and deterministic FFmpeg normalization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

from .errors import EnvironmentError as TranscriberEnvironmentError
from .errors import InputAudioError


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}


@dataclass(frozen=True, slots=True)
class AudioInfo:
    path: Path
    duration_ms: int
    sample_rate: int
    channels: int
    codec: str


def _validate_source(path: Path) -> Path:
    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InputAudioError(f"unsupported audio extension: {source.suffix or '<none>'}")
    if not source.is_file():
        raise InputAudioError(f"audio file does not exist: {source}")
    if source.stat().st_size <= 0:
        raise InputAudioError(f"audio file is empty: {source}")
    return source


def probe_audio(path: Path, max_seconds: float) -> AudioInfo:
    source = _validate_source(path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,sample_rate,channels",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TranscriberEnvironmentError("ffprobe is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise InputAudioError(f"ffprobe timed out for audio file: {source}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "unknown ffprobe error").strip()
        raise InputAudioError(f"unable to decode audio metadata: {detail}")
    try:
        payload = json.loads(completed.stdout)
        duration_seconds = float(payload["format"]["duration"])
        audio_stream = next(
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "audio"
        )
        sample_rate = int(audio_stream["sample_rate"])
        channels = int(audio_stream["channels"])
        codec = str(audio_stream.get("codec_name") or "unknown")
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        raise InputAudioError("ffprobe returned incomplete audio metadata") from exc
    if duration_seconds <= 0:
        raise InputAudioError("audio duration must be greater than zero")
    if duration_seconds > max_seconds:
        raise InputAudioError(
            f"audio exceeds the 2 hours limit: {duration_seconds:.3f} seconds"
        )
    return AudioInfo(
        path=source.resolve(),
        duration_ms=int(round(duration_seconds * 1000)),
        sample_rate=sample_rate,
        channels=channels,
        codec=codec,
    )


def normalize_audio(
    source: Path,
    destination: Path,
    max_seconds: float = 7200,
) -> AudioInfo:
    source_info = probe_audio(source, max_seconds=max_seconds)
    output = Path(destination)
    if output.suffix.lower() != ".wav":
        raise InputAudioError("normalized destination must use the .wav extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(source_info.path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(300, int(source_info.duration_ms / 1000)),
            check=False,
        )
    except FileNotFoundError as exc:
        raise TranscriberEnvironmentError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise InputAudioError(f"ffmpeg timed out while normalizing: {source}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "unknown ffmpeg error").strip()
        raise InputAudioError(f"audio normalization failed: {detail}")
    normalized = probe_audio(output, max_seconds=max_seconds)
    if normalized.sample_rate != 16000 or normalized.channels != 1:
        raise InputAudioError(
            "normalized audio must be 16 kHz mono PCM WAV"
        )
    return normalized


class FfmpegAudioSlicer:
    def slice(
        self,
        source: Path,
        destination: Path,
        start_ms: int,
        end_ms: int,
    ) -> Path:
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("audio slice range is invalid")
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(source),
            "-t",
            f"{(end_ms - start_ms) / 1000:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(output),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(60, int((end_ms - start_ms) / 1000) * 2),
                check=False,
            )
        except FileNotFoundError as exc:
            raise TranscriberEnvironmentError(
                "ffmpeg is not installed or not on PATH"
            ) from exc
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or "slice output is missing").strip()
            raise InputAudioError(f"audio slicing failed: {detail}")
        return output
