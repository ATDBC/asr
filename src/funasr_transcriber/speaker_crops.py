"""Create-only original-mixture crops for Nemotron speaker intervals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable

import soundfile as sf

from .errors import TranscriptionError
from .schemas import SpeakerSegment


@dataclass(frozen=True, slots=True)
class SpeakerCrop:
    index: int
    speaker_id: int
    start_ms: int
    end_ms: int
    path: Path
    sha256: str
    pieces_ms: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CropPlan:
    crops: tuple[SpeakerCrop, ...]
    duration_ms: int
    short_count: int
    subpiece_count: int
    overlap_duration_ms: int
    warnings: tuple[str, ...]


def fixed_pieces(duration_ms: int) -> tuple[tuple[int, int], ...]:
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    return tuple(
        (start, min(start + 10_000, duration_ms))
        for start in range(0, duration_ms, 10_000)
    )


def _validated_vad_pieces(
    raw: object,
    duration_ms: int,
) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    pieces: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        start, end = item
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start != int(start)
            or end != int(end)
        ):
            return None
        start_ms, end_ms = int(start), int(end)
        if (
            start_ms < 0
            or end_ms <= start_ms
            or end_ms > duration_ms
            or end_ms - start_ms > 10_000
        ):
            return None
        pieces.append((start_ms, end_ms))
    pieces.sort()
    if any(
        left[1] > right[0]
        for left, right in zip(pieces, pieces[1:])
    ):
        return None
    return tuple(pieces)


def _overlap_duration_ms(segments: tuple[SpeakerSegment, ...]) -> int:
    events: dict[int, list[tuple[int, int]]] = {}
    for segment in segments:
        events.setdefault(segment.start_ms, []).append(
            (segment.speaker_id, 1)
        )
        events.setdefault(segment.end_ms, []).append(
            (segment.speaker_id, -1)
        )
    active: dict[int, int] = {}
    overlap = 0
    times = sorted(events)
    for index, time in enumerate(times[:-1]):
        for speaker, delta in events[time]:
            count = active.get(speaker, 0) + delta
            if count <= 0:
                active.pop(speaker, None)
            else:
                active[speaker] = count
        if len(active) >= 2:
            overlap += times[index + 1] - time
    return overlap


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_crop_plan(
    normalized: Path,
    segments: tuple[SpeakerSegment, ...],
    output_dir: Path,
    refine_vad: Callable[
        [Path, list[list[int]]], list[list[int]]
    ] | None,
) -> CropPlan:
    source = Path(normalized)
    info = sf.info(str(source))
    if (
        info.samplerate != 16000
        or info.channels != 1
        or info.subtype != "PCM_16"
    ):
        raise TranscriptionError(
            "speaker crops require 16 kHz mono PCM_16 normalized WAV"
        )
    samples, sample_rate = sf.read(
        str(source), dtype="int16", always_2d=False
    )
    if sample_rate != 16000 or samples.ndim != 1 or len(samples) <= 0:
        raise TranscriptionError("invalid normalized WAV")
    duration_ms = round(len(samples) / 16)
    for segment in segments:
        if (
            segment.start_ms < 0
            or segment.end_ms * 16 > len(samples)
        ):
            raise TranscriptionError(
                "speaker segment exceeds normalized WAV"
            )

    target_root = Path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    ordered = tuple(
        sorted(
            segments,
            key=lambda s: (s.start_ms, s.speaker_id, s.end_ms),
        )
    )
    targets = tuple(
        target_root / f"crop-{index:04d}-speaker-{segment.speaker_id}.wav"
        for index, segment in enumerate(ordered)
        if segment.end_ms - segment.start_ms >= 100
    )
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"speaker crop already exists: {target}")

    crops: list[SpeakerCrop] = []
    warnings: list[str] = []
    short_count = 0
    for index, segment in enumerate(ordered):
        crop_duration_ms = segment.end_ms - segment.start_ms
        if crop_duration_ms < 100:
            short_count += 1
            warnings.append(
                f"speaker segment {index} shorter than 100 ms; skipped"
            )
            continue
        path = (
            target_root
            / f"crop-{index:04d}-speaker-{segment.speaker_id}.wav"
        )
        crop_samples = samples[
            segment.start_ms * 16:segment.end_ms * 16
        ]
        with sf.SoundFile(
            str(path),
            mode="x",
            samplerate=16000,
            channels=1,
            format="WAV",
            subtype="PCM_16",
        ) as output:
            output.write(crop_samples)
        if crop_duration_ms > 10_000:
            pieces = None
            if refine_vad is not None:
                try:
                    raw_pieces = refine_vad(
                        path, [[0, crop_duration_ms]]
                    )
                    pieces = _validated_vad_pieces(
                        raw_pieces, crop_duration_ms
                    )
                except Exception as exc:
                    warnings.append(
                        f"speaker segment {index} VAD failed: {exc}"
                    )
            if pieces is None:
                pieces = fixed_pieces(crop_duration_ms)
                warnings.append(
                    f"speaker segment {index} used fixed 10 s pieces"
                )
        else:
            pieces = ((0, crop_duration_ms),)
        crops.append(
            SpeakerCrop(
                index=index,
                speaker_id=segment.speaker_id,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                path=path,
                sha256=_sha256(path),
                pieces_ms=pieces,
            )
        )
    return CropPlan(
        crops=tuple(crops),
        duration_ms=duration_ms,
        short_count=short_count,
        subpiece_count=sum(len(crop.pieces_ms) for crop in crops),
        overlap_duration_ms=_overlap_duration_ms(ordered),
        warnings=tuple(warnings),
    )
