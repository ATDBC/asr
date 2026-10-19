"""Create-only original-mixture crops for Nemotron speaker intervals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import soundfile as sf

from .errors import TranscriptionError
from .schemas import SpeakerSegment

if TYPE_CHECKING:
    from .overlap import OverlapPreparation


@dataclass(frozen=True, slots=True)
class SpeakerCrop:
    index: int
    speaker_id: int
    start_ms: int
    end_ms: int
    path: Path
    sha256: str
    pieces_ms: tuple[tuple[int, int], ...]
    audio_offset_ms: int = 0


@dataclass(frozen=True, slots=True)
class CropPlan:
    crops: tuple[SpeakerCrop, ...]
    duration_ms: int
    short_count: int
    subpiece_count: int
    overlap_duration_ms: int
    warnings: tuple[str, ...]
    spliced_overlap_count: int = 0


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


def _splice_overlap_into_crop(
    crop_samples: np.ndarray,
    segment: SpeakerSegment,
    overlap_preparation: OverlapPreparation,
    track_cache: dict[Path, np.ndarray],
    min_score: float,
    min_margin: float,
    min_rms: float,
    warnings: list[str],
) -> int:
    """Splice clean separated tracks into crop_samples for this speaker segment.

    Applies a 5ms (80 sample) linear crossfade at boundary transitions to prevent
    acoustic discontinuities/pops.

    Returns the number of overlap regions successfully spliced into this crop.
    """
    spliced_count = 0
    seg_start = segment.start_ms
    seg_end = segment.end_ms

    for window in overlap_preparation.windows:
        win_start = window.start_ms
        win_end = window.end_ms
        intersect_start = max(seg_start, win_start)
        intersect_end = min(seg_end, win_end)
        if intersect_start >= intersect_end:
            continue

        matched_track = None
        for track in window.tracks:
            if track.speaker_id == segment.speaker_id:
                if (
                    track.score >= min_score
                    and track.margin >= min_margin
                    and track.rms >= min_rms
                ):
                    matched_track = track
                    break

        if matched_track is None:
            continue

        track_path = Path(matched_track.path).resolve()
        if track_path not in track_cache:
            try:
                data, sr = sf.read(str(track_path), dtype="int16", always_2d=False)
                if sr != 16000:
                    warnings.append(
                        f"overlap track {track_path.name} samplerate {sr} != 16000; skipped"
                    )
                    continue
                if data.ndim > 1:
                    data = data.mean(axis=1).astype(np.int16)
                track_cache[track_path] = data
            except Exception as exc:
                warnings.append(
                    f"overlap track {track_path.name} read failed: {exc}; kept original"
                )
                continue

        track_audio = track_cache[track_path]

        crop_start_sample = (intersect_start - seg_start) * 16
        crop_end_sample = (intersect_end - seg_start) * 16
        track_start_sample = (intersect_start - win_start) * 16
        track_end_sample = (intersect_end - win_start) * 16

        splice_samples = crop_end_sample - crop_start_sample
        if splice_samples <= 0:
            continue

        if track_start_sample >= len(track_audio):
            continue
        track_slice = track_audio[track_start_sample:track_end_sample]
        if len(track_slice) < splice_samples:
            track_slice = np.pad(
                track_slice, (0, splice_samples - len(track_slice))
            )
        elif len(track_slice) > splice_samples:
            track_slice = track_slice[:splice_samples]

        fade_len = min(80, splice_samples // 2)
        if fade_len > 0:
            spliced_part = track_slice.astype(np.float32)
            orig_part = crop_samples[
                crop_start_sample:crop_end_sample
            ].astype(np.float32)

            if intersect_start > seg_start:
                fade_in = np.linspace(
                    0.0, 1.0, fade_len, endpoint=False, dtype=np.float32
                )
                spliced_part[:fade_len] = (
                    orig_part[:fade_len] * (1.0 - fade_in)
                    + spliced_part[:fade_len] * fade_in
                )

            if intersect_end < seg_end:
                fade_out = np.linspace(
                    1.0, 0.0, fade_len, endpoint=False, dtype=np.float32
                )
                spliced_part[-fade_len:] = (
                    orig_part[-fade_len:] * (1.0 - fade_out)
                    + spliced_part[-fade_len:] * fade_out
                )

            crop_samples[crop_start_sample:crop_end_sample] = np.clip(
                np.round(spliced_part), -32768, 32767
            ).astype(np.int16)
        else:
            crop_samples[crop_start_sample:crop_end_sample] = track_slice

        spliced_count += 1

    return spliced_count


def build_crop_plan(
    normalized: Path,
    segments: tuple[SpeakerSegment, ...],
    output_dir: Path | None = None,
    refine_vad: Callable[
        [Path, list[list[int]]], list[list[int]]
    ] | None = None,
    overlap_preparation: OverlapPreparation | None = None,
    min_score: float = 0.5,
    min_margin: float = 0.05,
    min_rms: float = 0.005,
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

    target_root = Path(output_dir) if output_dir is not None else None
    ordered = tuple(
        sorted(
            segments,
            key=lambda s: (s.start_ms, s.speaker_id, s.end_ms),
        )
    )
    if target_root is not None:
        target_root.mkdir(parents=True, exist_ok=True)
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
    track_cache: dict[Path, np.ndarray] = {}
    total_spliced_count = 0

    for index, segment in enumerate(ordered):
        crop_duration_ms = segment.end_ms - segment.start_ms
        if crop_duration_ms < 100:
            short_count += 1
            warnings.append(
                f"speaker segment {index} shorter than 100 ms; skipped"
            )
            continue

        crop_samples = None
        spliced_in_crop = 0
        if overlap_preparation is not None:
            crop_samples = samples[
                segment.start_ms * 16 : segment.end_ms * 16
            ].copy()
            spliced_in_crop = _splice_overlap_into_crop(
                crop_samples,
                segment,
                overlap_preparation,
                track_cache,
                min_score,
                min_margin,
                min_rms,
                warnings,
            )
            total_spliced_count += spliced_in_crop

        if target_root is not None:
            path = (
                target_root
                / f"crop-{index:04d}-speaker-{segment.speaker_id}.wav"
            )
            if crop_samples is None:
                crop_samples = samples[
                    segment.start_ms * 16 : segment.end_ms * 16
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
            sha256 = _sha256(path)
            vad_target_path = path
            vad_target_range = [[0, crop_duration_ms]]
            audio_offset_ms = 0
        else:
            if spliced_in_crop > 0:
                spliced_dir = source.parent / "spliced-crops"
                spliced_dir.mkdir(parents=True, exist_ok=True)
                path = (
                    spliced_dir
                    / f"spliced-crop-{index:04d}-speaker-{segment.speaker_id}.wav"
                )
                with sf.SoundFile(
                    str(path),
                    mode="w",
                    samplerate=16000,
                    channels=1,
                    format="WAV",
                    subtype="PCM_16",
                ) as output:
                    output.write(crop_samples)
                sha256 = ""
                vad_target_path = path
                vad_target_range = [[0, crop_duration_ms]]
                audio_offset_ms = 0
            else:
                path = source
                sha256 = ""
                vad_target_path = source
                vad_target_range = [[segment.start_ms, segment.end_ms]]
                audio_offset_ms = segment.start_ms

        if crop_duration_ms > 10_000:
            pieces = None
            if refine_vad is not None:
                try:
                    raw_pieces = refine_vad(
                        vad_target_path, vad_target_range
                    )
                    if target_root is None and vad_target_path == source:
                        rel_pieces = [
                            [
                                max(0, p[0] - segment.start_ms),
                                min(crop_duration_ms, p[1] - segment.start_ms),
                            ]
                            for p in raw_pieces
                            if min(crop_duration_ms, p[1] - segment.start_ms)
                            > max(0, p[0] - segment.start_ms)
                        ]
                        pieces = _validated_vad_pieces(
                            rel_pieces, crop_duration_ms
                        )
                    else:
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
                sha256=sha256,
                pieces_ms=pieces,
                audio_offset_ms=audio_offset_ms,
            )
        )
    return CropPlan(
        crops=tuple(crops),
        duration_ms=duration_ms,
        short_count=short_count,
        subpiece_count=sum(len(crop.pieces_ms) for crop in crops),
        overlap_duration_ms=_overlap_duration_ms(ordered),
        warnings=tuple(warnings),
        spliced_overlap_count=total_spliced_count,
    )
