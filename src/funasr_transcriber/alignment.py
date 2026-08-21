"""Assign timestamped ASR tokens to global speaker segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import unicodedata

from .errors import TranscriptionError
from .schemas import (
    AlignmentConfidence,
    AsrResult,
    AttributedToken,
    SentenceSpan,
    SpeakerSegment,
    TimedToken,
)


def _overlap_ms(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> int:
    return max(0, min(first_end, second_end) - max(first_start, second_start))


def _distance_ms(token: TimedToken, segment: SpeakerSegment) -> int:
    if token.end_ms <= segment.start_ms:
        return segment.start_ms - token.end_ms
    if segment.end_ms <= token.start_ms:
        return token.start_ms - segment.end_ms
    return 0


def _midpoint_inside(token: TimedToken, segment: SpeakerSegment) -> bool:
    midpoint = (token.start_ms + token.end_ms) / 2
    return segment.start_ms <= midpoint < segment.end_ms


def attribute_tokens(
    tokens: tuple[TimedToken, ...],
    speaker_segments: tuple[SpeakerSegment, ...],
    high_coverage_ratio: float,
    medium_coverage_ratio: float,
    nearest_gap_ms: int,
) -> tuple[AttributedToken, ...]:
    if tokens and not speaker_segments:
        raise TranscriptionError("speaker diarization returned no segments")

    output: list[AttributedToken] = []
    for token in sorted(tokens, key=lambda item: (item.start_ms, item.end_ms)):
        duration = token.end_ms - token.start_ms
        source = "primary_asr"
        overlaps = sorted(
            (
                (
                    _overlap_ms(
                        token.start_ms,
                        token.end_ms,
                        segment.start_ms,
                        segment.end_ms,
                    ),
                    segment,
                )
                for segment in speaker_segments
            ),
            key=lambda item: (-item[0], item[1].speaker_id),
        )
        best_overlap, best_segment = overlaps[0]
        if best_overlap <= 0:
            nearest = min(
                speaker_segments,
                key=lambda segment: (
                    _distance_ms(token, segment),
                    segment.speaker_id,
                ),
            )
            speaker_id = nearest.speaker_id
            confidence = AlignmentConfidence.LOW
            if _distance_ms(token, nearest) > nearest_gap_ms:
                source = "nearest_speaker_fallback"
        else:
            speaker_id = best_segment.speaker_id
            coverage = best_overlap / duration
            second_overlap = overlaps[1][0] if len(overlaps) > 1 else 0
            separation = (best_overlap - second_overlap) / duration
            if (
                coverage >= high_coverage_ratio
                and separation >= high_coverage_ratio - medium_coverage_ratio
            ):
                confidence = AlignmentConfidence.HIGH
            elif (
                coverage >= medium_coverage_ratio
                or _midpoint_inside(token, best_segment)
            ):
                confidence = AlignmentConfidence.MEDIUM
            else:
                confidence = AlignmentConfidence.LOW
        output.append(
            AttributedToken(
                text=token.text,
                start_ms=token.start_ms,
                end_ms=token.end_ms,
                speaker_id=speaker_id,
                confidence=confidence,
                source=source,
            )
        )
    return tuple(output)


class RedecodeAsrEngine(Protocol):
    def transcribe(self, path: Path) -> AsrResult:
        """Transcribe one locally sliced sentence fragment."""


class AudioSlicer(Protocol):
    def slice(
        self,
        source: Path,
        destination: Path,
        start_ms: int,
        end_ms: int,
    ) -> Path:
        """Write the requested audio interval and return its path."""


@dataclass(frozen=True, slots=True)
class RedecodeResult:
    tokens: tuple[AttributedToken, ...]
    warning: str | None


def _normalized_text(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text)
        if not character.isspace()
    )


def _sentence_overlap_ms(
    sentence: SentenceSpan,
    segment: SpeakerSegment,
) -> int:
    return _overlap_ms(
        sentence.start_ms,
        sentence.end_ms,
        segment.start_ms,
        segment.end_ms,
    )


def _dominant_segment(
    sentence: SentenceSpan,
    segments: tuple[SpeakerSegment, ...],
) -> SpeakerSegment:
    candidates = [
        segment
        for segment in segments
        if _sentence_overlap_ms(sentence, segment) > 0
    ]
    if not candidates:
        raise TranscriptionError(
            f"no speaker overlaps untimed sentence at {sentence.start_ms} ms"
        )
    return max(
        candidates,
        key=lambda segment: (
            _sentence_overlap_ms(sentence, segment),
            -segment.speaker_id,
        ),
    )


class LocalRedecoder:
    def __init__(
        self,
        asr_engine: RedecodeAsrEngine,
        slicer: AudioSlicer,
        work_dir: Path,
    ):
        self.asr_engine = asr_engine
        self.slicer = slicer
        self.work_dir = Path(work_dir)

    def _fallback(
        self,
        sentence: SentenceSpan,
        segments: tuple[SpeakerSegment, ...],
        reason: str,
    ) -> RedecodeResult:
        dominant = _dominant_segment(sentence, segments)
        return RedecodeResult(
            tokens=(
                AttributedToken(
                    text=sentence.text,
                    start_ms=sentence.start_ms,
                    end_ms=sentence.end_ms,
                    speaker_id=dominant.speaker_id,
                    confidence=AlignmentConfidence.LOW,
                    source="primary_asr",
                ),
            ),
            warning=reason,
        )

    def redecode(
        self,
        audio_path: Path,
        sentence: SentenceSpan,
        speaker_segments: tuple[SpeakerSegment, ...],
    ) -> RedecodeResult:
        candidates = [
            segment
            for segment in speaker_segments
            if _sentence_overlap_ms(sentence, segment) > 0
        ]
        if not candidates:
            raise TranscriptionError(
                f"no speaker overlaps untimed sentence at {sentence.start_ms} ms"
            )
        if len({segment.speaker_id for segment in candidates}) == 1:
            speaker = candidates[0].speaker_id
            return RedecodeResult(
                (
                    AttributedToken(
                        text=sentence.text,
                        start_ms=sentence.start_ms,
                        end_ms=sentence.end_ms,
                        speaker_id=speaker,
                        confidence=AlignmentConfidence.MEDIUM,
                        source="primary_asr",
                    ),
                ),
                None,
            )

        self.work_dir.mkdir(parents=True, exist_ok=True)
        attributed: list[AttributedToken] = []
        local_text_parts: list[str] = []
        try:
            for index, segment in enumerate(candidates, start=1):
                start_ms = max(sentence.start_ms, segment.start_ms)
                end_ms = min(sentence.end_ms, segment.end_ms)
                if end_ms <= start_ms:
                    continue
                clip_path = self.work_dir / f"segment-{index:04d}.wav"
                self.slicer.slice(
                    Path(audio_path),
                    clip_path,
                    start_ms,
                    end_ms,
                )
                local = self.asr_engine.transcribe(clip_path)
                local_text_parts.append(local.text)
                if not local.tokens:
                    return self._fallback(
                        sentence,
                        speaker_segments,
                        "local redecode returned no timed tokens",
                    )
                for token in local.tokens:
                    global_start = min(end_ms, start_ms + token.start_ms)
                    global_end = min(end_ms, start_ms + token.end_ms)
                    if global_end <= global_start:
                        continue
                    attributed.append(
                        AttributedToken(
                            text=token.text,
                            start_ms=global_start,
                            end_ms=global_end,
                            speaker_id=segment.speaker_id,
                            confidence=AlignmentConfidence.MEDIUM,
                            source="local_redecode",
                        )
                    )
        except Exception as exc:
            return self._fallback(
                sentence,
                speaker_segments,
                f"local redecode failed: {exc}",
            )

        if _normalized_text("".join(local_text_parts)) != _normalized_text(
            sentence.text
        ):
            return self._fallback(
                sentence,
                speaker_segments,
                "local redecode text did not match the primary sentence",
            )
        if not attributed:
            return self._fallback(
                sentence,
                speaker_segments,
                "local redecode produced no usable tokens",
            )
        return RedecodeResult(
            tuple(sorted(attributed, key=lambda item: item.start_ms)),
            None,
        )
