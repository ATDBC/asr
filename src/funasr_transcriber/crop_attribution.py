"""Direct speaker ownership for independent original-mixture AED crops."""

from __future__ import annotations

import unicodedata

from .errors import TranscriptionError
from .firered_parallel import CropTranscript
from .schemas import (
    AlignmentConfidence,
    AsrResult,
    AttributedToken,
    RecognizedSentence,
    SpeakerSegment,
    TimedToken,
)


def _plain(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text)
        if not character.isspace()
    )


def attribute_crop_results(
    results: tuple[CropTranscript, ...],
    segments: tuple[SpeakerSegment, ...],
    duration_ms: int,
) -> tuple[AsrResult, tuple[AttributedToken, ...]]:
    if duration_ms <= 0:
        raise TranscriptionError("invalid normalized audio duration")
    parents = {
        (segment.speaker_id, segment.start_ms, segment.end_ms)
        for segment in segments
    }
    ordered: list[
        tuple[int, int, int, int, TimedToken, AttributedToken]
    ] = []
    sentences: list[RecognizedSentence] = []
    vad_segments: list[tuple[int, int]] = []
    seen_indexes: set[int] = set()
    for transcript in results:
        crop, result = transcript.crop, transcript.result
        if crop.index in seen_indexes:
            raise TranscriptionError("duplicate speaker crop index")
        seen_indexes.add(crop.index)
        if (
            crop.start_ms < 0
            or crop.end_ms > duration_ms
            or crop.end_ms <= crop.start_ms
            or (
                crop.speaker_id,
                crop.start_ms,
                crop.end_ms,
            ) not in parents
        ):
            raise TranscriptionError(
                "speaker crop does not match an original parent interval"
            )
        crop_duration = crop.end_ms - crop.start_ms
        if result.untimed_sentences:
            raise TranscriptionError(
                "Nemotron crop returned untimed ASR text"
            )
        if _plain(result.text) != _plain(
            "".join(token.text for token in result.tokens)
        ):
            raise TranscriptionError(
                "speaker crop text disagrees with timed tokens"
            )
        for token in result.tokens:
            if (
                token.start_ms < 0
                or token.end_ms > crop_duration
            ):
                raise TranscriptionError(
                    "AED crop token is outside parent crop"
                )
            start = crop.start_ms + token.start_ms
            end = crop.start_ms + token.end_ms
            overlap = any(
                segment.speaker_id != crop.speaker_id
                and segment.start_ms < end
                and segment.end_ms > start
                for segment in segments
            )
            global_token = TimedToken(
                token.text, start, end, token.confidence
            )
            attributed = AttributedToken(
                text=token.text,
                start_ms=start,
                end_ms=end,
                speaker_id=crop.speaker_id,
                confidence=AlignmentConfidence.HIGH,
                source="nemotron_direct_crop",
                overlap=overlap,
            )
            ordered.append(
                (
                    start,
                    crop.speaker_id,
                    crop.index,
                    end,
                    global_token,
                    attributed,
                )
            )
        for sentence in result.sentences:
            if sentence.end_ms > crop_duration:
                raise TranscriptionError(
                    "AED crop sentence is outside parent crop"
                )
            sentences.append(
                RecognizedSentence(
                    sentence.text,
                    crop.start_ms + sentence.start_ms,
                    crop.start_ms + sentence.end_ms,
                    sentence.asr_confidence,
                )
            )
        for start, end in crop.pieces_ms:
            if start < 0 or end > crop_duration or end <= start:
                raise TranscriptionError(
                    "AED crop piece is outside parent crop"
                )
            vad_segments.append(
                (crop.start_ms + start, crop.start_ms + end)
            )
    ordered.sort(key=lambda item: item[:4])
    tokens = tuple(item[4] for item in ordered)
    attributed = tuple(item[5] for item in ordered)
    asr = AsrResult(
        text="".join(token.text for token in tokens),
        tokens=tokens,
        sentences=tuple(
            sorted(
                sentences,
                key=lambda item: (item.start_ms, item.end_ms, item.text),
            )
        ),
        vad_segments_ms=tuple(sorted(vad_segments)),
        duration_ms=duration_ms,
    )
    return asr, attributed
