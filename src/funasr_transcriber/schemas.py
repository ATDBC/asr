"""Immutable data exchanged between pipeline stages."""

from dataclasses import dataclass
from enum import StrEnum


class AlignmentConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def _validate_range(start_ms: int, end_ms: int) -> None:
    if start_ms < 0:
        raise ValueError("start_ms must be non-negative")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")


@dataclass(frozen=True, slots=True)
class TimedToken:
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")
        _validate_range(self.start_ms, self.end_ms)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must stay within 0..1")


@dataclass(frozen=True, slots=True)
class RecognizedSentence:
    text: str
    start_ms: int
    end_ms: int
    asr_confidence: float

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")
        _validate_range(self.start_ms, self.end_ms)
        if not 0.0 <= self.asr_confidence <= 1.0:
            raise ValueError("asr_confidence must stay within 0..1")


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    text: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")
        _validate_range(self.start_ms, self.end_ms)


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    speaker_id: int
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.speaker_id < 1:
            raise ValueError("speaker_id must start at 1")
        _validate_range(self.start_ms, self.end_ms)


@dataclass(frozen=True, slots=True)
class AttributedToken:
    text: str
    start_ms: int
    end_ms: int
    speaker_id: int
    confidence: AlignmentConfidence
    source: str = "primary_asr"
    overlap: bool = False

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")
        if self.speaker_id < 1:
            raise ValueError("speaker_id must start at 1")
        _validate_range(self.start_ms, self.end_ms)


@dataclass(frozen=True, slots=True)
class Turn:
    index: int
    speaker_id: int
    start_ms: int
    end_ms: int
    text: str
    source: str
    confidence: AlignmentConfidence
    overlap: bool

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("index must start at 1")
        if self.speaker_id < 1:
            raise ValueError("speaker_id must start at 1")
        if not self.text:
            raise ValueError("text must not be empty")
        _validate_range(self.start_ms, self.end_ms)


@dataclass(frozen=True, slots=True)
class AsrResult:
    text: str
    tokens: tuple[TimedToken, ...]
    untimed_sentences: tuple[SentenceSpan, ...] = ()
    sentences: tuple[RecognizedSentence, ...] = ()
    vad_segments_ms: tuple[tuple[int, int], ...] = ()
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        for start_ms, end_ms in self.vad_segments_ms:
            _validate_range(start_ms, end_ms)
        if not self.duration_ms:
            return
        ranges = [
            *(item.end_ms for item in self.tokens),
            *(item.end_ms for item in self.untimed_sentences),
            *(item.end_ms for item in self.sentences),
            *(end_ms for _, end_ms in self.vad_segments_ms),
        ]
        if any(end_ms > self.duration_ms for end_ms in ranges):
            raise ValueError("timing must not exceed duration_ms")


@dataclass(frozen=True, slots=True)
class Transcript:
    language: str
    speaker_count: int
    turns: tuple[Turn, ...]
    warnings: tuple[str, ...]

    @property
    def full_text(self) -> str:
        return "".join(turn.text for turn in self.turns)
