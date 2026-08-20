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

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")
        _validate_range(self.start_ms, self.end_ms)


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


@dataclass(frozen=True, slots=True)
class Transcript:
    language: str
    speaker_count: int
    turns: tuple[Turn, ...]
    warnings: tuple[str, ...]

    @property
    def full_text(self) -> str:
        return "".join(turn.text for turn in self.turns)
