"""Stage orchestration for the non-overlap transcription path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
import unicodedata

from .alignment import LocalRedecoder, attribute_tokens
from .config import AppConfig
from .diagnostics import Diagnostics, collect_runtime_info
from .errors import TranscriptionError
from .overlap import OverlapEnhancement
from .schemas import (
    AsrResult,
    AttributedToken,
    SpeakerSegment,
    Transcript,
)
from .turns import build_turns


class AsrEngine(Protocol):
    def transcribe(self, path: Path) -> AsrResult:
        """Transcribe normalized audio."""


class DiarizationEngine(Protocol):
    def diarize(self, path: Path) -> tuple[SpeakerSegment, ...]:
        """Return global one-based speaker segments."""


class OverlapEnhancer(Protocol):
    def enhance(
        self,
        path: Path,
        tokens,
        speaker_segments: tuple[SpeakerSegment, ...],
    ) -> OverlapEnhancement:
        """Return validated overlap replacements or the original tokens."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    transcript: Transcript
    diagnostics: Diagnostics
    asr_text: str


def normalize_for_integrity(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(character for character in normalized if not character.isspace())


def attribute_primary(
    asr_result: AsrResult,
    speaker_segments: tuple[SpeakerSegment, ...],
    config: AppConfig,
) -> tuple[AttributedToken, ...]:
    if asr_result.untimed_sentences:
        raise TranscriptionError(
            "FireRed ASR requires complete timestamps for all text"
        )
    return attribute_tokens(
        tokens=asr_result.tokens,
        speaker_segments=speaker_segments,
        high_coverage_ratio=config.alignment.high_coverage_ratio,
        medium_coverage_ratio=config.alignment.medium_coverage_ratio,
        nearest_gap_ms=config.alignment.nearest_gap_ms,
    )


class TranscriptionPipeline:
    def __init__(
        self,
        config: AppConfig,
        asr_engine: AsrEngine | None = None,
        diarizer: DiarizationEngine | None = None,
        local_redecoder: LocalRedecoder | None = None,
        overlap_enhancer: OverlapEnhancer | None = None,
    ):
        self.config = config
        self.asr_engine = asr_engine
        self.diarizer = diarizer
        self.local_redecoder = local_redecoder
        self.overlap_enhancer = overlap_enhancer

    def new_diagnostics(self) -> Diagnostics:
        return Diagnostics(
            runtime=collect_runtime_info(),
            config=asdict(self.config),
        )

    def _record_asr_metadata(
        self,
        diagnostics: Diagnostics,
        asr_result: AsrResult,
    ) -> None:
        diagnostics.asr = {
            "duration_ms": asr_result.duration_ms,
            "vad_segments_ms": [list(item) for item in asr_result.vad_segments_ms],
            "sentences": [
                {
                    "text": sentence.text,
                    "start_ms": sentence.start_ms,
                    "end_ms": sentence.end_ms,
                    "asr_confidence": sentence.asr_confidence,
                }
                for sentence in asr_result.sentences
            ],
        }

    def finish(
        self,
        asr_result: AsrResult,
        attributed_tokens: tuple[AttributedToken, ...],
        speaker_segments: tuple[SpeakerSegment, ...],
        diagnostics: Diagnostics,
        overlap_enhancement: OverlapEnhancement | None = None,
    ) -> PipelineResult:
        self._record_asr_metadata(diagnostics, asr_result)
        attributed = attributed_tokens
        overlap_modified = False
        if overlap_enhancement is not None:
            attributed = overlap_enhancement.tokens
            overlap_modified = overlap_enhancement.modified
            for warning in overlap_enhancement.warnings:
                diagnostics.warn(warning)
        with diagnostics.stage("turns"):
            turns = build_turns(
                attributed,
                merge_gap_ms=int(
                    round(self.config.turns.merge_gap_seconds * 1000)
                ),
            )
        transcript = Transcript(
            language=self.config.general.language,
            speaker_count=len(
                {segment.speaker_id for segment in speaker_segments}
            ),
            turns=turns,
            warnings=tuple(diagnostics.warnings),
        )
        if (
            not overlap_modified
            and normalize_for_integrity(asr_result.text)
            != normalize_for_integrity(transcript.full_text)
        ):
            raise TranscriptionError(
                "text integrity check failed between ASR and final turns"
            )
        diagnostics.speaker_count = transcript.speaker_count
        return PipelineResult(transcript, diagnostics, asr_result.text)

    def run_normalized(self, path: Path) -> PipelineResult:
        if self.asr_engine is None or self.diarizer is None:
            raise TranscriptionError(
                "run_normalized requires ASR and diarization engines"
            )
        diagnostics = self.new_diagnostics()
        with diagnostics.stage("asr"):
            asr_result = self.asr_engine.transcribe(path)
        with diagnostics.stage("diarization"):
            speaker_segments = self.diarizer.diarize(path)
        if not speaker_segments:
            raise TranscriptionError("speaker diarization returned no segments")

        with diagnostics.stage("alignment"):
            attributed = list(attribute_tokens(
                tokens=asr_result.tokens,
                speaker_segments=speaker_segments,
                high_coverage_ratio=(
                    self.config.alignment.high_coverage_ratio
                ),
                medium_coverage_ratio=(
                    self.config.alignment.medium_coverage_ratio
                ),
                nearest_gap_ms=self.config.alignment.nearest_gap_ms,
            ))
            if asr_result.untimed_sentences and self.local_redecoder is None:
                raise TranscriptionError(
                    "untimed ASR sentences require local redecode support"
                )
            for sentence in asr_result.untimed_sentences:
                decoded = self.local_redecoder.redecode(
                    path,
                    sentence,
                    speaker_segments,
                )
                attributed.extend(decoded.tokens)
                if decoded.warning:
                    diagnostics.warn(decoded.warning)
        overlap_modified = False
        if self.overlap_enhancer is not None:
            with diagnostics.stage("overlap"):
                enhanced = self.overlap_enhancer.enhance(
                    path,
                    tuple(attributed),
                    speaker_segments,
                )
            attributed = list(enhanced.tokens)
            overlap_modified = enhanced.modified
            for warning in enhanced.warnings:
                diagnostics.warn(warning)
        enhancement = OverlapEnhancement(
            tokens=tuple(attributed),
            warnings=(),
            modified=overlap_modified,
        )
        return self.finish(
            asr_result,
            tuple(attributed),
            speaker_segments,
            diagnostics,
            overlap_enhancement=enhancement,
        )
