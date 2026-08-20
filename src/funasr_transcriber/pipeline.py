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
from .schemas import AsrResult, SpeakerSegment, Transcript
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


class TranscriptionPipeline:
    def __init__(
        self,
        config: AppConfig,
        asr_engine: AsrEngine,
        diarizer: DiarizationEngine,
        local_redecoder: LocalRedecoder | None = None,
        overlap_enhancer: OverlapEnhancer | None = None,
    ):
        self.config = config
        self.asr_engine = asr_engine
        self.diarizer = diarizer
        self.local_redecoder = local_redecoder
        self.overlap_enhancer = overlap_enhancer

    def run_normalized(self, path: Path) -> PipelineResult:
        diagnostics = Diagnostics(
            runtime=collect_runtime_info(),
            config=asdict(self.config),
        )
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
        with diagnostics.stage("turns"):
            turns = build_turns(
                tuple(attributed),
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
