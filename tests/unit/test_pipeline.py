from pathlib import Path

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.pipeline import TranscriptionPipeline
from funasr_transcriber.overlap import OverlapEnhancement
from funasr_transcriber.schemas import (
    AlignmentConfidence,
    AsrResult,
    AttributedToken,
    SpeakerSegment,
    TimedToken,
)


class FakeAsr:
    def __init__(self, result=None):
        self.result = result or AsrResult(
            text="你好。您好。",
            tokens=(
                TimedToken("你好。", 0, 900),
                TimedToken("您好。", 1200, 2000),
            ),
        )

    def transcribe(self, path: Path) -> AsrResult:
        return self.result


class FakeDiarizer:
    def __init__(self, segments=None):
        self.segments = (
            segments
            if segments is not None
            else (
                SpeakerSegment(1, 0, 1000),
                SpeakerSegment(2, 1000, 2100),
            )
        )

    def diarize(self, path: Path):
        return self.segments


def test_pipeline_produces_two_speaker_transcript(tmp_path: Path):
    pipeline = TranscriptionPipeline(load_config(), FakeAsr(), FakeDiarizer())

    result = pipeline.run_normalized(tmp_path / "normalized.wav")

    assert result.transcript.speaker_count == 2
    assert [turn.speaker_id for turn in result.transcript.turns] == [1, 2]
    assert result.transcript.full_text == "你好。您好。"
    assert result.asr_text == "你好。您好。"
    assert result.diagnostics.stage_seconds["asr"] >= 0
    assert result.diagnostics.stage_seconds["diarization"] >= 0
    assert result.diagnostics.runtime["python"]
    assert result.diagnostics.config["general"]["device"] == "cuda:0"


def test_pipeline_rejects_text_integrity_mismatch(tmp_path: Path):
    asr = FakeAsr(
        AsrResult(
            text="原文。",
            tokens=(TimedToken("别的文字。", 0, 1000),),
        )
    )
    pipeline = TranscriptionPipeline(
        load_config(),
        asr,
        FakeDiarizer((SpeakerSegment(1, 0, 1000),)),
    )

    with pytest.raises(TranscriptionError, match="integrity"):
        pipeline.run_normalized(tmp_path / "normalized.wav")


def test_pipeline_rejects_missing_diarization(tmp_path: Path):
    pipeline = TranscriptionPipeline(load_config(), FakeAsr(), FakeDiarizer(()))

    with pytest.raises(TranscriptionError, match="no segments"):
        pipeline.run_normalized(tmp_path / "normalized.wav")


def test_pipeline_allows_validated_overlap_replacement(tmp_path: Path):
    class Enhancer:
        def enhance(self, path, tokens, speaker_segments):
            return OverlapEnhancement(
                tokens=(
                    AttributedToken(
                        "双路甲。",
                        0,
                        900,
                        1,
                        AlignmentConfidence.HIGH,
                        overlap=True,
                    ),
                    AttributedToken(
                        "双路乙。",
                        100,
                        800,
                        2,
                        AlignmentConfidence.HIGH,
                        overlap=True,
                    ),
                ),
                warnings=(),
                modified=True,
            )

    pipeline = TranscriptionPipeline(
        load_config(),
        FakeAsr(),
        FakeDiarizer(),
        overlap_enhancer=Enhancer(),
    )

    result = pipeline.run_normalized(tmp_path / "normalized.wav")

    assert result.transcript.full_text == "双路甲。双路乙。"
    assert all(turn.overlap for turn in result.transcript.turns)
