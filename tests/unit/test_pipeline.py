from pathlib import Path

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.pipeline import (
    TranscriptionPipeline,
    attribute_primary,
    summarize_languages,
)
from funasr_transcriber.overlap import OverlapEnhancement
from funasr_transcriber.schemas import (
    AlignmentConfidence,
    AsrResult,
    AttributedToken,
    RecognizedSentence,
    SentenceSpan,
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


def test_language_summary_is_duration_weighted():
    summary = summarize_languages(
        (
            RecognizedSentence(
                "甲。", 0, 9000, "zh mandarin", 0.95, 0.9
            ),
            RecognizedSentence("Hi.", 9000, 10000, "en", 0.99, 0.9),
        )
    )

    assert summary.primary_language == "zh mandarin"
    assert summary.duration_ms == {"zh mandarin": 9000, "en": 1000}
    assert summary.duration_ratio["zh mandarin"] == pytest.approx(0.9)


def test_attribute_primary_rejects_untimed_firered_sentence():
    result = AsrResult(
        "原始句子。",
        (),
        untimed_sentences=(SentenceSpan("原始句子。", 0, 1000),),
        duration_ms=1000,
    )

    with pytest.raises(TranscriptionError, match="complete timestamps"):
        attribute_primary(
            result,
            (SpeakerSegment(1, 0, 1000),),
            load_config(),
        )


def test_staged_pipeline_records_lid_and_warnings(tmp_path: Path):
    config = load_config()
    asr_result = AsrResult(
        "你好。Hi.",
        (
            TimedToken("你好。", 0, 900, 0.95),
            TimedToken("Hi.", 1000, 1900, 0.90),
        ),
        sentences=(
            RecognizedSentence(
                "你好。", 0, 900, "zh mandarin", 0.95, 0.95
            ),
            RecognizedSentence("Hi.", 1000, 1900, "en", 0.70, 0.90),
        ),
        vad_segments_ms=((0, 1900),),
        duration_ms=2000,
    )
    segments = (SpeakerSegment(1, 0, 2000),)
    pipeline = TranscriptionPipeline(config)
    diagnostics = pipeline.new_diagnostics()

    attributed = attribute_primary(asr_result, segments, config)
    result = pipeline.finish(
        asr_result,
        attributed,
        segments,
        diagnostics,
    )

    assert result.transcript.full_text == "你好。Hi."
    assert diagnostics.asr["languages"]["primary"] == "en"
    assert diagnostics.asr["languages"]["duration_ratio"] == {
        "en": 0.5,
        "zh mandarin": 0.5,
    }
    assert any("primary language" in item for item in diagnostics.warnings)
    assert any("low LID confidence" in item for item in diagnostics.warnings)
