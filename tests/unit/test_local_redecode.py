from pathlib import Path

from funasr_transcriber.alignment import LocalRedecoder
from funasr_transcriber.config import load_config
from funasr_transcriber.pipeline import TranscriptionPipeline
from funasr_transcriber.schemas import (
    AsrResult,
    SentenceSpan,
    SpeakerSegment,
    TimedToken,
)


class FakeSlicer:
    def __init__(self):
        self.calls = []

    def slice(self, source, destination, start_ms, end_ms):
        self.calls.append((start_ms, end_ms))
        destination.write_bytes(b"audio")
        return destination


class SegmentAsr:
    def transcribe(self, path: Path):
        if "0001" in path.name:
            return AsrResult("你好。", (TimedToken("你好。", 0, 900),))
        return AsrResult("您好。", (TimedToken("您好。", 0, 900),))


def test_redecoder_cuts_at_speaker_boundaries(tmp_path: Path):
    slicer = FakeSlicer()
    redecoder = LocalRedecoder(SegmentAsr(), slicer, tmp_path)

    result = redecoder.redecode(
        audio_path=tmp_path / "source.wav",
        sentence=SentenceSpan("你好。您好。", 0, 2000),
        speaker_segments=(
            SpeakerSegment(1, 0, 1000),
            SpeakerSegment(2, 1000, 2000),
        ),
    )

    assert slicer.calls == [(0, 1000), (1000, 2000)]
    assert [item.speaker_id for item in result.tokens] == [1, 2]
    assert "".join(item.text for item in result.tokens) == "你好。您好。"
    assert result.tokens[1].start_ms == 1000
    assert result.warning is None


def test_redecoder_falls_back_when_local_text_does_not_match(tmp_path: Path):
    class WrongAsr:
        def transcribe(self, path: Path):
            return AsrResult("完全错误。", (TimedToken("完全错误。", 0, 500),))

    redecoder = LocalRedecoder(WrongAsr(), FakeSlicer(), tmp_path)
    result = redecoder.redecode(
        audio_path=tmp_path / "source.wav",
        sentence=SentenceSpan("原始句子。", 0, 2000),
        speaker_segments=(
            SpeakerSegment(1, 0, 1500),
            SpeakerSegment(2, 1500, 2000),
        ),
    )

    assert len(result.tokens) == 1
    assert result.tokens[0].text == "原始句子。"
    assert result.tokens[0].speaker_id == 1
    assert result.warning is not None


def test_pipeline_uses_redecoder_for_untimed_sentence(tmp_path: Path):
    class PrimaryAsr:
        def transcribe(self, path: Path):
            return AsrResult(
                "你好。您好。",
                (),
                (SentenceSpan("你好。您好。", 0, 2000),),
            )

    class Diarizer:
        def diarize(self, path: Path):
            return (
                SpeakerSegment(1, 0, 1000),
                SpeakerSegment(2, 1000, 2000),
            )

    redecoder = LocalRedecoder(SegmentAsr(), FakeSlicer(), tmp_path)
    pipeline = TranscriptionPipeline(
        load_config(),
        PrimaryAsr(),
        Diarizer(),
        local_redecoder=redecoder,
    )

    result = pipeline.run_normalized(tmp_path / "source.wav")

    assert [turn.speaker_id for turn in result.transcript.turns] == [1, 2]
    assert result.transcript.full_text == "你好。您好。"
