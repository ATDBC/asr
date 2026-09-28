from pathlib import Path

import pytest

from funasr_transcriber.crop_attribution import attribute_crop_results
from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.firered_parallel import CropTranscript
from funasr_transcriber.schemas import (
    AsrResult,
    RecognizedSentence,
    SpeakerSegment,
    TimedToken,
)
from funasr_transcriber.speaker_crops import SpeakerCrop


def _fixtures(tmp_path: Path):
    segments = (
        SpeakerSegment(1, 0, 6000),
        SpeakerSegment(2, 5000, 8000),
    )
    first = SpeakerCrop(
        0, 1, 0, 6000, tmp_path / "s1.wav", "a" * 64, ((0, 6000),)
    )
    second = SpeakerCrop(
        1, 2, 5000, 8000, tmp_path / "s2.wav", "b" * 64, ((0, 3000),)
    )
    return segments, first, second


def test_overlapping_crop_results_keep_both_speakers(tmp_path):
    segments, first, second = _fixtures(tmp_path)
    results = (
        CropTranscript(
            first,
            AsrResult(
                "甲",
                (TimedToken("甲", 5500, 5800),),
                sentences=(RecognizedSentence("甲", 5500, 5800, 0.9),),
                duration_ms=6000,
            ),
        ),
        CropTranscript(
            second,
            AsrResult(
                "乙",
                (TimedToken("乙", 500, 800),),
                sentences=(RecognizedSentence("乙", 500, 800, 0.9),),
                duration_ms=3000,
            ),
        ),
    )

    asr, attributed = attribute_crop_results(results, segments, 8000)

    assert [token.speaker_id for token in attributed] == [1, 2]
    assert all(token.overlap for token in attributed)
    assert [token.start_ms for token in attributed] == [5500, 5500]
    assert asr.text == "甲乙"


def test_overlapping_crop_sentences_remain_atomic_in_final_turns(tmp_path):
    from funasr_transcriber.config import load_config
    from funasr_transcriber.pipeline import TranscriptionPipeline

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[diarization]\nbackend = "nemotron3"\n',
        encoding="utf-8",
    )
    pipeline = TranscriptionPipeline(load_config(config_file))
    parents = (
        SpeakerSegment(1, 680, 4000),
        SpeakerSegment(2, 2110, 4250),
    )
    results = (
        CropTranscript(
            SpeakerCrop(
                0, 1, 680, 4000, tmp_path / "s1.wav",
                "a" * 64, ((0, 3320),)
            ),
            AsrResult(
                "就过一阵子就又火了。",
                (
                    TimedToken("就过一阵子", 1420, 2400, 0.9),
                    TimedToken("就又火了。", 2400, 3285, 0.9),
                ),
                sentences=(
                    RecognizedSentence(
                        "就过一阵子就又火了。", 1420, 3285, 0.9
                    ),
                ),
                duration_ms=3320,
            ),
        ),
        CropTranscript(
            SpeakerCrop(
                1, 2, 2110, 4250, tmp_path / "s2.wav",
                "b" * 64, ((0, 2140),)
            ),
            AsrResult(
                "对真的是被诱惑了是。",
                (
                    TimedToken("对真的是", 60, 820, 0.8),
                    TimedToken("被诱惑了是。", 820, 2085, 0.8),
                ),
                sentences=(
                    RecognizedSentence(
                        "对真的是被诱惑了是。", 60, 2085, 0.8
                    ),
                ),
                duration_ms=2140,
            ),
        ),
    )

    asr, attributed = attribute_crop_results(results, parents, 4250)
    result = pipeline.finish(
        asr, attributed, parents, pipeline.new_diagnostics()
    )

    assert [turn.text for turn in result.transcript.turns] == [
        "就过一阵子就又火了。",
        "对真的是被诱惑了是。",
    ]
    assert [turn.speaker_id for turn in result.transcript.turns] == [1, 2]
    assert all(turn.overlap for turn in result.transcript.turns)
    assert all(
        turn.confidence.value == "medium"
        for turn in result.transcript.turns
    )


def test_nonempty_crop_without_sentences_fails(tmp_path):
    segments, first, _ = _fixtures(tmp_path)
    results = (
        CropTranscript(
            first,
            AsrResult(
                "甲", (TimedToken("甲", 0, 500),), duration_ms=6000
            ),
        ),
    )

    with pytest.raises(TranscriptionError, match="without sentences"):
        attribute_crop_results(results, segments, 8000)


def test_crop_local_token_outside_parent_fails(tmp_path):
    segments, first, _ = _fixtures(tmp_path)
    results = (
        CropTranscript(
            first,
            AsrResult(
                "甲", (TimedToken("甲", 5900, 6100),), duration_ms=0
            ),
        ),
    )
    with pytest.raises(TranscriptionError, match="outside"):
        attribute_crop_results(results, segments, 8000)


def test_wrong_speaker_parent_is_not_reassigned(tmp_path):
    segments, first, _ = _fixtures(tmp_path)
    wrong = SpeakerCrop(
        0, 3, 0, 6000, first.path, first.sha256, first.pieces_ms
    )
    with pytest.raises(TranscriptionError, match="parent"):
        attribute_crop_results(
            (
                CropTranscript(
                    wrong,
                    AsrResult(
                        "甲", (TimedToken("甲", 0, 500),), duration_ms=6000
                    ),
                ),
            ),
            segments,
            8000,
        )

def test_pipeline_preserves_crop_index_ties_in_final_turns(tmp_path):
    from funasr_transcriber.config import load_config
    from funasr_transcriber.pipeline import TranscriptionPipeline

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[diarization]\nbackend = "nemotron3"\n',
        encoding="utf-8",
    )
    pipeline = TranscriptionPipeline(load_config(config_file))
    parents = (
        SpeakerSegment(1, 0, 6000),
        SpeakerSegment(1, 0, 7000),
    )
    results = (
        CropTranscript(
            SpeakerCrop(
                0, 1, 0, 6000, tmp_path / "a.wav",
                "a" * 64, ((0, 6000),)
            ),
            AsrResult(
                "甲",
                (TimedToken("甲", 500, 700),),
                sentences=(RecognizedSentence("甲", 500, 700, 0.9),),
                duration_ms=6000,
            ),
        ),
        CropTranscript(
            SpeakerCrop(
                1, 1, 0, 7000, tmp_path / "b.wav",
                "b" * 64, ((0, 7000),)
            ),
            AsrResult(
                "乙",
                (TimedToken("乙", 500, 600),),
                sentences=(RecognizedSentence("乙", 500, 600, 0.9),),
                duration_ms=7000,
            ),
        ),
    )
    asr, attributed = attribute_crop_results(results, parents, 7000)

    result = pipeline.finish(
        asr, attributed, parents, pipeline.new_diagnostics()
    )

    assert result.transcript.full_text == "甲乙"
