import pytest

from funasr_transcriber.alignment import attribute_tokens
from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.schemas import (
    AlignmentConfidence,
    SpeakerSegment,
    TimedToken,
)


def test_token_goes_to_maximum_overlap_speaker():
    result = attribute_tokens(
        tokens=(TimedToken("你好", 900, 1300),),
        speaker_segments=(
            SpeakerSegment(1, 0, 1000),
            SpeakerSegment(2, 1000, 2000),
        ),
        high_coverage_ratio=0.70,
        medium_coverage_ratio=0.50,
        nearest_gap_ms=300,
    )

    assert result[0].speaker_id == 2
    assert result[0].confidence is AlignmentConfidence.HIGH


def test_short_interjection_keeps_its_speaker():
    result = attribute_tokens(
        tokens=(
            TimedToken("开始。", 0, 800),
            TimedToken("对。", 900, 1100),
            TimedToken("继续。", 1200, 2000),
        ),
        speaker_segments=(
            SpeakerSegment(1, 0, 850),
            SpeakerSegment(2, 850, 1150),
            SpeakerSegment(1, 1150, 2100),
        ),
        high_coverage_ratio=0.70,
        medium_coverage_ratio=0.50,
        nearest_gap_ms=300,
    )

    assert [item.speaker_id for item in result] == [1, 2, 1]


def test_token_uses_nearest_speaker_inside_gap():
    result = attribute_tokens(
        tokens=(TimedToken("嗯", 1050, 1150),),
        speaker_segments=(SpeakerSegment(1, 0, 1000),),
        high_coverage_ratio=0.70,
        medium_coverage_ratio=0.50,
        nearest_gap_ms=300,
    )

    assert result[0].speaker_id == 1
    assert result[0].confidence is AlignmentConfidence.LOW


def test_token_preserves_text_when_diarization_has_an_unbounded_gap():
    result = attribute_tokens(
        tokens=(TimedToken("嗯", 2000, 2100),),
        speaker_segments=(SpeakerSegment(1, 0, 1000),),
        high_coverage_ratio=0.70,
        medium_coverage_ratio=0.50,
        nearest_gap_ms=300,
    )

    assert result[0].speaker_id == 1
    assert result[0].confidence is AlignmentConfidence.LOW
    assert result[0].source == "nearest_speaker_fallback"


def test_attribute_by_segment_uses_containing_segment():
    from funasr_transcriber.alignment import attribute_by_segment

    result = attribute_by_segment(
        tokens=(
            TimedToken("你好。", 100, 500),
            TimedToken("再见。", 800, 1100),
        ),
        segments=(SpeakerSegment(1, 0, 600), SpeakerSegment(2, 700, 1200)),
    )

    assert [item.speaker_id for item in result] == [1, 2]
    assert all(
        item.confidence is AlignmentConfidence.HIGH for item in result
    )
    assert all(item.source == "primary_asr" for item in result)


def test_attribute_by_segment_falls_back_to_nearest_in_gap():
    from funasr_transcriber.alignment import attribute_by_segment

    result = attribute_by_segment(
        tokens=(TimedToken("嗯。", 620, 680),),
        segments=(SpeakerSegment(1, 0, 600), SpeakerSegment(2, 700, 1200)),
    )

    assert result[0].speaker_id == 1
    assert result[0].source == "nearest_speaker_fallback"


def test_attribute_by_segment_rejects_empty_segments():
    from funasr_transcriber.alignment import attribute_by_segment

    with pytest.raises(TranscriptionError, match="no segments"):
        attribute_by_segment(tokens=(TimedToken("嗯。", 0, 100),), segments=())
