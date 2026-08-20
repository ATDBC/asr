import pytest

from funasr_transcriber.schemas import (
    AlignmentConfidence,
    TimedToken,
    Transcript,
    Turn,
)


def test_timed_token_rejects_invalid_range():
    with pytest.raises(ValueError, match="end_ms"):
        TimedToken(text="测试", start_ms=200, end_ms=100)


def test_transcript_full_text_follows_turn_order():
    transcript = Transcript(
        language="zh",
        speaker_count=2,
        turns=(
            Turn(
                index=1,
                speaker_id=1,
                start_ms=0,
                end_ms=1000,
                text="你好。",
                source="primary_asr",
                confidence=AlignmentConfidence.HIGH,
                overlap=False,
            ),
            Turn(
                index=2,
                speaker_id=2,
                start_ms=1100,
                end_ms=2000,
                text="您好。",
                source="primary_asr",
                confidence=AlignmentConfidence.HIGH,
                overlap=False,
            ),
        ),
        warnings=(),
    )

    assert transcript.full_text == "你好。您好。"
