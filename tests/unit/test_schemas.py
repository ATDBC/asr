import pytest

from funasr_transcriber.schemas import (
    AlignmentConfidence,
    AsrResult,
    RecognizedSentence,
    TimedToken,
    Transcript,
    Turn,
)


def test_timed_token_rejects_invalid_range():
    with pytest.raises(ValueError, match="end_ms"):
        TimedToken(text="测试", start_ms=200, end_ms=100)


def test_asr_result_carries_firered_metadata():
    sentence = RecognizedSentence(
        "你好。",
        100,
        900,
        0.95,
    )
    result = AsrResult(
        "你好。",
        (TimedToken("你好。", 100, 900, 0.95),),
        sentences=(sentence,),
        vad_segments_ms=((80, 920),),
        duration_ms=1000,
    )

    assert result.tokens[0].confidence == 0.95
    assert result.sentences == (sentence,)


@pytest.mark.parametrize("confidence", (-0.01, 1.01))
def test_timed_token_rejects_invalid_confidence(confidence: float):
    with pytest.raises(ValueError, match="confidence"):
        TimedToken("测试", 0, 100, confidence)


def test_asr_result_rejects_timing_past_duration():
    with pytest.raises(ValueError, match="duration_ms"):
        AsrResult(
            "测试",
            (TimedToken("测试", 0, 1001, 0.9),),
            duration_ms=1000,
        )


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
