from funasr_transcriber.metrics import (
    cer,
    cpcer,
    evaluate_speaker_tracks,
    speaker_count_metrics,
)


def test_cer_uses_chinese_characters():
    assert cer("今天很好", "今天好") == 0.25


def test_cpcer_is_anonymous_to_speaker_ids():
    reference = {1: "你好", 2: "您好"}
    hypothesis = {7: "您好", 9: "你好"}

    assert cpcer(reference, hypothesis) == 0.0


def test_cpcer_penalizes_missing_speaker_text():
    reference = {1: "你好", 2: "您好"}
    hypothesis = {9: "你好"}

    assert cpcer(reference, hypothesis) == 0.5


def test_count_metrics_do_not_cancel_signed_errors():
    result = speaker_count_metrics([2, 2], [1, 3])

    assert result["signed_mean"] == 0.0
    assert result["mae"] == 1.0
    assert result["exact_accuracy"] == 0.0


def test_speaker_track_metrics_are_label_invariant():
    reference = [
        {"start": 0.0, "end": 2.0, "speaker": "A"},
        {"start": 2.0, "end": 4.0, "speaker": "B"},
    ]
    hypothesis = [
        {"start": 0.0, "end": 2.0, "speaker": 9},
        {"start": 2.0, "end": 4.0, "speaker": 7},
    ]

    result = evaluate_speaker_tracks(reference, hypothesis)

    assert result["weighted_purity"] == 1.0
    assert result["der_like"] == 0.0
