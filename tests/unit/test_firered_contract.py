from copy import deepcopy

import pytest

from funasr_transcriber.firered_contract import (
    FireRedContractError,
    NoSpeechError,
    _sentence_owners_linear,
    normalize_official_result,
    validate_stable_response,
)


OFFICIAL = {
    "uttid": "req-1",
    "text": "你好，世界。Hello 2!",
    "sentences": [
        {
            "start_ms": 100,
            "end_ms": 900,
            "text": "你好，世界。",
            "asr_confidence": 0.95,
        },
        {
            "start_ms": 1000,
            "end_ms": 1800,
            "text": "Hello 2!",
            "asr_confidence": 0.91,
        },
    ],
    "vad_segments_ms": [(80, 920), (980, 1820)],
    "dur_s": 2.0,
    "words": [
        {"start_ms": 120, "end_ms": 240, "text": "你"},
        {"start_ms": 240, "end_ms": 360, "text": "好"},
        {"start_ms": 400, "end_ms": 520, "text": "世"},
        {"start_ms": 520, "end_ms": 700, "text": "界"},
        {"start_ms": 1050, "end_ms": 1450, "text": "hello"},
        {"start_ms": 1450, "end_ms": 1700, "text": "2"},
    ],
}


def test_normalize_attaches_punctuation_and_sentence_confidence():
    response = normalize_official_result("req-1", OFFICIAL)

    assert response["id"] == "req-1"
    assert response["ok"] is True
    assert [item["text"] for item in response["tokens"]] == [
        "你",
        "好，",
        "世",
        "界。",
        "Hello",
        "2!",
    ]
    assert [item["confidence"] for item in response["tokens"]] == (
        [0.95] * 4 + [0.91] * 2
    )
    assert response["duration_ms"] == 2000
    assert response["sentences"][0] == {
        "text": "你好，世界。",
        "start_ms": 100,
        "end_ms": 900,
        "asr_confidence": 0.95,
    }
    assert "language" not in response["sentences"][0]
    validate_stable_response(response)


def test_normalize_uses_nfkc_and_ignores_whitespace_for_integrity():
    raw = deepcopy(OFFICIAL)
    raw["text"] = "你好，世界。Ｈｅｌｌｏ　２！"
    raw["sentences"][1]["text"] = "Ｈｅｌｌｏ　２！"

    response = normalize_official_result("req-1", raw)

    assert [item["text"] for item in response["tokens"]][-2:] == [
        "Ｈｅｌｌｏ",
        "２！",
    ]


def test_leading_punctuation_attaches_to_following_token():
    raw = deepcopy(OFFICIAL)
    raw["text"] = "“你好，世界。Hello 2!"
    raw["sentences"][0]["text"] = "“你好，世界。"

    response = normalize_official_result("req-1", raw)

    assert response["tokens"][0]["text"] == "“你"


def test_empty_vad_is_no_speech():
    raw = deepcopy(OFFICIAL)
    raw["vad_segments_ms"] = []

    with pytest.raises(NoSpeechError, match="no speech"):
        normalize_official_result("req-1", raw)


def test_rejects_vad_segment_over_sixty_seconds():
    raw = deepcopy(OFFICIAL)
    raw["dur_s"] = 61.0
    raw["vad_segments_ms"] = [(0, 60001)]

    with pytest.raises(FireRedContractError, match="60"):
        normalize_official_result("req-1", raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda raw: raw.update(words=[]), "without words"),
        (
            lambda raw: raw["words"][0].update(text="错"),
            "aligned",
        ),
        (
            lambda raw: raw["words"][0].update(start_ms=-1),
            "bounds",
        ),
        (
            lambda raw: raw["words"][0].update(end_ms=2001),
            "bounds",
        ),
        (
            lambda raw: raw["words"][1].update(start_ms=100),
            "monotonic",
        ),
        (
            lambda raw: raw["sentences"][0].update(text="不一致。"),
            "aligned",
        ),
    ),
)
def test_rejects_incomplete_or_invalid_official_results(mutation, message):
    raw = deepcopy(OFFICIAL)
    mutation(raw)

    with pytest.raises(FireRedContractError, match=message):
        normalize_official_result("req-1", raw)


def test_rejects_word_that_has_no_sentence_owner():
    raw = deepcopy(OFFICIAL)
    raw["words"][4].update(start_ms=930, end_ms=950)

    with pytest.raises(FireRedContractError, match="sentence"):
        normalize_official_result("req-1", raw)


def test_stable_validator_rejects_text_token_mismatch():
    response = normalize_official_result("req-1", OFFICIAL)
    response["tokens"][0]["text"] = "错"

    with pytest.raises(FireRedContractError, match="full text"):
        validate_stable_response(response)


def test_sentence_owners_linear_walks_adjacent_ranges_once():
    tokens = [
        {"start_ms": 100, "end_ms": 200, "text": "甲"},
        {"start_ms": 900, "end_ms": 1000, "text": "乙"},
        {"start_ms": 1100, "end_ms": 1200, "text": "丙"},
    ]
    sentences = [
        {"start_ms": 0, "end_ms": 1000, "text": "甲乙"},
        {"start_ms": 1000, "end_ms": 2000, "text": "丙"},
    ]

    assert _sentence_owners_linear(tokens, sentences) == [0, 0, 1]


def test_sentence_owners_linear_rejects_gap_without_nested_scan():
    tokens = [{"start_ms": 1500, "end_ms": 1600, "text": "错"}]
    sentences = [
        {"start_ms": 0, "end_ms": 1000, "text": "前"},
        {"start_ms": 2000, "end_ms": 3000, "text": "后"},
    ]

    with pytest.raises(FireRedContractError, match="exactly one sentence"):
        _sentence_owners_linear(tokens, sentences)
