"""Unit tests for phonetic gate verification."""

from __future__ import annotations

import pytest

from funasr_transcriber.pinyin_gate import (
    get_pinyin_syllables,
    sequence_phonetic_similarity,
    split_syllable,
    syllable_distance,
    verify_phonetic_gate,
)


def test_split_syllable():
    assert split_syllable("zhu") == ("zh", "u")
    assert split_syllable("ren") == ("r", "en")
    assert split_syllable("ang") == ("", "ang")
    assert split_syllable("gong") == ("g", "ong")


def test_syllable_distance_identical():
    assert syllable_distance("dan", "dan") == 0.0


def test_syllable_distance_confusable_initials():
    # c and ch
    dist = syllable_distance("ci", "zhi")
    assert dist < 0.4


def test_syllable_distance_confusable_finals():
    # in and ing
    dist = syllable_distance("ping", "pin")
    assert dist < 0.3


def test_get_pinyin_syllables():
    assert get_pinyin_syllables("丹丹主任") == ["dan", "dan", "zhu", "ren"]
    assert get_pinyin_syllables("公屏六度") == ["gong", "ping", "liu", "du"]
    assert get_pinyin_syllables("下至！") == ["xia", "zhi"]


def test_phonetic_gate_homophone_and_near_homophone():
    # Identical pinyin
    res1 = verify_phonetic_gate("丹丹主人", "丹丹主任")
    assert res1.passed is True
    assert res1.similarity == 1.0

    # Near-homophone (ping -> ting)
    res2 = verify_phonetic_gate("公屏六度", "宫廷六度")
    assert res2.passed is True
    assert res2.similarity >= 0.85

    # Near-homophone (ci -> zhi)
    res3 = verify_phonetic_gate("下次", "下至")
    assert res3.passed is True
    assert res3.similarity >= 0.60

    # Fast-speech contraction (jiu -> zhu ren)
    res4 = verify_phonetic_gate("丹丹酒", "丹丹主任")
    assert res4.passed is True
    assert res4.similarity >= 0.65


def test_phonetic_gate_rejects_hallucinations():
    res1 = verify_phonetic_gate("我们去吃饭", "明天做手术")
    assert res1.passed is False
    assert res1.similarity < 0.35
    assert "below threshold" in (res1.rejection_reason or "")

    res2 = verify_phonetic_gate("买包", "隆胸")
    assert res2.passed is False
    assert res2.similarity < 0.30


def test_phonetic_gate_empty_inputs():
    res1 = verify_phonetic_gate("", "丹丹主任")
    assert res1.passed is False
    assert res1.rejection_reason == "original text is empty"

    res2 = verify_phonetic_gate("丹丹酒", "")
    assert res2.passed is False
    assert res2.rejection_reason == "candidate text is empty"


def test_phonetic_gate_custom_threshold():
    # If threshold is 0.95, near-homophone with 0.90 is rejected
    res = verify_phonetic_gate("公屏六度", "宫廷六度", threshold=0.98)
    assert res.passed is False
