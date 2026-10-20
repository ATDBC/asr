"""Unit tests for Qwen-Audio transcription client and parser."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from funasr_transcriber.qwen_client import (
    QwenSentence,
    QwenTranscribeError,
    QwenTranscriberClient,
    QwenWordToken,
    parse_qwen_transcription,
)


def test_parse_qwen_transcription_valid() -> None:
    sample_data = {
        "transcription": {
            "transcripts": [
                {
                    "sentences": [
                        {
                            "sentence_id": 1,
                            "speaker_id": 0,
                            "begin_time": 2090,
                            "end_time": 3010,
                            "text": "这是眼睛哈。",
                            "words": [
                                {
                                    "begin_time": 2090,
                                    "end_time": 2450,
                                    "text": "这是",
                                    "punctuation": "",
                                    "confidence": 1.0,
                                },
                                {
                                    "begin_time": 2450,
                                    "end_time": 2850,
                                    "text": "眼睛",
                                    "punctuation": "",
                                    "confidence": 1.0,
                                },
                                {
                                    "begin_time": 2850,
                                    "end_time": 3010,
                                    "text": "哈",
                                    "punctuation": "。",
                                    "confidence": 1.0,
                                },
                            ],
                        }
                    ]
                }
            ]
        }
    }

    result = parse_qwen_transcription(sample_data)
    assert len(result.sentences) == 1
    assert len(result.words) == 3

    s = result.sentences[0]
    assert s.sentence_id == 1
    assert s.speaker_id == 0
    assert s.start_ms == 2090
    assert s.end_ms == 3010
    assert s.text == "这是眼睛哈。"
    assert len(s.words) == 3

    w1 = s.words[0]
    assert w1.text == "这是"
    assert w1.start_ms == 2090
    assert w1.end_ms == 2450
    assert w1.punctuation == ""

    assert result.full_text == "这是眼睛哈。"


def test_parse_qwen_transcription_empty() -> None:
    result = parse_qwen_transcription({})
    assert len(result.sentences) == 0
    assert len(result.words) == 0
    assert result.full_text == ""


def test_qwen_word_token_validation() -> None:
    with pytest.raises(ValueError, match="start_ms must be non-negative"):
        QwenWordToken(text="测试", start_ms=-1, end_ms=100)

    with pytest.raises(ValueError, match="end_ms must be >= start_ms"):
        QwenWordToken(text="测试", start_ms=200, end_ms=100)


def test_qwen_client_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN_ASR_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(QwenTranscribeError, match="Missing Qwen API key"):
        QwenTranscriberClient(api_key="")
