from pathlib import Path

import pytest

from funasr_transcriber.asr import FunAsrEngine, parse_funasr_result
from funasr_transcriber.errors import TranscriptionError


def test_parse_sentence_raw_text_and_timestamps():
    result = parse_funasr_result(
        {
            "text": "今天好。",
            "sentence_info": [
                {
                    "text": "今天好。",
                    "raw_text": "今 天 好",
                    "start": 100,
                    "end": 900,
                    "timestamp": [[100, 300], [300, 600], [600, 900]],
                }
            ],
        }
    )

    assert result.text == "今天好。"
    assert [(t.text, t.start_ms, t.end_ms) for t in result.tokens] == [
        ("今", 100, 300),
        ("天", 300, 600),
        ("好。", 600, 900),
    ]
    assert result.untimed_sentences == ()


def test_parser_rejects_mismatched_raw_tokens():
    raw = {
        "text": "今天。",
        "sentence_info": [
            {
                "text": "今天。",
                "raw_text": "今 天",
                "start": 0,
                "end": 500,
                "timestamp": [[0, 200]],
            }
        ],
    }

    with pytest.raises(ValueError, match="timestamp count"):
        parse_funasr_result(raw)


def test_parser_preserves_untimed_sentence():
    result = parse_funasr_result(
        {
            "text": "没有时间。",
            "sentence_info": [
                {
                    "text": "没有时间。",
                    "start": 100,
                    "end": 800,
                }
            ],
        }
    )

    assert result.tokens == ()
    assert result.untimed_sentences[0].text == "没有时间。"
    assert result.untimed_sentences[0].start_ms == 100


def test_engine_requests_raw_text_and_sentence_timestamps(tmp_path: Path):
    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return [
                {
                    "text": "你好。",
                    "sentence_info": [
                        {
                            "text": "你好。",
                            "raw_text": "你 好",
                            "start": 0,
                            "end": 500,
                            "timestamp": [[0, 200], [200, 500]],
                        }
                    ],
                }
            ]

    model = FakeModel()
    engine = FunAsrEngine(model)
    result = engine.transcribe(tmp_path / "sample.wav")

    assert result.text == "你好。"
    assert model.kwargs["return_raw_text"] is True
    assert model.kwargs["sentence_timestamp"] is True
    assert model.kwargs["batch_size_s"] == 300


def test_engine_rejects_empty_result(tmp_path: Path):
    class EmptyModel:
        def generate(self, **kwargs):
            return []

    with pytest.raises(TranscriptionError, match="empty"):
        FunAsrEngine(EmptyModel()).transcribe(tmp_path / "sample.wav")

