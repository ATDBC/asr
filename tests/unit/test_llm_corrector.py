"""Unit tests for DeepSeek LLM corrector with phonetic gating."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from funasr_transcriber.config import LLMConfig
from funasr_transcriber.diagnostics import Diagnostics
from funasr_transcriber.llm_corrector import DeepSeekCorrector
from funasr_transcriber.schemas import AlignmentConfidence, Transcript, Turn


def create_sample_transcript() -> Transcript:
    turns = (
        Turn(
            index=1,
            speaker_id=1,
            start_ms=0,
            end_ms=5000,
            text="你有了解过丹丹酒吗？我给你再次讲解一下。",
            source="primary_asr",
            confidence=AlignmentConfidence.HIGH,
            overlap=False,
        ),
        Turn(
            index=2,
            speaker_id=1,
            start_ms=5000,
            end_ms=10000,
            text="因为目前丹丹主任他是在手术，公屏六度三点零价格是一万二。",
            source="primary_asr",
            confidence=AlignmentConfidence.HIGH,
            overlap=False,
        ),
    )
    return Transcript(
        language="zh",
        speaker_count=1,
        turns=turns,
        warnings=(),
    )


def test_corrector_noop_when_key_missing_and_disabled():
    config = LLMConfig(
        enabled=False,
        provider="deepseek",
        api_key="",
        api_key_env="NON_EXISTENT_KEY_FOR_TEST_XYZ",
        base_url="https://api.deepseek.com",
        model="deepseek-v4.1-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    corrector = DeepSeekCorrector(config)
    transcript = create_sample_transcript()
    result = corrector.correct(transcript)

    assert result.transcript == transcript
    assert len(result.patches_applied) == 0
    assert len(result.patches_rejected) == 0


def test_corrector_warns_when_key_missing_and_enabled():
    config = LLMConfig(
        enabled=True,
        provider="deepseek",
        api_key="",
        api_key_env="NON_EXISTENT_KEY_FOR_TEST_XYZ",
        base_url="https://api.deepseek.com",
        model="deepseek-v4.1-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    corrector = DeepSeekCorrector(config)
    transcript = create_sample_transcript()
    result = corrector.correct(transcript)

    assert result.transcript == transcript
    assert any("not configured" in w for w in result.warnings)


def test_corrector_dry_run_patches_applied_and_rejected():
    config = LLMConfig(
        enabled=True,
        provider="deepseek",
        api_key="test-key",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-v4.1-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    corrector = DeepSeekCorrector(config)
    transcript = create_sample_transcript()

    dry_run_patches = [
        {
            "turn_index": 1,
            "original": "丹丹酒",
            "corrected": "丹丹主任",
            "reason": "上下文同音误识",
        },
        {
            "turn_index": 2,
            "original": "公屏六度",
            "corrected": "宫廷六度",
            "reason": "医美项目近音词",
        },
        {
            # Should be rejected by phonetic gate
            "turn_index": 2,
            "original": "在手术",
            "corrected": "喝咖啡",
            "reason": "语义幻觉",
        },
        {
            # Non-existent turn index
            "turn_index": 99,
            "original": "foo",
            "corrected": "bar",
            "reason": "invalid turn",
        },
        {
            # Text not in turn
            "turn_index": 1,
            "original": "不存在的词",
            "corrected": "替换词",
            "reason": "missing text",
        },
    ]

    result = corrector.correct(transcript, dry_run_patches=dry_run_patches)

    # Applied checks
    assert len(result.patches_applied) == 2
    applied_originals = [p.original for p in result.patches_applied]
    assert "丹丹酒" in applied_originals
    assert "公屏六度" in applied_originals

    # Check updated text in transcript
    assert "你有了解过丹丹主任吗？" in result.transcript.turns[0].text
    assert "宫廷六度三点零价格是一万二" in result.transcript.turns[1].text
    assert result.transcript.turns[0].source == "llm_corrected"

    # Rejected checks
    assert len(result.patches_rejected) == 3
    rejected_reasons = [p.rejection_reason for p in result.patches_rejected]
    assert any("below threshold" in r for r in rejected_reasons if r)
    assert any("turn_index 99 does not exist" in r for r in rejected_reasons if r)
    assert any("not found in turn text" in r for r in rejected_reasons if r)


def test_corrector_mock_api_call():
    config = LLMConfig(
        enabled=True,
        provider="deepseek",
        api_key="sk-test-key-12345",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-v4.1-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    corrector = DeepSeekCorrector(config)
    transcript = create_sample_transcript()

    mock_llm_payload = {
        "entities": ["丹丹主任", "宫廷六度"],
        "patches": [
            {
                "turn_index": 1,
                "original": "丹丹酒",
                "corrected": "丹丹主任",
                "reason": "近音纠错",
            }
        ],
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(mock_llm_payload),
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = corrector.correct(transcript)

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        sent_json = call_kwargs["json"]
        assert sent_json["model"] == "deepseek-flash"
        assert sent_json["thinking"] == {"type": "disabled"}
        assert sent_json["temperature"] == 0.1
        assert "Authorization" in call_kwargs["headers"]
        assert "Bearer sk-test-key-12345" == call_kwargs["headers"]["Authorization"]

        assert len(result.patches_applied) == 1
        assert result.patches_applied[0].original == "丹丹酒"
        assert result.patches_applied[0].corrected == "丹丹主任"
        assert "丹丹主任" in result.transcript.turns[0].text
        assert "丹丹主任" in result.entities_discovered


def test_corrector_api_network_error_fallback():
    config = LLMConfig(
        enabled=True,
        provider="deepseek",
        api_key="sk-test-key-12345",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-v4.1-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    corrector = DeepSeekCorrector(config)
    transcript = create_sample_transcript()

    with patch(
        "httpx.Client.post",
        side_effect=httpx.ConnectError("Connection refused to api.deepseek.com"),
    ):
        result = corrector.correct(transcript)

        # Fallback without crash
        assert result.transcript.turns[0].text == transcript.turns[0].text
        assert len(result.patches_applied) == 0
        assert any("Connection refused" in w for w in result.warnings)


def test_corrector_mimo_config_and_mock_api_call():
    config = LLMConfig(
        enabled=True,
        provider="mimo",
        api_key="mimo-test-key-999",
        api_key_env="MIMO_API_KEY",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model="mimo-v2.6-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    corrector = DeepSeekCorrector(config)
    transcript = create_sample_transcript()

    mock_llm_payload = {
        "entities": ["丹丹主任"],
        "patches": [
            {
                "turn_index": 1,
                "original": "丹丹酒",
                "corrected": "丹丹主任",
                "reason": "同音词纠错",
            }
        ],
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(mock_llm_payload),
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = corrector.correct(transcript)

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        url = mock_post.call_args.args[0]
        assert url == "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
        sent_json = call_kwargs["json"]
        assert sent_json["model"] == "mimo-v2.6-flash"
        assert "thinking" not in sent_json
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer mimo-test-key-999"

        assert result.provider == "mimo"
        assert result.model == "mimo-v2.6-flash"
        assert len(result.patches_applied) == 1
        assert result.patches_applied[0].corrected == "丹丹主任"


def test_corrector_dashscope_config_and_mock_api_call():
    config = LLMConfig(
        enabled=True,
        provider="dashscope",
        api_key="dashscope-test-key-777",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    corrector = DeepSeekCorrector(config)
    transcript = create_sample_transcript()

    mock_llm_payload = {
        "entities": ["丹丹主任"],
        "patches": [
            {
                "turn_index": 1,
                "original": "丹丹酒",
                "corrected": "丹丹主任",
                "reason": "上下文修正",
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(mock_llm_payload, ensure_ascii=False)
                }
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        result = corrector.correct(transcript)

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        url = mock_post.call_args.args[0]
        assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        sent_json = call_kwargs["json"]
        assert sent_json["model"] == "qwen3.7-flash"
        assert "thinking" not in sent_json
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer dashscope-test-key-777"

        assert result.provider == "dashscope"
        assert result.model == "qwen3.7-flash"
        assert len(result.patches_applied) == 1
        assert result.patches_applied[0].corrected == "丹丹主任"

def test_llm_config_resolve_api_key_for_mimo_and_deepseek(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "env-mimo-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-dashscope-key")
    monkeypatch.setenv("QWEN_API_KEY", "env-qwen-key")

    cfg_mimo = LLMConfig(
        enabled=True,
        provider="mimo",
        api_key="",
        api_key_env="MIMO_API_KEY",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model="mimo-v2.6-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    assert cfg_mimo.resolve_api_key() == "env-mimo-key"

    cfg_mimo_fallback_empty = LLMConfig(
        enabled=True,
        provider="mimo",
        api_key="",
        api_key_env="",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model="mimo-v2.6-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    assert cfg_mimo_fallback_empty.resolve_api_key() == "env-mimo-key"

    cfg_mimo_fallback_default = LLMConfig(
        enabled=True,
        provider="mimo",
        api_key="",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model="mimo-v2.6-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    assert cfg_mimo_fallback_default.resolve_api_key() == "env-mimo-key"

    cfg_custom_missing = LLMConfig(
        enabled=True,
        provider="mimo",
        api_key="",
        api_key_env="NON_EXISTENT_CUSTOM_KEY",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model="mimo-v2.6-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    assert cfg_custom_missing.resolve_api_key() == ""

    cfg_deepseek = LLMConfig(
        enabled=True,
        provider="deepseek",
        api_key="",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-v4.1-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    assert cfg_deepseek.resolve_api_key() == "env-deepseek-key"

    cfg_dashscope = LLMConfig(
        enabled=True,
        provider="dashscope",
        api_key="",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    assert cfg_dashscope.resolve_api_key() == "env-dashscope-key"

    cfg_qwen_fallback_default = LLMConfig(
        enabled=True,
        provider="qwen",
        api_key="",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-flash",
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=60,
        pinyin_gate_threshold=0.5,
        max_context_turns=100,
    )
    assert cfg_qwen_fallback_default.resolve_api_key() == "env-dashscope-key"
