import os
from pathlib import Path

import pytest

from funasr_transcriber.asr import FunAsrEngine
from funasr_transcriber.model_registry import build_asr_model


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_MODEL_TESTS") != "1",
    reason="model integration tests are disabled",
)
def test_real_asr_returns_monotonic_tokens():
    fixture = Path(__file__).parents[1] / "fixtures" / "short_mandarin.wav"
    if not fixture.is_file():
        pytest.skip("short_mandarin.wav fixture is not installed")
    model_dir = Path(os.environ["FUNASR_TRANSCRIBER_MODEL_DIR"])
    result = FunAsrEngine(build_asr_model(model_dir, "cuda:0")).transcribe(fixture)

    assert result.text
    assert result.tokens
    assert all(
        current.end_ms <= following.start_ms
        for current, following in zip(result.tokens, result.tokens[1:])
    )
