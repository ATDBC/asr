import os
from pathlib import Path
import unicodedata

import pytest

from funasr_transcriber.firered_client import FireRedWorkerClient


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_MODEL_TESTS") != "1",
    reason="model integration tests are disabled",
)
def test_real_asr_returns_monotonic_tokens(tmp_path: Path):
    fixture = Path(__file__).parents[1] / "fixtures" / "short_mandarin.wav"
    if not fixture.is_file():
        pytest.skip("short_mandarin.wav fixture is not installed")
    model_dir = Path(os.environ["FUNASR_TRANSCRIBER_MODEL_DIR"])
    worker_python = Path(
        os.environ.get(
            "FUNASR_TRANSCRIBER_ASR_WORKER_PYTHON",
            Path(__file__).parents[2] / ".venv-asr" / "bin" / "python",
        )
    )
    with FireRedWorkerClient(
        worker_python=worker_python,
        model_root=model_dir,
        device="cuda:0",
        timeout_seconds=1800,
        stderr_log_path=tmp_path / "firered-worker.log",
    ) as client:
        result = client.transcribe(fixture.resolve())

    assert result.text
    assert result.tokens
    assert result.sentences
    assert result.vad_segments_ms
    assert result.duration_ms > 0
    assert all(token.confidence is not None for token in result.tokens)
    assert all(sentence.language for sentence in result.sentences)
    assert all(
        0.0 <= sentence.language_confidence <= 1.0
        and 0.0 <= sentence.asr_confidence <= 1.0
        for sentence in result.sentences
    )
    assert any(
        unicodedata.category(character).startswith("P")
        for token in result.tokens
        for character in token.text
    )
    assert all(
        current.end_ms <= following.start_ms
        for current, following in zip(result.tokens, result.tokens[1:])
    )
