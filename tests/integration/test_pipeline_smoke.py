import os
from pathlib import Path

import pytest

from funasr_transcriber.asr import FunAsrEngine
from funasr_transcriber.config import load_config
from funasr_transcriber.model_registry import build_asr_model
from funasr_transcriber.pipeline import TranscriptionPipeline
from funasr_transcriber.speaker_diarization import SpeakerDiarizer


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_MODEL_TESTS") != "1",
    reason="model integration tests are disabled",
)
def test_real_non_overlap_pipeline_returns_turns():
    fixture = Path(__file__).parents[1] / "fixtures" / "short_mandarin.wav"
    if not fixture.is_file():
        pytest.skip("short_mandarin.wav fixture is not installed")
    config = load_config()
    pipeline = TranscriptionPipeline(
        config,
        FunAsrEngine(build_asr_model(config.paths.model_dir, config.general.device)),
        SpeakerDiarizer.from_config(config),
    )

    result = pipeline.run_normalized(fixture)

    assert result.transcript.turns
    assert 1 <= result.transcript.speaker_count <= 10
    assert result.transcript.full_text
