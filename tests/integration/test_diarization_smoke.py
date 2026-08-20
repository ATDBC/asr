import os
from pathlib import Path

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.speaker_diarization import SpeakerDiarizer


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_MODEL_TESTS") != "1",
    reason="model integration tests are disabled",
)
def test_real_diarizer_is_deterministic():
    fixture = Path(__file__).parents[1] / "fixtures" / "short_two_speaker.wav"
    if not fixture.is_file():
        pytest.skip("short_two_speaker.wav fixture is not installed")
    config = load_config()
    diarizer = SpeakerDiarizer.from_config(config)

    first = diarizer.diarize(fixture)
    second = diarizer.diarize(fixture)

    assert first == second
    assert 1 <= len({segment.speaker_id for segment in first}) <= 10
