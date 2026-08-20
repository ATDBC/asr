from pathlib import Path
import shutil
import struct
import wave

import pytest

from funasr_transcriber.audio import normalize_audio, probe_audio


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed",
)
def test_normalize_converts_8khz_stereo_to_16khz_mono(tmp_path: Path):
    source = tmp_path / "stereo_8k.wav"
    destination = tmp_path / "normalized.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        frame = struct.pack("<hh", 1000, -1000)
        stream.writeframes(frame * 8000)

    normalize_audio(source, destination)
    info = probe_audio(destination, max_seconds=7200)

    assert info.sample_rate == 16000
    assert info.channels == 1
    assert 990 <= info.duration_ms <= 1010
