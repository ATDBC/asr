import json
from pathlib import Path

import pytest

from funasr_transcriber.audio import FfmpegAudioSlicer, probe_audio
from funasr_transcriber.errors import InputAudioError


class Completed:
    returncode = 0
    stderr = ""

    def __init__(self, payload):
        self.stdout = json.dumps(payload)


def test_probe_rejects_audio_over_two_hours(monkeypatch, tmp_path: Path):
    source = tmp_path / "long.mp3"
    source.write_bytes(b"x")
    monkeypatch.setattr(
        "funasr_transcriber.audio.subprocess.run",
        lambda *args, **kwargs: Completed(
            {
                "format": {"duration": "7200.001"},
                "streams": [
                    {
                        "codec_type": "audio",
                        "sample_rate": "44100",
                        "channels": 2,
                        "codec_name": "mp3",
                    }
                ],
            }
        ),
    )

    with pytest.raises(InputAudioError, match="2 hours"):
        probe_audio(source, max_seconds=7200)


def test_probe_reads_audio_stream(monkeypatch, tmp_path: Path):
    source = tmp_path / "sample.flac"
    source.write_bytes(b"x")
    monkeypatch.setattr(
        "funasr_transcriber.audio.subprocess.run",
        lambda *args, **kwargs: Completed(
            {
                "format": {"duration": "12.5"},
                "streams": [
                    {
                        "codec_type": "audio",
                        "sample_rate": "48000",
                        "channels": 1,
                        "codec_name": "flac",
                    }
                ],
            }
        ),
    )

    info = probe_audio(source, max_seconds=7200)

    assert info.duration_ms == 12_500
    assert info.sample_rate == 48_000
    assert info.channels == 1
    assert info.codec == "flac"


def test_probe_rejects_unsupported_extension(tmp_path: Path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"x")

    with pytest.raises(InputAudioError, match="unsupported"):
        probe_audio(source, max_seconds=7200)


def test_slicer_uses_exact_millisecond_boundaries(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.wav"
    destination = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    captured = {}

    class SliceDone:
        returncode = 0
        stdout = ""
        stderr = ""

    def run(command, **kwargs):
        captured["command"] = command
        destination.write_bytes(b"clip")
        return SliceDone()

    monkeypatch.setattr(
        "funasr_transcriber.audio.subprocess.run",
        run,
    )

    result = FfmpegAudioSlicer().slice(source, destination, 1250, 2750)

    assert result == destination
    assert captured["command"][captured["command"].index("-ss") + 1] == "1.250"
    assert captured["command"][captured["command"].index("-t") + 1] == "1.500"
