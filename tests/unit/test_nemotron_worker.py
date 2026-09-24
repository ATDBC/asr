from pathlib import Path

import pytest

from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.schemas import SpeakerSegment
from funasr_transcriber.nemotron_worker import (
    normalize_nemotron_segments,
    validate_job_wav_path,
)


@pytest.mark.parametrize(
    "raw",
    [
        {"Speaker": 0, "Start": float("nan"), "End": 1.0},
        {"Speaker": 0, "Start": -0.1, "End": 1.0},
        {"Speaker": 0, "Start": 1.0, "End": 0.5},
        {"Speaker": 0, "Start": 0.0, "End": 99.0},
        {"Speaker": 8, "Start": 0.0, "End": 1.0},
        {"Speaker": True, "Start": 0.0, "End": 1.0},
    ],
)
def test_invalid_segment_fails_before_crop(raw):
    with pytest.raises(TranscriptionError):
        normalize_nemotron_segments([raw], duration_ms=8000)


def test_overlap_is_preserved_and_unsorted_segments_are_sorted():
    segments = normalize_nemotron_segments(
        [
            {"Speaker": 1, "Start": 5.0, "End": 8.0},
            {"Speaker": 0, "Start": 0.0, "End": 6.0},
        ],
        duration_ms=8000,
    )
    assert segments == (
        SpeakerSegment(1, 0, 6000),
        SpeakerSegment(2, 5000, 8000),
    )


def test_duplicate_segment_is_rejected():
    raw = {"Speaker": 0, "Start": 0.0, "End": 1.0}
    with pytest.raises(TranscriptionError, match="duplicate"):
        normalize_nemotron_segments([raw, raw], duration_ms=8000)


def test_worker_accepts_only_existing_job_wav_under_own_root(tmp_path: Path):
    job = tmp_path / "job-abc"
    job.mkdir()
    path = job / "normalized.wav"
    path.write_bytes(b"wav")
    assert validate_job_wav_path(path, tmp_path) == path
    outside = tmp_path.parent / "outside.wav"
    with pytest.raises(TranscriptionError):
        validate_job_wav_path(outside, tmp_path)


def test_worker_stdout_contains_only_json_when_runtime_prints(
    tmp_path: Path, monkeypatch
):
    import io
    import json
    import sys

    import funasr_transcriber.nemotron_worker as worker

    class NoisyRuntime:
        def __init__(self, model_dir):
            print("[ERROR] library import diagnostic")
            self.identity = {"device": "cuda:0"}

        def diarize(self, path, allowed_root):
            print("library inference diagnostic")
            return (SpeakerSegment(1, 0, 1000),), {"inference_seconds": 0.1}

    output = io.StringIO()
    errors = io.StringIO()
    monkeypatch.setattr(worker, "NemotronRuntime", NoisyRuntime)
    monkeypatch.setattr(
        sys, "stdin",
        io.StringIO(
            '{"id":"one","action":"status"}\n'
            '{"id":"two","action":"diarize","audio_path":"/tmp/input.wav"}\n'
            '{"id":"three","action":"close"}\n'
        ),
    )
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)

    assert worker.serve(tmp_path, tmp_path) == 0
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [message["id"] for message in messages] == ["one", "two", "three"]
    assert "[ERROR] library import diagnostic" in errors.getvalue()
    assert "library inference diagnostic" in errors.getvalue()
