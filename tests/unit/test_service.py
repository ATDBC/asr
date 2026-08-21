from __future__ import annotations

from pathlib import Path
import threading

import pytest

from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.schemas import (
    AsrResult,
    RecognizedSentence,
    SpeakerSegment,
    TimedToken,
)
from funasr_transcriber.service import HotModelHost, run_normal_branches


def _result() -> AsrResult:
    return AsrResult(
        text="原始。",
        tokens=(TimedToken("原始。", 0, 500, 0.9),),
        sentences=(
            RecognizedSentence("原始。", 0, 500, "zh mandarin", 0.99, 0.9),
        ),
        vad_segments_ms=((0, 500),),
        duration_ms=600,
    )


class FakeAsr:
    def __init__(self, events: list[str]):
        self.events = events

    def start(self):
        self.events.append("asr:start")
        return self

    def status(self):
        return {"modules": {}}

    def close(self):
        self.events.append("asr:close")

    def transcribe(self, path: Path):
        self.events.append("asr:transcribe")
        return _result()


class FakeDiarizer:
    def __init__(self, events: list[str]):
        self.events = events

    def start(self):
        self.events.append("speaker:start")
        return self

    def close(self):
        self.events.append("speaker:close")

    def diarize(self, path: Path):
        self.events.append("speaker:diarize")
        return (SpeakerSegment(1, 0, 600),)

    def resource_inventory(self):
        return {"embedding": {}}


def test_hot_model_host_loads_once_and_closes_only_owned_models():
    events: list[str] = []
    host = HotModelHost(
        asr_factory=lambda: FakeAsr(events),
        diarizer_factory=lambda: FakeDiarizer(events),
        gpu_check=lambda: events.append("gpu-check"),
    )

    host.start()
    host.start()
    host.close()

    assert events[0] == "gpu-check"
    assert events.count("asr:start") == 1
    assert events.count("speaker:start") == 1
    assert events.count("asr:close") == 1
    assert events.count("speaker:close") == 1


class CoordinatedAsr:
    def __init__(self, gate: threading.Barrier):
        self.gate = gate

    def transcribe(self, path: Path):
        self.gate.wait()
        return _result()


class CoordinatedDiarizer:
    def __init__(self, gate: threading.Barrier):
        self.gate = gate

    def diarize(self, path: Path):
        self.gate.wait()
        return (SpeakerSegment(1, 0, 600),)


def test_normal_job_starts_asr_and_diarization_before_waiting_for_results(
    tmp_path: Path,
):
    gate = threading.Barrier(2, timeout=1)

    result = run_normal_branches(
        tmp_path / "normalized.wav",
        CoordinatedAsr(gate),
        CoordinatedDiarizer(gate),
    )

    assert result.asr_result.text == "原始。"
    assert result.speaker_segments == (SpeakerSegment(1, 0, 600),)
    assert gate.broken is False


class FailingAsr:
    def transcribe(self, path: Path):
        raise TranscriptionError("asr failed")


class BlockingDiarizer:
    def __init__(self):
        self.finished = False

    def diarize(self, path: Path):
        self.finished = True
        return (SpeakerSegment(1, 0, 600),)


def test_normal_branch_waits_for_the_other_owner_before_raising(tmp_path: Path):
    diarizer = BlockingDiarizer()

    with pytest.raises(TranscriptionError, match="asr failed"):
        run_normal_branches(tmp_path / "normalized.wav", FailingAsr(), diarizer)

    assert diarizer.finished is True
