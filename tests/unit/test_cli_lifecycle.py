from __future__ import annotations

from pathlib import Path

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.overlap import (
    OverlapPreparation,
    PreparedTrack,
    PreparedWindow,
)
from funasr_transcriber.schemas import (
    AsrResult,
    RecognizedSentence,
    SpeakerSegment,
    TimedToken,
)
from funasr_transcriber.cli import execute_model_phases


def _result(text: str = "原始。") -> AsrResult:
    return AsrResult(
        text,
        (TimedToken(text, 0, 500, 0.9),),
        sentences=(
            RecognizedSentence(
                text, 0, 500, "zh mandarin", 0.99, 0.9
            ),
        ),
        vad_segments_ms=((0, 500),),
        duration_ms=600,
    )


def _run(
    tmp_path: Path,
    with_overlap: bool = True,
    broken_track: bool = False,
    empty_tracks: bool = False,
):
    events: list[str] = []
    normalized = (tmp_path / "normalized.wav").resolve()
    normalized.write_bytes(b"audio")
    track_one = (tmp_path / "track-1.wav").resolve()
    track_two = (tmp_path / "track-2.wav").resolve()
    track_one.write_bytes(b"one")
    track_two.write_bytes(b"two")

    class AsrClient:
        last_response = {}

        def __enter__(self):
            events.append("asr:start")
            return self

        def __exit__(self, *args):
            events.append("asr:close")

        def transcribe(self, path: Path):
            path = Path(path)
            if path == normalized:
                events.append("asr:transcribe")
                return _result()
            events.append(f"asr:{path.stem}")
            if broken_track:
                raise RuntimeError("track failed")
            return _result("甲。" if path == track_one else "乙。")

    class Diarizer:
        def __enter__(self):
            events.append("speaker:start")
            return self

        def __exit__(self, *args):
            events.append("speaker:close")

        def diarize(self, path):
            events.append("speaker:diarize")
            return (
                SpeakerSegment(1, 0, 300),
                SpeakerSegment(2, 300, 600),
            )

    class Preparer:
        def prepare(self, path, segments):
            events.append("overlap:prepare")
            if not with_overlap:
                return OverlapPreparation((), ())
            return OverlapPreparation(
                (
                    PreparedWindow(
                        "w1",
                        0,
                        600,
                        () if empty_tracks else (
                            PreparedTrack(track_one, 1, 0.9, 0.2, 0.1),
                            PreparedTrack(track_two, 2, 0.9, 0.2, 0.1),
                        ),
                    ),
                ),
                (),
            )

    result = execute_model_phases(
        normalized,
        load_config(),
        overlap_enabled=True,
        asr_client_factory=lambda stage: AsrClient(),
        diarizer_factory=Diarizer,
        overlap_preparer_factory=lambda diarizer: Preparer(),
        gpu_check=lambda stage: events.append(f"gpu-check:{stage}"),
        result_consumer=lambda result: events.append("export"),
    )
    return result, events


def test_gpu_models_follow_required_serial_lifecycle(tmp_path: Path):
    result, events = _run(tmp_path)

    assert events == [
        "gpu-check:primary-asr",
        "asr:start",
        "asr:transcribe",
        "asr:close",
        "gpu-check:speaker",
        "speaker:start",
        "speaker:diarize",
        "overlap:prepare",
        "speaker:close",
        "gpu-check:overlap-asr",
        "asr:start",
        "asr:track-1",
        "asr:track-2",
        "asr:close",
        "export",
    ]
    assert result.transcript.full_text == "甲。乙。"


def test_no_overlap_starts_firered_only_once(tmp_path: Path):
    result, events = _run(tmp_path, with_overlap=False)

    assert events.count("asr:start") == 1
    assert "gpu-check:overlap-asr" not in events
    assert result.transcript.full_text == "原始。"


def test_filtered_overlap_windows_do_not_start_second_firered(tmp_path: Path):
    result, events = _run(tmp_path, empty_tracks=True)

    assert events.count("asr:start") == 1
    assert "gpu-check:overlap-asr" not in events
    assert result.transcript.full_text == "原始。"
    assert any(
        "overlap window w1 kept original: track_count_not_two" in warning
        for warning in result.transcript.warnings
    )


def test_track_asr_failure_degrades_to_base_tokens(tmp_path: Path):
    result, _ = _run(tmp_path, broken_track=True)

    assert result.transcript.full_text == "原始。"
    assert any("track_count_not_two" in item for item in result.transcript.warnings)


def test_primary_asr_failure_has_no_fallback(tmp_path: Path):
    normalized = (tmp_path / "normalized.wav").resolve()
    normalized.write_bytes(b"audio")
    events = []

    class BrokenClient:
        last_response = {}

        def __enter__(self):
            events.append("asr:start")
            return self

        def __exit__(self, *args):
            events.append("asr:close")

        def transcribe(self, path):
            raise TranscriptionError("FireRed failed")

    with pytest.raises(TranscriptionError, match="FireRed failed"):
        execute_model_phases(
            normalized,
            load_config(),
            overlap_enabled=False,
            asr_client_factory=lambda stage: BrokenClient(),
            diarizer_factory=lambda: None,
            overlap_preparer_factory=lambda diarizer: None,
            gpu_check=lambda stage: events.append(f"gpu-check:{stage}"),
        )

    assert events == ["gpu-check:primary-asr", "asr:start", "asr:close"]


def test_production_source_has_no_legacy_asr_route():
    source = Path("src/funasr_transcriber")
    text = "\n".join(
        path.read_text("utf-8")
        for path in source.glob("*.py")
    )

    assert "from funasr import AutoModel" not in text
    assert "FunAsrEngine" not in text
    assert "build_asr_model" not in text
