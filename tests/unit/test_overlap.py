from pathlib import Path

import numpy as np

from funasr_transcriber.overlap import (
    SubprocessOverlapEnhancer,
    accept_separated_window,
)
from funasr_transcriber.schemas import (
    AlignmentConfidence,
    AsrResult,
    AttributedToken,
    SpeakerSegment,
    TimedToken,
)


def test_rejects_two_tracks_mapped_to_same_speaker():
    decision = accept_separated_window(
        tracks=[
            {
                "speaker_id": 1,
                "score": 0.82,
                "margin": 0.20,
                "text": "甲",
            },
            {
                "speaker_id": 1,
                "score": 0.79,
                "margin": 0.18,
                "text": "乙",
            },
        ],
        min_score=0.60,
        min_margin=0.08,
    )

    assert decision.accepted is False
    assert decision.reason == "tracks_not_distinct"


def test_accepts_two_distinct_confident_tracks():
    decision = accept_separated_window(
        tracks=[
            {
                "speaker_id": 1,
                "score": 0.82,
                "margin": 0.20,
                "text": "甲",
            },
            {
                "speaker_id": 2,
                "score": 0.79,
                "margin": 0.18,
                "text": "乙",
            },
        ],
        min_score=0.60,
        min_margin=0.08,
    )

    assert decision.accepted is True
    assert decision.reason == "accepted"


def test_rejects_empty_or_duplicate_text():
    empty = accept_separated_window(
        tracks=[
            {"speaker_id": 1, "score": 0.8, "margin": 0.2, "text": ""},
            {"speaker_id": 2, "score": 0.8, "margin": 0.2, "text": "乙"},
        ],
        min_score=0.6,
        min_margin=0.08,
    )
    duplicate = accept_separated_window(
        tracks=[
            {"speaker_id": 1, "score": 0.8, "margin": 0.2, "text": "相同"},
            {"speaker_id": 2, "score": 0.8, "margin": 0.2, "text": "相同"},
        ],
        min_score=0.6,
        min_margin=0.08,
    )

    assert empty.reason == "empty_track_text"
    assert duplicate.reason == "duplicate_track_text"


def test_rejects_tracks_without_timed_tokens():
    decision = accept_separated_window(
        tracks=[
            {
                "speaker_id": 1,
                "score": 0.8,
                "margin": 0.2,
                "text": "first",
                "timed_token_count": 0,
            },
            {
                "speaker_id": 2,
                "score": 0.8,
                "margin": 0.2,
                "text": "second",
                "timed_token_count": 1,
            },
        ],
        min_score=0.6,
        min_margin=0.08,
    )

    assert decision.accepted is False
    assert decision.reason == "missing_timed_tokens"


def test_rejects_track_whose_timed_tokens_do_not_cover_full_text():
    decision = accept_separated_window(
        tracks=[
            {
                "speaker_id": 1,
                "score": 0.8,
                "margin": 0.2,
                "text": "timed and untimed",
                "timed_token_count": 1,
                "timed_text_complete": False,
            },
            {
                "speaker_id": 2,
                "score": 0.8,
                "margin": 0.2,
                "text": "complete",
                "timed_token_count": 1,
                "timed_text_complete": True,
            },
        ],
        min_score=0.6,
        min_margin=0.08,
    )

    assert decision.accepted is False
    assert decision.reason == "incomplete_timed_text"


def test_enhancer_replaces_window_only_after_two_tracks_pass(tmp_path: Path):
    first_track = tmp_path / "track-1.wav"
    second_track = tmp_path / "track-2.wav"
    first_track.write_bytes(b"one")
    second_track.write_bytes(b"two")

    class Worker:
        def call(self, action, payload):
            if action == "detect":
                return {
                    "ok": True,
                    "windows": [
                        {"id": "w1", "start_ms": 1000, "end_ms": 2000}
                    ],
                }
            return {
                "ok": True,
                "windows": [
                    {
                        "id": "w1",
                        "start_ms": 1000,
                        "end_ms": 2000,
                        "tracks": [
                            {"path": str(first_track), "rms": 0.1},
                            {"path": str(second_track), "rms": 0.1},
                        ],
                    }
                ],
            }

    class Asr:
        def transcribe(self, path):
            if Path(path) == first_track:
                return AsrResult("甲。", (TimedToken("甲。", 0, 800),))
            return AsrResult("乙。", (TimedToken("乙。", 0, 700),))

    class Diarizer:
        def speaker_centroids(self, path, segments):
            return {
                1: np.array([1.0, 0.0], dtype="float32"),
                2: np.array([0.0, 1.0], dtype="float32"),
            }

        def embedding_for_audio(self, path):
            if Path(path) == first_track:
                return np.array([1.0, 0.0], dtype="float32")
            return np.array([0.0, 1.0], dtype="float32")

    base = (
        AttributedToken(
            "原始。",
            1000,
            2000,
            1,
            AlignmentConfidence.HIGH,
        ),
    )
    segments = (
        SpeakerSegment(1, 0, 1500),
        SpeakerSegment(2, 1500, 2500),
    )
    enhancer = SubprocessOverlapEnhancer(
        worker=Worker(),
        asr_engine=Asr(),
        diarizer=Diarizer(),
        min_score=0.60,
        min_margin=0.08,
        min_rms=0.001,
    )

    result = enhancer.enhance(tmp_path / "source.wav", base, segments)

    assert result.modified is True
    assert [item.speaker_id for item in result.tokens] == [1, 2]
    assert all(item.overlap for item in result.tokens)
    assert "".join(item.text for item in result.tokens) == "甲。乙。"
