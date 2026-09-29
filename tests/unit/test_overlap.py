from pathlib import Path

import numpy as np

from funasr_transcriber.overlap import (
    SeparatedOverlap,
    SubprocessOverlapPreparer,
    accept_separated_window,
    apply_prepared_overlap,
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


def test_overlap_preparation_does_not_run_asr(tmp_path: Path):
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

    segments = (
        SpeakerSegment(1, 0, 1500),
        SpeakerSegment(2, 1500, 2500),
    )
    preparer = SubprocessOverlapPreparer(
        worker=Worker(),
        diarizer=Diarizer(),
        min_score=0.60,
        min_margin=0.08,
        min_rms=0.001,
    )

    preparation = preparer.prepare(tmp_path / "source.wav", segments)

    assert len(preparation.windows) == 1
    assert [track.speaker_id for track in preparation.windows[0].tracks] == [1, 2]
    assert preparation.warnings == ()


def test_detect_and_separate_can_finish_before_temporary_speaker_matching(
    tmp_path: Path,
):
    track = tmp_path / "track.wav"
    track.write_bytes(b"track")
    calls = []

    class Worker:
        def call(self, action, payload):
            calls.append(action)
            if action == "detect":
                return {
                    "ok": True,
                    "windows": [{"id": "w1", "start_ms": 0, "end_ms": 100}],
                }
            return {
                "ok": True,
                "windows": [
                    {
                        "id": "w1",
                        "start_ms": 0,
                        "end_ms": 100,
                        "tracks": [{"path": str(track), "rms": 0.1}],
                    }
                ],
            }

    class Diarizer:
        def embedding_for_audio(self, path):
            return np.array([1.0, 0.0], dtype="float32")

    preparer = SubprocessOverlapPreparer(
        worker=Worker(),
        diarizer=None,
        min_score=0.6,
        min_margin=0.08,
        min_rms=0.001,
    )

    separated = preparer.detect_and_separate(tmp_path / "source.wav")
    preparation = preparer.match_tracks(
        separated,
        {1: np.array([1.0, 0.0], dtype="float32")},
        Diarizer(),
    )

    assert isinstance(separated, SeparatedOverlap)
    assert calls == ["detect", "separate"]
    assert preparation.windows[0].tracks[0].speaker_id == 1


def test_detect_stage_forwards_overlap_worker_lifetime_callbacks(tmp_path: Path):
    started = []
    finished = []

    class Worker:
        def call(self, action, payload, on_started=None, on_finished=None):
            if on_started is not None:
                on_started(123)
            if on_finished is not None:
                on_finished(123)
            return {"ok": True, "windows": []}

    preparer = SubprocessOverlapPreparer(
        worker=Worker(),
        diarizer=None,
        min_score=0.6,
        min_margin=0.08,
        min_rms=0.001,
        on_worker_started=started.append,
        on_worker_finished=finished.append,
    )

    preparer.detect_and_separate(tmp_path / "source.wav")

    assert started == [123]
    assert finished == [123]


def test_apply_replaces_window_only_after_two_tracks_pass(tmp_path: Path):
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

    class Asr:
        def __init__(self):
            self.calls = []

        def transcribe(self, path):
            self.calls.append(Path(path))
            if Path(path) == first_track:
                return AsrResult("甲。", (TimedToken("甲。", 0, 800),))
            return AsrResult("乙。", (TimedToken("乙。", 0, 700),))

    segments = (
        SpeakerSegment(1, 0, 1500),
        SpeakerSegment(2, 1500, 2500),
    )
    preparation = SubprocessOverlapPreparer(
        worker=Worker(),
        diarizer=Diarizer(),
        min_score=0.60,
        min_margin=0.08,
        min_rms=0.001,
    ).prepare(tmp_path / "source.wav", segments)
    base = (
        AttributedToken(
            "原始。",
            1000,
            2000,
            1,
            AlignmentConfidence.HIGH,
        ),
    )
    asr = Asr()

    result = apply_prepared_overlap(
        preparation,
        base,
        asr,
        min_score=0.60,
        min_margin=0.08,
    )

    assert result.modified is True
    assert asr.calls == [first_track, second_track]
    assert [item.speaker_id for item in result.tokens] == [1, 2]
    assert all(item.overlap for item in result.tokens)
    assert "".join(item.text for item in result.tokens) == "甲。乙。"


def test_apply_degrades_only_failed_overlap_window(tmp_path: Path):
    from funasr_transcriber.overlap import (
        OverlapPreparation,
        PreparedTrack,
        PreparedWindow,
    )

    base = (
        AttributedToken(
            "原始。",
            1000,
            2000,
            1,
            AlignmentConfidence.HIGH,
        ),
    )
    preparation = OverlapPreparation(
        windows=(
            PreparedWindow(
                "w1",
                1000,
                2000,
                (
                    PreparedTrack(tmp_path / "one.wav", 1, 0.9, 0.2, 0.1),
                    PreparedTrack(tmp_path / "two.wav", 2, 0.9, 0.2, 0.1),
                ),
            ),
        ),
        warnings=(),
    )

    class BrokenAsr:
        def transcribe(self, path):
            raise RuntimeError("track failed")

    result = apply_prepared_overlap(
        preparation,
        base,
        BrokenAsr(),
        min_score=0.6,
        min_margin=0.08,
    )

    assert result.tokens == base
    assert result.modified is False
    assert "track_count_not_two" in result.warnings[0]


def test_extract_overlap_windows_from_segments():
    from funasr_transcriber.overlap import extract_overlap_windows_from_segments

    # Single speaker, no overlap
    assert extract_overlap_windows_from_segments(
        (SpeakerSegment(1, 0, 1000), SpeakerSegment(1, 1500, 2500))
    ) == ()

    # Two non-overlapping speakers
    assert extract_overlap_windows_from_segments(
        (SpeakerSegment(1, 0, 1000), SpeakerSegment(2, 1200, 2000))
    ) == ()

    # Two overlapping speakers
    windows = extract_overlap_windows_from_segments(
        (
            SpeakerSegment(1, 0, 3000),
            SpeakerSegment(2, 2000, 5000),
        )
    )
    assert len(windows) == 1
    assert windows[0]["start_ms"] == 2000
    assert windows[0]["end_ms"] == 3000

    # Short overlap under min_duration_ms ignored
    windows_short = extract_overlap_windows_from_segments(
        (
            SpeakerSegment(1, 0, 2100),
            SpeakerSegment(2, 2000, 5000),
        ),
        min_duration_ms=200,
    )
    assert windows_short == ()

    # Merging close overlaps
    windows_merged = extract_overlap_windows_from_segments(
        (
            SpeakerSegment(1, 0, 3000),
            SpeakerSegment(2, 2000, 3000),
            SpeakerSegment(1, 3100, 5000),
            SpeakerSegment(2, 3100, 5000),
        ),
        merge_gap_ms=200,
    )
    assert len(windows_merged) == 1
    assert windows_merged[0]["start_ms"] == 2000
    assert windows_merged[0]["end_ms"] == 5000


def test_match_tracks_uses_batched_embeddings_for_audios(tmp_path: Path):
    track1 = tmp_path / "w1-s1.wav"
    track2 = tmp_path / "w1-s2.wav"
    track1.write_bytes(b"dummy1")
    track2.write_bytes(b"dummy2")

    batched_calls = []

    class BatchedDiarizer:
        def speaker_centroids(self, path, segments):
            return {
                1: np.array([1.0, 0.0], dtype="float32"),
                2: np.array([0.0, 1.0], dtype="float32"),
            }

        def embeddings_for_audios(self, paths):
            batched_calls.append(list(paths))
            return [
                np.array([1.0, 0.0], dtype="float32"),
                np.array([0.0, 1.0], dtype="float32"),
            ]

        def embedding_for_audio(self, path):
            raise AssertionError("Should have used embeddings_for_audios")

    preparer = SubprocessOverlapPreparer(
        worker=None,
        diarizer=BatchedDiarizer(),
        min_score=0.60,
        min_margin=0.08,
        min_rms=0.001,
    )
    separated = SeparatedOverlap(
        windows=(
            {
                "id": "w1",
                "start_ms": 1000,
                "end_ms": 2000,
                "tracks": [
                    {"index": 1, "path": str(track1), "rms": 0.05, "duration_ms": 1000},
                    {"index": 2, "path": str(track2), "rms": 0.05, "duration_ms": 1000},
                ],
            },
        ),
        warnings=(),
    )
    prep = preparer.match_tracks(separated, {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}, BatchedDiarizer())
    assert len(batched_calls) == 1
    assert batched_calls[0] == [track1, track2]
    assert len(prep.windows) == 1
    assert [t.speaker_id for t in prep.windows[0].tracks] == [1, 2]
