from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from funasr_transcriber.schemas import SpeakerSegment
from funasr_transcriber.speaker_crops import build_crop_plan


def _source(path: Path, duration_ms: int):
    samples = np.arange(duration_ms * 16, dtype=np.int32).astype(np.int16)
    sf.write(path, samples, 16000, subtype="PCM_16")
    return samples


def test_crossing_segments_copy_same_original_overlap(tmp_path):
    source = tmp_path / "normalized.wav"
    _source(source, 8000)
    plan = build_crop_plan(
        source,
        (
            SpeakerSegment(1, 0, 6000),
            SpeakerSegment(2, 5000, 8000),
        ),
        tmp_path / "crops",
        refine_vad=None,
    )
    assert len(plan.crops) == 2
    first, _ = sf.read(plan.crops[0].path, dtype="int16")
    second, _ = sf.read(plan.crops[1].path, dtype="int16")
    assert np.array_equal(
        first[5000 * 16:6000 * 16], second[:1000 * 16]
    )
    assert plan.overlap_duration_ms == 1000
    assert (plan.crops[0].speaker_id, plan.crops[1].speaker_id) == (1, 2)


def test_nested_overlap_counts_union_not_pairwise_sum(tmp_path):
    source = tmp_path / "normalized.wav"
    _source(source, 4000)
    plan = build_crop_plan(
        source,
        (
            SpeakerSegment(1, 0, 4000),
            SpeakerSegment(2, 1000, 3000),
            SpeakerSegment(2, 1500, 2500),
        ),
        tmp_path / "crops",
        refine_vad=None,
    )
    assert len(plan.crops) == 3
    assert plan.overlap_duration_ms == 2000


@pytest.mark.parametrize(
    ("duration_ms", "expected_crops"),
    ((10, 0), (100, 1), (190, 1)),
)
def test_short_boundary(tmp_path, duration_ms, expected_crops):
    source = tmp_path / "normalized.wav"
    _source(source, 190)
    plan = build_crop_plan(
        source,
        (SpeakerSegment(1, 0, duration_ms),),
        tmp_path / "crops",
        refine_vad=None,
    )
    assert len(plan.crops) == expected_crops
    assert plan.short_count == (1 if duration_ms < 100 else 0)


def test_long_segment_is_bounded_and_bad_vad_falls_back(tmp_path):
    source = tmp_path / "normalized.wav"
    _source(source, 31_000)
    plan = build_crop_plan(
        source,
        (SpeakerSegment(1, 0, 31_000),),
        tmp_path / "crops",
        refine_vad=lambda path, ranges: [[0, 31_000]],
    )
    assert plan.crops[0].pieces_ms == (
        (0, 10_000),
        (10_000, 20_000),
        (20_000, 30_000),
        (30_000, 31_000),
    )
    assert plan.subpiece_count == 4
    assert plan.warnings


def test_create_only_refuses_existing_target(tmp_path):
    source = tmp_path / "normalized.wav"
    _source(source, 1000)
    target = tmp_path / "crops"
    target.mkdir()
    existing = target / "crop-0000-speaker-1.wav"
    existing.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        build_crop_plan(
            source,
            (SpeakerSegment(1, 0, 1000),),
            target,
            refine_vad=None,
        )

    assert existing.read_bytes() == b"keep"

def test_valid_vad_pieces_and_checksum(tmp_path):
    import hashlib

    source = tmp_path / "normalized.wav"
    _source(source, 12_000)
    plan = build_crop_plan(
        source,
        (SpeakerSegment(1, 0, 12_000),),
        tmp_path / "crops",
        refine_vad=lambda path, ranges: [[0, 9000], [9500, 12_000]],
    )

    crop = plan.crops[0]
    assert crop.pieces_ms == ((0, 9000), (9500, 12_000))
    assert plan.subpiece_count == 2
    assert crop.sha256 == hashlib.sha256(crop.path.read_bytes()).hexdigest()


def test_adjacent_same_speaker_intervals_remain_distinct(tmp_path):
    source = tmp_path / "normalized.wav"
    _source(source, 1000)
    plan = build_crop_plan(
        source,
        (
            SpeakerSegment(1, 0, 500),
            SpeakerSegment(1, 500, 1000),
        ),
        tmp_path / "crops",
        refine_vad=None,
    )

def test_waveform_splicing_replaces_overlap_region_with_track(tmp_path):
    from funasr_transcriber.overlap import OverlapPreparation, PreparedTrack, PreparedWindow

    source = tmp_path / "normalized.wav"
    # Source has 100 everywhere
    samples = np.full(8000 * 16, 100, dtype=np.int16)
    sf.write(source, samples, 16000, subtype="PCM_16")

    track1 = tmp_path / "t1.wav"
    track2 = tmp_path / "t2.wav"
    # Overlap window is 2000ms to 4000ms (2000ms long = 32000 samples)
    t1_samples = np.full(2000 * 16, 500, dtype=np.int16)
    t2_samples = np.full(2000 * 16, 800, dtype=np.int16)
    sf.write(track1, t1_samples, 16000, subtype="PCM_16")
    sf.write(track2, t2_samples, 16000, subtype="PCM_16")

    preparation = OverlapPreparation(
        windows=(
            PreparedWindow(
                id="w1",
                start_ms=2000,
                end_ms=4000,
                tracks=(
                    PreparedTrack(track1, speaker_id=1, score=0.9, margin=0.3, rms=0.1),
                    PreparedTrack(track2, speaker_id=2, score=0.9, margin=0.3, rms=0.1),
                ),
            ),
        ),
        warnings=(),
    )

    plan = build_crop_plan(
        source,
        (
            SpeakerSegment(1, 0, 6000),
            SpeakerSegment(2, 2000, 5000),
        ),
        tmp_path / "crops",
        refine_vad=None,
        overlap_preparation=preparation,
    )

    assert plan.spliced_overlap_count == 2
    c0, _ = sf.read(plan.crops[0].path, dtype="int16")
    # In crop 0 (Speaker 1, 0 to 6000ms):
    # Before overlap window (0 to 1900ms), values should be 100
    assert np.all(c0[: 1900 * 16] == 100)
    # Well inside the overlap window (2100 to 3900ms), values should be 500 (from track1)
    assert np.all(c0[2100 * 16 : 3900 * 16] == 500)
    # After overlap window (4100 to 6000ms), values should be 100
    assert np.all(c0[4100 * 16 :] == 100)

    # In crop 1 (Speaker 2, 2000 to 5000ms):
    # Segment starts right at overlap window start (2000ms).
    # Well inside overlap (crop offset 100ms to 1900ms = global 2100 to 3900ms), values should be 800
    c1, _ = sf.read(plan.crops[1].path, dtype="int16")
    assert np.all(c1[100 * 16 : 1900 * 16] == 800)
    # After overlap window (global 4100 to 5000ms = crop offset 2100 to 3000ms), values should be 100
    assert np.all(c1[2100 * 16 :] == 100)


def test_waveform_splicing_skips_when_score_or_margin_low(tmp_path):
    from funasr_transcriber.overlap import OverlapPreparation, PreparedTrack, PreparedWindow

    source = tmp_path / "normalized.wav"
    samples = np.full(4000 * 16, 100, dtype=np.int16)
    sf.write(source, samples, 16000, subtype="PCM_16")

    track1 = tmp_path / "t1.wav"
    sf.write(track1, np.full(1000 * 16, 500, dtype=np.int16), 16000, subtype="PCM_16")

    # Score is 0.2, below min_score 0.5
    preparation = OverlapPreparation(
        windows=(
            PreparedWindow(
                id="w1",
                start_ms=1000,
                end_ms=2000,
                tracks=(
                    PreparedTrack(track1, speaker_id=1, score=0.2, margin=0.3, rms=0.1),
                ),
            ),
        ),
        warnings=(),
    )

    plan = build_crop_plan(
        source,
        (SpeakerSegment(1, 0, 4000),),
        tmp_path / "crops",
        refine_vad=None,
        overlap_preparation=preparation,
        min_score=0.5,
    )

    assert plan.spliced_overlap_count == 0
    c0, _ = sf.read(plan.crops[0].path, dtype="int16")
    assert np.all(c0 == 100)
