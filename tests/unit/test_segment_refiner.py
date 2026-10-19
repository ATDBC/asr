from funasr_transcriber.schemas import SpeakerSegment
from funasr_transcriber.segment_refiner import (
    align_nemotron_to_base_speakers,
    refine_segments_with_nemotron,
)


def test_align_nemotron_to_base_speakers_matches_highest_overlap():
    base = (
        SpeakerSegment(1, 0, 5000),
        SpeakerSegment(2, 5000, 10000),
    )
    nem = (
        SpeakerSegment(1, 5000, 10000),
        SpeakerSegment(2, 0, 5000),
    )
    mapping, aligned = align_nemotron_to_base_speakers(base, nem)
    assert mapping == {1: 2, 2: 1}
    assert [s.speaker_id for s in aligned] == [1, 2]


def test_refine_segments_splits_when_other_speaker_has_independent_speech():
    # Base has Speaker 1 and Speaker 2
    base = (
        SpeakerSegment(1, 0, 10000),
        SpeakerSegment(2, 10000, 15000),
    )
    # Nemotron shows:
    # Spk 1: [0, 4500]
    # Spk 2: [4000, 6000] (overlaps with 1 between 4000-4500 (500ms), but independent 4500-6000 (1500ms))
    # Spk 1: [6200, 10000]
    # Spk 2: [10000, 15000]
    nem = (
        SpeakerSegment(1, 0, 4500),
        SpeakerSegment(2, 4000, 6000),
        SpeakerSegment(1, 6200, 10000),
        SpeakerSegment(2, 10000, 15000),
    )
    refined, stats = refine_segments_with_nemotron(base, nem)
    assert stats["splits"] == 1
    assert stats["inserted_count"] == 1
    # Check strictly monotonic non-overlapping cuts
    assert [s.speaker_id for s in refined] == [1, 2, 1, 2]
    assert refined[0].start_ms == 0
    assert refined[0].end_ms == 4000
    assert refined[1].speaker_id == 2
    assert refined[1].start_ms == 4000
    assert refined[1].end_ms == 6200
    assert refined[2].speaker_id == 1
    assert refined[2].start_ms == 6200
    assert refined[2].end_ms == 10000
    assert refined[3].speaker_id == 2
    assert refined[3].start_ms == 10000
    assert refined[3].end_ms == 15000

    # Ensure zero overlaps
    for i in range(len(refined) - 1):
        assert refined[i+1].start_ms >= refined[i].end_ms


def test_refine_segments_does_not_split_when_mostly_overlapping():
    base = (
        SpeakerSegment(1, 0, 10000),
        SpeakerSegment(2, 10000, 15000),
    )
    nem = (
        SpeakerSegment(1, 0, 10000),
        SpeakerSegment(2, 4000, 4500),
        SpeakerSegment(2, 10000, 15000),
    )
    refined, stats = refine_segments_with_nemotron(base, nem, max_overlap_ratio=0.60)
    assert stats["splits"] == 0
    assert len(refined) == 2
    assert refined[0] == base[0]


def test_refine_segments_does_not_split_brief_noise():
    base = (
        SpeakerSegment(1, 0, 10000),
        SpeakerSegment(2, 10000, 15000),
    )
    nem = (
        SpeakerSegment(1, 0, 10000),
        SpeakerSegment(2, 4000, 4150),
        SpeakerSegment(2, 10000, 15000),
    )
    refined, stats = refine_segments_with_nemotron(base, nem)
    assert stats["splits"] == 0
    assert len(refined) == 2
