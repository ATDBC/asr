from funasr_transcriber.schemas import AlignmentConfidence, AttributedToken
from funasr_transcriber.turns import build_turns


def token(
    text,
    start_ms,
    end_ms,
    speaker_id=1,
    source="primary_asr",
    overlap=False,
    confidence=AlignmentConfidence.HIGH,
):
    return AttributedToken(
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        speaker_id=speaker_id,
        source=source,
        overlap=overlap,
        confidence=confidence,
    )


def test_same_speaker_merges_at_exactly_one_second():
    turns = build_turns(
        (token("第一句。", 0, 1000), token("第二句。", 2000, 3000)),
        merge_gap_ms=1000,
    )

    assert len(turns) == 1
    assert turns[0].text == "第一句。第二句。"
    assert turns[0].start_ms == 0
    assert turns[0].end_ms == 3000


def test_same_speaker_does_not_merge_after_one_second():
    turns = build_turns(
        (token("第一句。", 0, 1000), token("第二句。", 2001, 3000)),
        merge_gap_ms=1000,
    )

    assert len(turns) == 2


def test_different_speakers_do_not_merge():
    turns = build_turns(
        (token("第一句。", 0, 1000), token("第二句。", 1100, 2000, speaker_id=2)),
        merge_gap_ms=1000,
    )

    assert [turn.speaker_id for turn in turns] == [1, 2]


def test_overlap_tokens_remain_independent():
    turns = build_turns(
        (
            token("第一路。", 0, 1000, overlap=True),
            token("第二路。", 100, 900, overlap=True),
        ),
        merge_gap_ms=1000,
    )

    assert len(turns) == 2


def test_simultaneous_turns_sort_by_speaker_id():
    turns = build_turns(
        (
            token("第一路。", 100, 1000, speaker_id=1, overlap=True),
            token("第二路。", 100, 900, speaker_id=2, overlap=True),
        ),
        merge_gap_ms=1000,
    )

    assert [turn.speaker_id for turn in turns] == [1, 2]


def test_preserve_input_ties_keeps_different_speaker_crop_order():
    turns = build_turns(
        (
            token("第二路。", 100, 900, speaker_id=2, overlap=True),
            token("第一路。", 100, 1000, speaker_id=1, overlap=True),
        ),
        merge_gap_ms=1000,
        preserve_input_ties=True,
    )

    assert [turn.speaker_id for turn in turns] == [2, 1]
    assert [turn.text for turn in turns] == ["第二路。", "第一路。"]


def test_same_speaker_merges_when_one_or_both_tokens_have_overlap():
    turns = build_turns(
        (
            token("第一句。", 0, 1000, speaker_id=1, overlap=False),
            token("第二句。", 1000, 2000, speaker_id=1, overlap=True),
        ),
        merge_gap_ms=1000,
    )

    assert len(turns) == 1
    assert turns[0].text == "第一句。第二句。"
    assert turns[0].start_ms == 0
    assert turns[0].end_ms == 2000
    assert turns[0].overlap is True
