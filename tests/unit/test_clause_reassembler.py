"""Unit tests for Nemotron-3 clause-level interruption reassembly."""

from __future__ import annotations

import pytest

from funasr_transcriber.clause_reassembler import (
    reassemble_transcript_with_nemotron,
    split_text_into_clauses,
)
from funasr_transcriber.schemas import AlignmentConfidence, SpeakerSegment, Transcript, Turn


def test_split_text_into_clauses():
    text = "对这是周年庆，我是想跟你讲，因为你不是，如果你今年考虑今天做？"
    clauses = split_text_into_clauses(text)
    assert clauses == [
        "对这是周年庆，",
        "我是想跟你讲，",
        "因为你不是，",
        "如果你今年考虑今天做？",
    ]


def test_clause_reassembler_preserves_single_clause_turns():
    turn = Turn(
        index=1,
        speaker_id=1,
        start_ms=0,
        end_ms=2000,
        text="好的谢谢。",
        source="primary_asr",
        confidence=AlignmentConfidence.HIGH,
        overlap=False,
    )
    transcript = Transcript(
        language="zh",
        speaker_count=1,
        turns=(turn,),
        warnings=(),
    )
    nemotron_segs = (
        SpeakerSegment(speaker_id=1, start_ms=0, end_ms=2000),
    )

    result = reassemble_transcript_with_nemotron(transcript, nemotron_segs)
    assert result.transcript.full_text == transcript.full_text
    assert len(result.transcript.turns) == 1
    assert len(result.interruption_events) == 0


def test_clause_reassembler_splits_interruption():
    # Turn 1: Speaker 2 speaks a long sentence, but middle clause was spoken by Speaker 1 (interruption)
    # Total length: 60 chars.
    # Clause 1: "价格是很透明的这里，" (10 chars, 0-3000ms)
    # Clause 2: "呃如果你们周年庆不是搞活动吗？" (15 chars, 3000-7500ms) -> Interrupted by Speaker 1!
    # Clause 3: "对这是周年庆我跟你讲。" (11 chars, 7500-11000ms)
    text = "价格是很透明的这里，呃如果你们周年庆不是搞活动吗？对这是周年庆我跟你讲。"
    turn = Turn(
        index=1,
        speaker_id=2,
        start_ms=0,
        end_ms=11000,
        text=text,
        source="primary_asr",
        confidence=AlignmentConfidence.HIGH,
        overlap=False,
    )
    transcript = Transcript(
        language="zh",
        speaker_count=2,
        turns=(turn,),
        warnings=(),
    )

    # Nemotron segments:
    # Speaker 2 active at 0-3000ms and 7500-11000ms
    # Speaker 1 active at 3000-7500ms (interruption!)
    nemotron_segs = (
        SpeakerSegment(speaker_id=2, start_ms=0, end_ms=3000),
        SpeakerSegment(speaker_id=1, start_ms=3000, end_ms=7500),
        SpeakerSegment(speaker_id=2, start_ms=7500, end_ms=11000),
    )

    result = reassemble_transcript_with_nemotron(
        transcript,
        nemotron_segs,
        min_interruption_ratio=0.50,
        min_clause_ms=500,
    )

    # Text integrity must be 100% identical!
    assert result.transcript.full_text == transcript.full_text

    # Should be split into 3 turns: Spk 2 -> Spk 1 -> Spk 2
    assert len(result.transcript.turns) == 3
    t1, t2, t3 = result.transcript.turns

    assert t1.speaker_id == 2
    assert "价格是很透明的这里，" in t1.text

    assert t2.speaker_id == 1
    assert "呃如果你们周年庆不是搞活动吗？" in t2.text
    assert t2.overlap is True
    assert t2.source == "nemotron_interruption"

    assert t3.speaker_id == 2
    assert "对这是周年庆我跟你讲。" in t3.text

    assert len(result.interruption_events) == 1
    event = result.interruption_events[0]
    assert event.original_speaker == 2
    assert event.assigned_speaker == 1
    assert "搞活动吗" in event.clause_text


def test_clause_reassembler_handles_inverted_speaker_labels():
    # If Nemotron labels are inverted (Channel 1 = Spk 2, Channel 2 = Spk 1)
    text = "我是第一位发言人，中间这句由第二位发言人插话，最后由第一位发言人总结。"
    turn = Turn(
        index=1,
        speaker_id=1,
        start_ms=0,
        end_ms=12000,
        text=text,
        source="primary_asr",
        confidence=AlignmentConfidence.HIGH,
        overlap=False,
    )
    transcript = Transcript(
        language="zh",
        speaker_count=2,
        turns=(turn,),
        warnings=(),
    )

    # In Nemotron, channel 2 is Spk 1, channel 1 is Spk 2
    nemotron_segs = (
        SpeakerSegment(speaker_id=2, start_ms=0, end_ms=4000),
        SpeakerSegment(speaker_id=1, start_ms=4000, end_ms=8000),
        SpeakerSegment(speaker_id=2, start_ms=8000, end_ms=12000),
    )

    result = reassemble_transcript_with_nemotron(transcript, nemotron_segs)

    # Bipartite mapping should align Nemotron 2 -> Transcript 1 and Nemotron 1 -> Transcript 2
    assert result.speaker_mapping.get(2) == 1
    assert len(result.transcript.turns) == 3
    assert result.transcript.turns[1].speaker_id == 2
    assert result.transcript.turns[1].source == "nemotron_interruption"


def test_clause_reassembler_projects_excess_nemotron_channels_without_inflating_speakers():
    """Verify Method A: excess channels are projected onto 3D-Speaker authority space without introducing new speakers."""
    # Transcript has strictly 2 speakers: Spk 1 (Turn 1) and Spk 2 (Turn 2)
    turn1 = Turn(
        index=1,
        speaker_id=1,
        start_ms=0,
        end_ms=10000,
        text="我是主讲人，这里有一段说明，请问你有什么看法？",
        source="primary_asr",
        confidence=AlignmentConfidence.HIGH,
        overlap=False,
    )
    turn2 = Turn(
        index=2,
        speaker_id=2,
        start_ms=10000,
        end_ms=20000,
        text="我是第二位说话人，我完全同意你的看法，没有任何异议。",
        source="primary_asr",
        confidence=AlignmentConfidence.HIGH,
        overlap=False,
    )
    transcript = Transcript(
        language="zh",
        speaker_count=2,
        turns=(turn1, turn2),
        warnings=(),
    )

    # Nemotron activates 4 channels:
    # Channel 1: 0-5000ms (overlaps Spk 1)
    # Channel 2: 10000-15000ms (overlaps Spk 2)
    # Channel 3: 5000-10000ms (overlaps Spk 1 -> fragmented consultant channel!)
    # Channel 4: 15000-15300ms (overlaps Spk 2 for 300ms -> low-overlap noise < 1000ms)
    nemotron_segs = (
        SpeakerSegment(speaker_id=1, start_ms=0, end_ms=5000),
        SpeakerSegment(speaker_id=2, start_ms=10000, end_ms=15000),
        SpeakerSegment(speaker_id=3, start_ms=5000, end_ms=10000),
        SpeakerSegment(speaker_id=4, start_ms=15000, end_ms=15300),
    )

    result = reassemble_transcript_with_nemotron(
        transcript,
        nemotron_segs,
        min_overlap_ms=1000,
        min_purity=0.50,
    )

    # Channel 1 -> 1, Channel 2 -> 2, Channel 3 -> 1 (projected!), Channel 4 -> pruned
    assert result.speaker_mapping == {1: 1, 2: 2, 3: 1}

    # Authoritative speaker count must remain strictly 2 (no dummy Spk 3 or 4!)
    assert result.transcript.speaker_count == 2
    turn_speakers = {t.speaker_id for t in result.transcript.turns}
    assert turn_speakers.issubset({1, 2})
    assert 3 not in turn_speakers
    assert 4 not in turn_speakers
