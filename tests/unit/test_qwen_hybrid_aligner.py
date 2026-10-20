"""Unit tests for Qwen hybrid aligner with 3D-Speaker + Nemotron diarization."""

from __future__ import annotations

import pytest

from funasr_transcriber.qwen_client import (
    QwenSentence,
    QwenTranscriptionResult,
    QwenWordToken,
)
from funasr_transcriber.qwen_hybrid_aligner import (
    Clause,
    align_clauses_with_diarization,
    align_qwen_with_diarization,
    split_sentences_into_clauses,
)
from funasr_transcriber.schemas import AlignmentConfidence, SpeakerSegment


def test_split_sentences_into_clauses() -> None:
    sentences = [
        QwenSentence(
            sentence_id=1,
            speaker_id=0,
            start_ms=1000,
            end_ms=4000,
            text="你好，今天天气不错。对吧？",
            words=(
                QwenWordToken("你好", 1000, 1500, punctuation="，"),
                QwenWordToken("今天", 1550, 2000),
                QwenWordToken("天气", 2000, 2500),
                QwenWordToken("不错", 2500, 3000, punctuation="。"),
                QwenWordToken("对吧", 3200, 4000, punctuation="？"),
            ),
        )
    ]

    clauses = split_sentences_into_clauses(sentences)
    assert len(clauses) == 3
    assert clauses[0].text == "你好，"
    assert clauses[0].start_ms == 1000
    assert clauses[0].end_ms == 1500
    assert clauses[0].sentence_id == 1

    assert clauses[1].text == "今天天气不错。"
    assert clauses[1].start_ms == 1550
    assert clauses[1].end_ms == 3000

    assert clauses[2].text == "对吧？"
    assert clauses[2].start_ms == 3200
    assert clauses[2].end_ms == 4000


def test_align_clauses_with_diarization_turn_switch() -> None:
    # Speaker 1 speaks 1000-3000, Speaker 2 speaks 3100-5000
    diarized = (
        SpeakerSegment(1, 1000, 3000),
        SpeakerSegment(2, 3100, 5000),
    )

    clauses = [
        Clause(
            index=1,
            sentence_id=1,
            text="第一句由说话人一说。",
            start_ms=1100,
            end_ms=2900,
            words=(),
        ),
        Clause(
            index=2,
            sentence_id=2,
            text="第二句由说话人二说。",
            start_ms=3200,
            end_ms=4800,
            words=(),
        ),
    ]

    turns = align_clauses_with_diarization(clauses, diarized)
    assert len(turns) == 2
    assert turns[0].speaker_id == 1
    assert turns[0].text == "第一句由说话人一说。"
    assert turns[1].speaker_id == 2
    assert turns[1].text == "第二句由说话人二说。"


def test_align_clauses_with_diarization_turn_merge() -> None:
    # Same speaker speaking consecutive clauses within 1.5s gap
    diarized = (
        SpeakerSegment(1, 1000, 5000),
    )

    clauses = [
        Clause(
            index=1,
            sentence_id=1,
            text="这是前半句，",
            start_ms=1000,
            end_ms=2000,
            words=(),
        ),
        Clause(
            index=2,
            sentence_id=1,
            text="这是后半句。",
            start_ms=2200,
            end_ms=3500,
            words=(),
        ),
    ]

    turns = align_clauses_with_diarization(clauses, diarized, turn_merge_gap_ms=1500)
    assert len(turns) == 1
    assert turns[0].speaker_id == 1
    assert turns[0].text == "这是前半句，这是后半句。"
    assert turns[0].start_ms == 1000
    assert turns[0].end_ms == 3500


def test_align_qwen_with_diarization_end_to_end() -> None:
    s = QwenSentence(
        sentence_id=1,
        speaker_id=0,
        start_ms=1000,
        end_ms=3000,
        text="欢迎光临。",
        words=(
            QwenWordToken("欢迎", 1000, 1800),
            QwenWordToken("光临", 1850, 3000, punctuation="。"),
        ),
    )
    result = QwenTranscriptionResult(
        sentences=(s,),
        words=s.words,
        raw_response={},
    )
    diarized = (SpeakerSegment(1, 1000, 3000),)

    transcript = align_qwen_with_diarization(result, diarized)
    assert transcript.language == "zh"
    assert transcript.speaker_count == 1
    assert len(transcript.turns) == 1
    assert transcript.turns[0].speaker_id == 1
    assert transcript.turns[0].text == "欢迎光临。"
