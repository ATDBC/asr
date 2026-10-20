"""Hybrid aligner uniting Qwen-Audio ASR with authoritative 3D-Speaker + Nemotron diarization.

Uses word-level timestamps and punctuation boundaries to partition text into
clauses, projects them onto authoritative speaker segments, and reassembles them
into natural, non-fragmented conversational turns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from .qwen_client import QwenSentence, QwenTranscriptionResult, QwenWordToken
from .schemas import AlignmentConfidence, SpeakerSegment, Transcript, Turn

LOGGER = logging.getLogger(__name__)

PUNCTUATIONS = {
    ",", "。", "，", "？", "！", "、", "；", "：", "…",
    "?", "!", ";", ":",
}


@dataclass(frozen=True, slots=True)
class Clause:
    """A minimal semantic clause delimited by punctuation or natural pause."""

    index: int
    sentence_id: int
    text: str
    start_ms: int
    end_ms: int
    words: tuple[QwenWordToken, ...]
    orig_speaker_id: int = 0

    @property
    def duration_ms(self) -> int:
        return max(1, self.end_ms - self.start_ms)


def split_sentences_into_clauses(
    sentences: Sequence[QwenSentence],
    *,
    max_silence_gap_ms: int = 300,
) -> list[Clause]:
    """Segment Qwen sentences into punctuation-delimited clauses while preserving sentence context."""
    clauses: list[Clause] = []

    for s in sentences:
        words = s.words
        if not words:
            continue

        current_words: list[QwenWordToken] = []
        for i, token in enumerate(words):
            current_words.append(token)
            has_terminal_punct = bool(
                token.punctuation in PUNCTUATIONS
                or token.text in PUNCTUATIONS
                or any(p in token.punctuation for p in PUNCTUATIONS)
            )

            has_silence_break = False
            if i + 1 < len(words):
                gap = words[i + 1].start_ms - token.end_ms
                if gap >= max_silence_gap_ms:
                    has_silence_break = True

            if has_terminal_punct or has_silence_break or i == len(words) - 1:
                if not current_words:
                    continue
                c_text = "".join(w.text + w.punctuation for w in current_words).strip()
                if c_text:
                    c_start = current_words[0].start_ms
                    c_end = max(c_start + 10, current_words[-1].end_ms)
                    clauses.append(
                        Clause(
                            index=len(clauses) + 1,
                            sentence_id=s.sentence_id,
                            text=c_text,
                            start_ms=c_start,
                            end_ms=c_end,
                            words=tuple(current_words),
                            orig_speaker_id=s.speaker_id,
                        )
                    )
                current_words = []

    return clauses


def find_closest_segment_speaker(
    clause: Clause,
    diarized_segments: Sequence[SpeakerSegment],
    fallback_speaker: int = 1,
) -> int:
    """Find the speaker of the temporally nearest segment when no direct overlap exists."""
    if not diarized_segments:
        return fallback_speaker

    closest_dist = float("inf")
    closest_spk = fallback_speaker

    for seg in diarized_segments:
        if seg.end_ms < clause.start_ms:
            dist = clause.start_ms - seg.end_ms
        elif seg.start_ms > clause.end_ms:
            dist = seg.start_ms - clause.end_ms
        else:
            return seg.speaker_id

        if dist < closest_dist:
            closest_dist = dist
            closest_spk = seg.speaker_id

    return closest_spk


def align_clauses_with_diarization(
    clauses: Sequence[Clause],
    diarized_segments: Sequence[SpeakerSegment],
    *,
    turn_merge_gap_ms: int = 1500,
    default_speaker: int = 1,
) -> tuple[Turn, ...]:
    """Project clauses onto authoritative 3D-Speaker + Nemotron segments and merge turns."""
    if not clauses:
        return ()

    # First pass: calculate direct overlaps
    clause_overlaps: list[dict[int, int]] = []
    sentence_speaker_votes: dict[int, dict[int, int]] = {}

    for clause in clauses:
        c_start = clause.start_ms
        c_end = clause.end_ms

        speaker_overlaps: dict[int, int] = {}
        for seg in diarized_segments:
            if seg.end_ms <= c_start or seg.start_ms >= c_end:
                continue
            ov = max(0, min(c_end, seg.end_ms) - max(c_start, seg.start_ms))
            if ov > 0:
                speaker_overlaps[seg.speaker_id] = speaker_overlaps.get(seg.speaker_id, 0) + ov

        clause_overlaps.append(speaker_overlaps)

        # Accumulate sentence-level evidence
        if clause.sentence_id not in sentence_speaker_votes:
            sentence_speaker_votes[clause.sentence_id] = {}
        for spk, ov in speaker_overlaps.items():
            sentence_speaker_votes[clause.sentence_id][spk] = (
                sentence_speaker_votes[clause.sentence_id].get(spk, 0) + ov
            )

    # Second pass: attribute each clause with sentence context fallback
    attributed_clauses: list[dict[str, object]] = []
    last_known_speaker = default_speaker

    for i, clause in enumerate(clauses):
        c_dur = clause.duration_ms
        speaker_overlaps = clause_overlaps[i]

        if speaker_overlaps:
            best_spk, best_ov = max(speaker_overlaps.items(), key=lambda item: item[1])
            ratio = best_ov / c_dur
            if ratio >= 0.60:
                conf = AlignmentConfidence.HIGH
            elif ratio >= 0.30:
                conf = AlignmentConfidence.MEDIUM
            else:
                conf = AlignmentConfidence.LOW

            has_overlap = sum(1 for ov in speaker_overlaps.values() if ov / c_dur >= 0.20) > 1
            last_known_speaker = best_spk
        else:
            # Check sentence-level consensus if current clause had 0 direct overlap (e.g. short particle)
            s_votes = sentence_speaker_votes.get(clause.sentence_id, {})
            if s_votes:
                best_spk = max(s_votes.items(), key=lambda item: item[1])[0]
                conf = AlignmentConfidence.MEDIUM
            else:
                best_spk = find_closest_segment_speaker(
                    clause, diarized_segments, fallback_speaker=last_known_speaker
                )
                conf = AlignmentConfidence.LOW
            has_overlap = False
            last_known_speaker = best_spk

        attributed_clauses.append(
            {
                "clause": clause,
                "speaker_id": best_spk,
                "confidence": conf,
                "overlap": has_overlap,
            }
        )

    # Third pass: Reassemble attributed clauses into conversational turns
    turns: list[Turn] = []
    curr_spk: int | None = None
    curr_start: int = 0
    curr_end: int = 0
    curr_texts: list[str] = []
    curr_overlap = False
    curr_conf = AlignmentConfidence.HIGH

    def _flush_turn() -> None:
        nonlocal curr_spk, curr_start, curr_end, curr_texts, curr_overlap, curr_conf
        if curr_spk is None or not curr_texts:
            return
        turn_text = "".join(curr_texts).strip()
        if turn_text:
            turns.append(
                Turn(
                    index=len(turns) + 1,
                    speaker_id=curr_spk,
                    start_ms=curr_start,
                    end_ms=max(curr_start + 10, curr_end),
                    text=turn_text,
                    source="qwen_audio_hybrid",
                    confidence=curr_conf,
                    overlap=curr_overlap,
                )
            )
        curr_texts.clear()
        curr_spk = None

    for item in attributed_clauses:
        c: Clause = item["clause"]  # type: ignore[assignment]
        spk: int = item["speaker_id"]  # type: ignore[assignment]
        conf: AlignmentConfidence = item["confidence"]  # type: ignore[assignment]
        ov: bool = item["overlap"]  # type: ignore[assignment]

        if curr_spk is None:
            curr_spk = spk
            curr_start = c.start_ms
            curr_end = c.end_ms
            curr_texts.append(c.text)
            curr_overlap = ov
            curr_conf = conf
        elif curr_spk == spk and (c.start_ms - curr_end < turn_merge_gap_ms):
            curr_texts.append(c.text)
            curr_end = max(curr_end, c.end_ms)
            curr_overlap = curr_overlap or ov
            if conf == AlignmentConfidence.LOW:
                curr_conf = AlignmentConfidence.LOW
            elif conf == AlignmentConfidence.MEDIUM and curr_conf == AlignmentConfidence.HIGH:
                curr_conf = AlignmentConfidence.MEDIUM
        else:
            _flush_turn()
            curr_spk = spk
            curr_start = c.start_ms
            curr_end = c.end_ms
            curr_texts.append(c.text)
            curr_overlap = ov
            curr_conf = conf

    _flush_turn()
    return tuple(turns)


def align_qwen_with_diarization(
    qwen_result: QwenTranscriptionResult,
    diarized_segments: Sequence[SpeakerSegment],
    *,
    turn_merge_gap_ms: int = 1500,
) -> Transcript:
    """Produce a standard Transcript from Qwen word stream aligned with authoritative segments."""
    clauses = split_sentences_into_clauses(qwen_result.sentences)
    turns = align_clauses_with_diarization(
        clauses,
        diarized_segments,
        turn_merge_gap_ms=turn_merge_gap_ms,
    )
    speaker_count = len(set(t.speaker_id for t in turns)) if turns else 1
    return Transcript(
        language="zh",
        speaker_count=speaker_count,
        turns=turns,
        warnings=(),
    )
