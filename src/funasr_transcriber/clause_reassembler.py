"""Non-destructive clause-level interruption and overlap reassembly using Nemotron-3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schemas import AlignmentConfidence, SpeakerSegment, Transcript, Turn

PUNCTS_STR = "，。？！、；：,!?;:"
PUNCTS = set(PUNCTS_STR)


@dataclass(frozen=True, slots=True)
class InterruptionEvent:
    turn_index: int
    clause_text: str
    original_speaker: int
    assigned_speaker: int
    start_ms: int
    end_ms: int
    interruption_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "clause_text": self.clause_text,
            "original_speaker": self.original_speaker,
            "assigned_speaker": self.assigned_speaker,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "interruption_ratio": round(self.interruption_ratio, 3),
        }


@dataclass(frozen=True, slots=True)
class ClauseReassemblyResult:
    transcript: Transcript
    interruption_events: tuple[InterruptionEvent, ...]
    speaker_mapping: dict[int, int]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "interruption_count": len(self.interruption_events),
            "speaker_mapping": {str(k): v for k, v in self.speaker_mapping.items()},
            "duration_seconds": round(self.duration_seconds, 3),
            "interruption_events": [e.to_dict() for e in self.interruption_events],
        }


def split_text_into_clauses(text: str) -> list[str]:
    """Split text into clause chunks, keeping ending punctuation attached."""
    clauses: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in PUNCTS:
            clauses.append("".join(current))
            current = []
    if current:
        clauses.append("".join(current))
    return clauses


def compute_bipartite_speaker_mapping(
    transcript_speakers: Sequence[int],
    nemotron_speakers: Sequence[int],
    overlap_matrix: np.ndarray,
    min_overlap_ms: int = 1000,
    min_purity: float = 0.50,
) -> dict[int, int]:
    """Compute optimal mapping from Nemotron-3 speaker channels to authoritative Transcript speakers.

    Transcript speakers are the authoritative speaker space (determined by 3D-Speaker).
    Nemotron-3 channels are projected onto this authoritative space.
    If Nemotron channels outnumber transcript speakers (k <= m), the top-k channels
    are matched via Hungarian assignment, and remaining unassigned channels are
    projected to their dominant transcript speaker if their overlap >= min_overlap_ms
    and purity >= min_purity (otherwise discarded as noise/chaff).
    No new speakers are ever introduced.
    """
    if len(transcript_speakers) == 0 or len(nemotron_speakers) == 0:
        return {}

    k = len(transcript_speakers)
    m = len(nemotron_speakers)
    mapping: dict[int, int] = {}
    assigned_cols: set[int] = set()

    if k <= m:
        row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
        for r, c in zip(row_ind, col_ind):
            mapping[nemotron_speakers[c]] = transcript_speakers[r]
            assigned_cols.add(c)

        for c in range(m):
            if c not in assigned_cols:
                col_ov = overlap_matrix[:, c]
                tot = float(np.sum(col_ov))
                if tot >= min_overlap_ms:
                    best_r = int(np.argmax(col_ov))
                    purity = float(col_ov[best_r]) / tot if tot > 0 else 0.0
                    if purity >= min_purity:
                        mapping[nemotron_speakers[c]] = transcript_speakers[best_r]
    else:
        row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
        for r, c in zip(row_ind, col_ind):
            mapping[nemotron_speakers[c]] = transcript_speakers[r]

    return mapping


def align_nemotron_speakers(
    turns: Sequence[Turn],
    nemotron_segments: Sequence[SpeakerSegment],
    expected_speaker_count: int = 1,
    min_overlap_ms: int = 1000,
    min_purity: float = 0.50,
) -> tuple[dict[int, int], list[SpeakerSegment]]:
    """Align Nemotron-3 speaker channels to Transcript speaker IDs via maximum overlap."""
    spk_ids = {t.speaker_id for t in turns}
    for i in range(1, expected_speaker_count + 1):
        spk_ids.add(i)
    t_spk_set = sorted(list(spk_ids))
    n_spk_set = sorted(list({s.speaker_id for s in nemotron_segments}))

    if not t_spk_set or not n_spk_set:
        return {}, []

    cost_matrix = np.zeros((len(t_spk_set), len(n_spk_set)), dtype=float)

    for turn in turns:
        if turn.speaker_id in t_spk_set:
            i = t_spk_set.index(turn.speaker_id)
            for seg in nemotron_segments:
                if seg.speaker_id in n_spk_set:
                    j = n_spk_set.index(seg.speaker_id)
                    overlap = max(
                        0,
                        min(turn.end_ms, seg.end_ms) - max(turn.start_ms, seg.start_ms),
                    )
                    cost_matrix[i, j] += overlap

    mapping = compute_bipartite_speaker_mapping(
        t_spk_set,
        n_spk_set,
        cost_matrix,
        min_overlap_ms=min_overlap_ms,
        min_purity=min_purity,
    )

    mapped_segments = [
        SpeakerSegment(
            speaker_id=mapping[seg.speaker_id],
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
        )
        for seg in nemotron_segments
        if seg.speaker_id in mapping
    ]
    return mapping, mapped_segments


def _calculate_overlaps(
    start_ms: int,
    end_ms: int,
    segments: Sequence[SpeakerSegment],
) -> dict[int, int]:
    overlaps: dict[int, int] = {}
    for seg in segments:
        ov = max(0, min(end_ms, seg.end_ms) - max(start_ms, seg.start_ms))
        if ov > 0:
            overlaps[seg.speaker_id] = overlaps.get(seg.speaker_id, 0) + ov
    return overlaps


def reassemble_transcript_with_nemotron(
    transcript: Transcript,
    nemotron_segments: Sequence[SpeakerSegment],
    min_interruption_ratio: float = 0.50,
    min_clause_ms: int = 400,
    min_overlap_ms: int = 1000,
    min_purity: float = 0.50,
) -> ClauseReassemblyResult:
    """Reattribute interrupting clauses using Nemotron-3 multi-label acoustic activity."""
    from time import perf_counter

    start_time = perf_counter()

    if not transcript.turns or not nemotron_segments:
        return ClauseReassemblyResult(
            transcript=transcript,
            interruption_events=(),
            speaker_mapping={},
            duration_seconds=perf_counter() - start_time,
        )

    mapping, aligned_segments = align_nemotron_speakers(
        transcript.turns,
        nemotron_segments,
        expected_speaker_count=transcript.speaker_count,
        min_overlap_ms=min_overlap_ms,
        min_purity=min_purity,
    )

    valid_speakers = {t.speaker_id for t in transcript.turns}
    for i in range(1, transcript.speaker_count + 1):
        valid_speakers.add(i)

    events: list[InterruptionEvent] = []
    new_turns_data: list[dict[str, Any]] = []

    for turn in transcript.turns:
        t_spk = turn.speaker_id
        t_start = turn.start_ms
        t_end = turn.end_ms
        t_dur = max(1, t_end - t_start)
        t_text = turn.text

        clauses = split_text_into_clauses(t_text)
        if len(clauses) <= 1:
            new_turns_data.append(
                {
                    "speaker_id": t_spk,
                    "start_ms": t_start,
                    "end_ms": t_end,
                    "text": t_text,
                    "source": turn.source,
                    "confidence": turn.confidence,
                    "overlap": turn.overlap,
                }
            )
            continue

        total_len = max(1, len(t_text))
        curr_ms = t_start

        for clause in clauses:
            clean_clause = clause.strip(PUNCTS_STR)
            c_dur = max(1, round(t_dur * (len(clause) / total_len)))
            c_start = curr_ms
            c_end = min(t_end, curr_ms + c_dur)
            curr_ms = c_end

            assigned_spk = t_spk
            is_interruption = False
            assigned_ratio = 0.0

            if c_dur >= min_clause_ms and len(clean_clause) > 1:
                overlaps = _calculate_overlaps(c_start, c_end, aligned_segments)
                curr_spk_ov = overlaps.get(t_spk, 0)
                clause_dur = max(1, c_end - c_start)

                for spk, ov_dur in overlaps.items():
                    if spk != t_spk and spk in valid_speakers:
                        ratio = ov_dur / clause_dur
                        if ratio >= min_interruption_ratio and ov_dur > curr_spk_ov:
                            assigned_spk = spk
                            is_interruption = True
                            assigned_ratio = ratio
                            break

            if is_interruption:
                events.append(
                    InterruptionEvent(
                        turn_index=turn.index,
                        clause_text=clause,
                        original_speaker=t_spk,
                        assigned_speaker=assigned_spk,
                        start_ms=c_start,
                        end_ms=c_end,
                        interruption_ratio=assigned_ratio,
                    )
                )

            new_turns_data.append(
                {
                    "speaker_id": assigned_spk,
                    "start_ms": c_start,
                    "end_ms": c_end,
                    "text": clause,
                    "source": (
                        "nemotron_interruption"
                        if is_interruption
                        else turn.source
                    ),
                    "confidence": (
                        AlignmentConfidence.MEDIUM
                        if is_interruption
                        else turn.confidence
                    ),
                    "overlap": is_interruption or turn.overlap,
                }
            )

    # Merge contiguous sub-clauses of the same speaker
    merged_turns_data: list[dict[str, Any]] = []
    for item in new_turns_data:
        if not merged_turns_data:
            merged_turns_data.append(dict(item))
            continue

        prev = merged_turns_data[-1]
        if prev["speaker_id"] == item["speaker_id"]:
            prev["end_ms"] = max(prev["end_ms"], item["end_ms"])
            prev["text"] += item["text"]
            prev["overlap"] = prev["overlap"] or item["overlap"]
            if item["source"] == "nemotron_interruption":
                prev["source"] = "nemotron_interruption"
        else:
            merged_turns_data.append(dict(item))

    final_turns = tuple(
        Turn(
            index=idx,
            speaker_id=item["speaker_id"],
            start_ms=item["start_ms"],
            end_ms=item["end_ms"],
            text=item["text"],
            source=item["source"],
            confidence=item["confidence"],
            overlap=item["overlap"],
        )
        for idx, item in enumerate(merged_turns_data, start=1)
    )

    new_transcript = Transcript(
        language=transcript.language,
        speaker_count=transcript.speaker_count,
        turns=final_turns,
        warnings=transcript.warnings,
    )

    duration = perf_counter() - start_time
    return ClauseReassemblyResult(
        transcript=new_transcript,
        interruption_events=tuple(events),
        speaker_mapping=mapping,
        duration_seconds=duration,
    )
