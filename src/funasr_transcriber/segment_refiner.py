"""Pre-ASR acoustic segment refinement using concurrent Nemotron-3 multi-label diarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schemas import SpeakerSegment


def align_nemotron_to_base_speakers(
    base_segments: Sequence[SpeakerSegment],
    nemotron_segments: Sequence[SpeakerSegment],
    min_overlap_ms: int = 500,
    min_purity: float = 0.50,
) -> tuple[dict[int, int], list[SpeakerSegment]]:
    """Map Nemotron-3 speaker channels onto 3D-Speaker authoritative speaker space.

    3D-Speaker provides the authoritative global speaker clusters.
    Nemotron channels are mapped via maximum bipartite overlap.
    Unassigned excess Nemotron channels are projected to their dominant base
    speaker if overlap >= min_overlap_ms and purity >= min_purity, or discarded.
    No new speaker IDs are ever introduced.
    """
    base_spks = sorted({s.speaker_id for s in base_segments})
    nem_spks = sorted({s.speaker_id for s in nemotron_segments})
    if not base_spks or not nem_spks:
        return {}, []

    cost_matrix = np.zeros((len(base_spks), len(nem_spks)), dtype=float)
    for b in base_segments:
        i = base_spks.index(b.speaker_id)
        for n in nemotron_segments:
            j = nem_spks.index(n.speaker_id)
            ov = max(0, min(b.end_ms, n.end_ms) - max(b.start_ms, n.start_ms))
            cost_matrix[i, j] += ov

    k = len(base_spks)
    m = len(nem_spks)
    mapping: dict[int, int] = {}
    assigned_cols: set[int] = set()

    if k <= m:
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        for r, c in zip(row_ind, col_ind):
            mapping[nem_spks[c]] = base_spks[r]
            assigned_cols.add(c)
        for c in range(m):
            if c not in assigned_cols:
                col_ov = cost_matrix[:, c]
                tot = float(np.sum(col_ov))
                if tot >= min_overlap_ms:
                    best_r = int(np.argmax(col_ov))
                    purity = float(col_ov[best_r]) / tot if tot > 0 else 0.0
                    if purity >= min_purity:
                        mapping[nem_spks[c]] = base_spks[best_r]
    else:
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        for r, c in zip(row_ind, col_ind):
            mapping[nem_spks[c]] = base_spks[r]

    mapped_segments = [
        SpeakerSegment(
            speaker_id=mapping[s.speaker_id],
            start_ms=s.start_ms,
            end_ms=s.end_ms,
        )
        for s in nemotron_segments
        if s.speaker_id in mapping
    ]
    mapped_segments.sort(key=lambda s: (s.start_ms, s.end_ms))
    return mapping, mapped_segments


def refine_segments_with_nemotron(
    base_segments: Sequence[SpeakerSegment],
    nemotron_segments: Sequence[SpeakerSegment],
    *,
    max_overlap_ratio: float = 0.60,
    min_independent_ms: int = 250,
) -> tuple[tuple[SpeakerSegment, ...], dict[str, int]]:
    """Refine 3D-Speaker coarse segments with high-resolution Nemotron multi-label activity.

    If 3D-Speaker missed a turn transition (e.g. attributed全程 to Speaker 1, but
    Nemotron reveals Speaker 2 spoke with substantial non-overlapping speech),
    split the segment into 4-cut acoustic intervals. If Nemotron also shows heavy
    overlap (> max_overlap_ratio), keep it unbroken to preserve primary sentence context.
    """
    if not base_segments or not nemotron_segments:
        return tuple(base_segments), {"splits": 0, "inserted_count": 0}

    mapping, aligned_nemotron = align_nemotron_to_base_speakers(
        base_segments, nemotron_segments
    )
    if not aligned_nemotron:
        return tuple(base_segments), {"splits": 0, "inserted_count": 0}

    refined: list[SpeakerSegment] = []
    splits = 0
    total_inserted = 0

    for s in base_segments:
        others = [
            n for n in aligned_nemotron
            if n.speaker_id != s.speaker_id
            and max(s.start_ms, n.start_ms) < min(s.end_ms, n.end_ms)
        ]
        if not others:
            refined.append(s)
            continue

        raw_inserted = []
        for o in others:
            o_start = max(s.start_ms, o.start_ms)
            o_end = min(s.end_ms, o.end_ms)
            o_dur = o_end - o_start
            if o_dur < min_independent_ms:
                continue

            s_nem = [
                n for n in aligned_nemotron
                if n.speaker_id == s.speaker_id
                and max(o_start, n.start_ms) < min(o_end, n.end_ms)
            ]
            overlap_ms = sum(
                min(o_end, n.end_ms) - max(o_start, n.start_ms)
                for n in s_nem
            )
            ratio = overlap_ms / o_dur if o_dur > 0 else 0.0
            if ratio <= max_overlap_ratio and (o_dur - overlap_ms) >= min_independent_ms:
                raw_inserted.append((o.speaker_id, o_start, o_end))

        raw_inserted.sort(key=lambda x: (x[1], x[2]))
        inserted: list[tuple[int, int, int]] = []
        last_ins_end = -1
        for ins_spk, ins_start, ins_end in raw_inserted:
            if ins_start < last_ins_end:
                ins_start = last_ins_end
            if ins_end >= ins_start + min_independent_ms:
                inserted.append((ins_spk, ins_start, ins_end))
                last_ins_end = ins_end

        if not inserted:
            refined.append(s)
            continue

        splits += 1
        total_inserted += len(inserted)
        current_start = s.start_ms
        for ins_spk, ins_start, ins_end in inserted:
            s_nem_before = [
                n for n in aligned_nemotron
                if n.speaker_id == s.speaker_id
                and n.start_ms < ins_start and n.end_ms > current_start
            ]
            part1_end = max([n.end_ms for n in s_nem_before], default=ins_start)
            part1_end = min(part1_end, ins_start)
            if part1_end > current_start + 100:
                refined.append(SpeakerSegment(s.speaker_id, current_start, part1_end))

            refined.append(SpeakerSegment(ins_spk, ins_start, ins_end))

            s_nem_after = [
                n for n in aligned_nemotron
                if n.speaker_id == s.speaker_id
                and n.end_ms > ins_end and n.start_ms < s.end_ms
            ]
            next_start = min([n.start_ms for n in s_nem_after], default=ins_end)
            current_start = max(next_start, ins_end)

        if s.end_ms > current_start + 100:
            refined.append(SpeakerSegment(s.speaker_id, current_start, s.end_ms))

    refined.sort(key=lambda seg: (seg.start_ms, seg.end_ms))

    cleaned: list[SpeakerSegment] = []
    prev_e = -1
    for seg in refined:
        if seg.end_ms <= prev_e + 100:
            continue
        cur_s = max(seg.start_ms, prev_e)
        cur_e = max(seg.end_ms, cur_s + 100)
        cleaned.append(SpeakerSegment(seg.speaker_id, cur_s, cur_e))
        prev_e = cur_e

    return tuple(cleaned), {"splits": splits, "inserted_count": total_inserted}
