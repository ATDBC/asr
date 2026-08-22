#!/usr/bin/env python
"""Evaluate merge rules on the recombined corpus with count + DER metrics.

For each conversation the diarization internals run once (VAD, chunks,
embeddings, cached under ``<corpus>/.cache``).  Clustering then runs with
oracle speaker counts auto/K/K+1/K+2 so each merge rule faces both true
over-segmentation (should merge) and real distinct speakers including a
fragment speaker (must not merge).

Rules: ``none`` (no merge), ``bare*`` (mean threshold only, guards disabled),
``final`` (the guarded production rule from
:func:`funasr_transcriber.speaker_diarization.merge_oversegmented_clusters`
with its default parameters).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from funasr_transcriber.config import load_config
from funasr_transcriber.speaker_diarization import (
    SpeakerDiarizer,
    _compress_overlapping_chunks,
    merge_oversegmented_clusters,
    relabel_by_first_appearance,
    seeded_cluster,
    smooth_segments,
)

FRAME_S = 0.01

RULES = [
    ("none", None),
    ("bare0.23", ("bare", 0.23)),
    ("bare0.35", ("bare", 0.35)),
    ("bare0.5", ("bare", 0.5)),
    ("final", ("final",)),
]


def apply_rule(embeddings, labels, rule):
    if rule is None:
        return labels
    if rule[0] == "bare":
        return merge_oversegmented_clusters(
            embeddings, labels, rule[1], min_fraction=0.0, relative_ratio=0.0
        )
    return merge_oversegmented_clusters(embeddings, labels, 0.23)


def segments_from(labels, chunks):
    labels = relabel_by_first_appearance(labels)
    raw = [(s, e, spk) for (s, e), spk in zip(chunks, labels)]
    return smooth_segments(_compress_overlapping_chunks(raw))


def frame_labels(segments, total_frames):
    frames = np.zeros(total_frames, dtype="int64")
    for start_s, end_s, speaker in segments:
        start = int(start_s / FRAME_S)
        end = min(total_frames, int(round(end_s / FRAME_S)))
        frames[start:end] = speaker
    return frames


def der(reference_turns, system_segments, duration_s):
    total_frames = int(round(duration_s / FRAME_S)) + 1
    ref = frame_labels(
        [(t["start_ms"] / 1000, t["end_ms"] / 1000, t["speaker_id"]) for t in reference_turns],
        total_frames,
    )
    sys_frames = frame_labels(system_segments, total_frames)
    speech = ref > 0
    ref_speakers = sorted(set(ref[speech].tolist()))
    sys_speakers = sorted(set(sys_frames[sys_frames > 0].tolist()))
    overlap = np.zeros((len(sys_speakers), len(ref_speakers)))
    for i, s in enumerate(sys_speakers):
        for j, r in enumerate(ref_speakers):
            overlap[i, j] = float(((sys_frames == s) & (ref == r)).sum())
    row, col = linear_sum_assignment(-overlap)
    mapping = {sys_speakers[i]: ref_speakers[j] for i, j in zip(row, col)}
    mapped = np.zeros_like(sys_frames)
    for s, r in mapping.items():
        mapped[sys_frames == s] = r
    miss = int((speech & (sys_frames == 0)).sum())
    fa = int(((~speech) & (sys_frames > 0)).sum())
    confusion = int((speech & (sys_frames > 0) & (mapped != ref)).sum())
    total = max(int(speech.sum()), 1)
    return (miss + fa + confusion) / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/recombined_corpus"),
        help="directory built by tools/build_recombined_corpus.py",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    args = parser.parse_args()

    config = load_config(args.config)
    diarizer = SpeakerDiarizer.from_config(config)
    cache_dir = args.corpus / ".cache"
    print("conversation oracle rule speakers der", flush=True)
    for wav in sorted(args.corpus.glob("*.wav")):
        truth = json.loads((args.corpus / f"{wav.stem}.json").read_text(encoding="utf-8"))
        true_k = len(truth["speakers"])
        duration_s = truth["turns"][-1]["end_ms"] / 1000
        cache = cache_dir / f"{wav.stem}.npz"
        if cache.is_file():
            saved = np.load(cache)
            embeddings = saved["embeddings"]
            chunks = [tuple(item) for item in saved["chunks"]]
        else:
            audio = diarizer._load_audio(wav)
            vad_segments = diarizer._vad(audio)
            chunks = [c for s, e in vad_segments for c in diarizer._chunks(s, e)]
            embeddings = diarizer._extract_embeddings(chunks, audio)
        for oracle in (None, true_k, true_k + 1, true_k + 2):
            oracle_name = "auto" if oracle is None else f"K{oracle - true_k:+d}"
            base = seeded_cluster(
                diarizer.cluster, embeddings, oracle, config.general.random_seed
            )
            base_k = len(set(base.tolist()))
            for rule_name, rule in RULES:
                labels = apply_rule(embeddings, base, rule)
                segments = segments_from(labels, chunks)
                value = der(truth["turns"], segments, duration_s)
                speakers = len({spk for _, _, spk in segments})
                print(
                    f"{wav.stem} {oracle_name}(base{base_k}) {rule_name}"
                    f" speakers={speakers} der={value:.4f}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
