#!/usr/bin/env python
"""Per-merge-event analysis on the recombined corpus (with embedding cache).

Replays the bare-0.23 merge loop and labels every merged pair as good or bad
by its effect on speaker count and DER, dumping the statistics each guard
condition would inspect: sizes, cross mean, cross p10, fraction above base,
within-cluster means.  Use this to recalibrate merge thresholds/guards.
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
    _normalized_embeddings,
    relabel_by_first_appearance,
    seeded_cluster,
    smooth_segments,
)

FRAME_S = 0.01
BASE = 0.23


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


def score(reference_turns, labels, chunks, duration_s):
    segments = segments_from(labels, chunks)
    total_frames = int(round(duration_s / FRAME_S)) + 1
    ref = frame_labels(
        [(t["start_ms"] / 1000, t["end_ms"] / 1000, t["speaker_id"]) for t in reference_turns],
        total_frames,
    )
    sys_frames = frame_labels(segments, total_frames)
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
    count = len({spk for _, _, spk in segments})
    return count, (miss + fa + confusion) / total


def embeddings_for(diarizer, wav: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{wav.stem}.npz"
    if cache.is_file():
        saved = np.load(cache)
        return saved["embeddings"], [tuple(item) for item in saved["chunks"]]
    audio = diarizer._load_audio(wav)
    vad_segments = diarizer._vad(audio)
    chunks = [c for s, e in vad_segments for c in diarizer._chunks(s, e)]
    embeddings = diarizer._extract_embeddings(chunks, audio)
    np.savez(cache, embeddings=embeddings, chunks=np.asarray(chunks))
    return embeddings, chunks


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
    print("event conversation oracle verdict sizes cross p10 frac ratio_within")
    for wav in sorted(args.corpus.glob("*.wav")):
        truth = json.loads((args.corpus / f"{wav.stem}.json").read_text(encoding="utf-8"))
        true_k = len(truth["speakers"])
        duration_s = truth["turns"][-1]["end_ms"] / 1000
        embeddings, chunks = embeddings_for(diarizer, wav, cache_dir)
        normalized = _normalized_embeddings(embeddings)
        for oracle in (None, true_k, true_k + 1, true_k + 2):
            oracle_name = "auto" if oracle is None else f"K{oracle - true_k:+d}"
            labels = [int(x) for x in seeded_cluster(
                diarizer.cluster, embeddings, oracle, config.general.random_seed
            )]
            groups = {}
            for index, label in enumerate(labels):
                groups.setdefault(label, []).append(index)
            count_before, der_before = score(truth["turns"], labels, chunks, duration_s)
            while len(groups) > 1:
                ids = sorted(groups)
                best_pair, best_value = None, -1.0
                for position, left in enumerate(ids):
                    for right in ids[position + 1:]:
                        value = float((normalized[groups[left]] @ normalized[groups[right]].T).mean())
                        if value > best_value:
                            best_value, best_pair = value, (left, right)
                if best_pair is None or best_value < BASE:
                    break
                keep, drop = best_pair
                la, lb = groups[keep], groups[drop]
                cross = normalized[la] @ normalized[lb].T
                within = []
                for group in (la, lb):
                    if len(group) > 1:
                        matrix = normalized[group] @ normalized[group].T
                        within.append(float(matrix[np.triu_indices(len(group), 1)].mean()))
                trial = {k: list(v) for k, v in groups.items()}
                trial[keep] = la + lb
                del trial[drop]
                trial_labels = [0] * len(labels)
                for new_label, indexes in enumerate(sorted(trial.values(), key=min)):
                    for index in indexes:
                        trial_labels[index] = new_label
                count_after, der_after = score(truth["turns"], trial_labels, chunks, duration_s)
                bad = count_after < true_k or der_after > der_before + 0.01
                print(
                    f"merge {wav.stem} {oracle_name}"
                    f" {'BAD' if bad else 'good'}"
                    f" sizes={len(la)}/{len(lb)}"
                    f" cross={best_value:.3f}"
                    f" p10={float(np.percentile(cross, 10)):.3f}"
                    f" frac={float((cross >= BASE).mean()):.3f}"
                    f" within={'/'.join(f'{w:.3f}' for w in within)}"
                    f" ratio={best_value / min(within) if within else float('nan'):.3f}",
                    flush=True,
                )
                groups = trial
                count_before, der_before = count_after, der_after
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
