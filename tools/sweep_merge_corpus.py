#!/usr/bin/env python
"""Corpus-wide merge sweep with reference speaker counts and best-pair stats.

For every WAV in the RAMC corpus this prints the reference speaker count, the
base cluster count, the surviving speaker count under each merge rule
(``None`` = no merge, bare thresholds with guards disabled, ``guarded`` = the
production rule with default parameters), plus statistics of the most similar
cluster pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from funasr_transcriber.config import load_config
from funasr_transcriber.speaker_diarization import (
    SpeakerDiarizer,
    _compress_overlapping_chunks,
    _normalized_embeddings,
    merge_oversegmented_clusters,
    relabel_by_first_appearance,
    seeded_cluster,
    smooth_segments,
)

THRESHOLDS = (None, 0.23, 0.5, "guarded")


def reference_speaker_count(txt_path: Path) -> int:
    speakers = set()
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].startswith("G") and parts[1] != "G00000000":
            speakers.add(parts[1])
    return len(speakers)


def outcome(embeddings, labels, chunks, threshold):
    if threshold is None:
        merged = labels
    elif threshold == "guarded":
        merged = merge_oversegmented_clusters(embeddings, labels, 0.23)
    else:
        merged = merge_oversegmented_clusters(
            embeddings, labels, threshold,
            min_fraction=0.0, relative_ratio=0.0,
        )
    merged = relabel_by_first_appearance(merged)
    raw = [(s, e, spk) for (s, e), spk in zip(chunks, merged)]
    smoothed = smooth_segments(_compress_overlapping_chunks(raw))
    return len({spk for _, _, spk in smoothed}), len(smoothed)


def best_pair_stats(embeddings, labels):
    normalized = _normalized_embeddings(embeddings)
    labels = [int(label) for label in labels]
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    stats = []
    ids = sorted(groups)
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            la, lb = groups[left], groups[right]
            cross = normalized[la] @ normalized[lb].T
            within_a = normalized[la] @ normalized[la].T
            within_b = normalized[lb] @ normalized[lb].T
            stats.append(
                (
                    float(cross.mean()),
                    len(la),
                    len(lb),
                    float(np.percentile(cross, 10)),
                    float(within_a[np.triu_indices(len(la), 1)].mean()) if len(la) > 1 else 1.0,
                    float(within_b[np.triu_indices(len(lb), 1)].mean()) if len(lb) > 1 else 1.0,
                )
            )
    stats.sort(reverse=True)
    return stats[0] if stats else None, len(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/release_validation_corpus/ramc_core"),
        help="RAMC corpus root containing WAV/ and TXT/",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    args = parser.parse_args()

    config = load_config(args.config)
    diarizer = SpeakerDiarizer.from_config(config)

    for wav in sorted((args.root / "WAV").glob("*.wav")):
        ref = reference_speaker_count(args.root / "TXT" / f"{wav.stem}.txt")
        audio = diarizer._load_audio(wav)
        vad_segments = diarizer._vad(audio)
        chunks = [c for s, e in vad_segments for c in diarizer._chunks(s, e)]
        embeddings = diarizer._extract_embeddings(chunks, audio)
        base = seeded_cluster(
            diarizer.cluster, embeddings,
            config.diarization.speaker_count, config.general.random_seed,
        )
        counts = [
            outcome(embeddings, base, chunks, threshold)[0] for threshold in THRESHOLDS
        ]
        pair, n_clusters = best_pair_stats(embeddings, base)
        line = (
            f"{wav.stem} ref={ref} base={n_clusters} "
            f"speakers@None/0.23/0.5/guarded={counts}"
        )
        if pair is not None:
            cross, size_a, size_b, p10, within_a, within_b = pair
            line += (
                f" | best-pair cross={cross:.3f} p10={p10:.3f}"
                f" sizes={size_a}/{size_b} within={within_a:.3f}/{within_b:.3f}"
            )
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
