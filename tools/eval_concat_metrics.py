#!/usr/bin/env python
"""CP-CER, CP-CER delta and cluster purity for a concatenated evaluation run.

Compares a transcript JSON produced by the pipeline against the RAMC TXT
references of the concatenated source recordings.  CP-CER uses the Hungarian
assignment over per-speaker concatenated character streams; purity is
duration-weighted overlap between predicted speaker turns and reference
segments.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

_CLEAN = re.compile(r"[^0-9a-zA-Z一-鿿]+")


def clean(text: str) -> str:
    return _CLEAN.sub("", text)


def load_ref(txt_root: Path, stems: list[str], block_shift_s: float):
    """Return (segments, per-speaker text, all-text) on one global timeline."""
    segments = []  # (start_s, end_s, global_speaker)
    spk_text = {}
    all_parts = []
    for block, stem in enumerate(stems):
        shift = block * block_shift_s
        for line in (txt_root / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            start_s, end_s = (float(x) for x in parts[0].strip("[]").split(","))
            gspk = f"{stem}:{parts[1]}"
            text = clean(parts[3])
            segments.append((start_s + shift, end_s + shift, gspk))
            spk_text.setdefault(gspk, []).append(text)
            all_parts.append(text)
    ref_spk = {k: "".join(v) for k, v in spk_text.items()}
    return segments, ref_spk, "".join(all_parts)


def lev(a: str, b: str) -> int:
    """Levenshtein distance, numpy-vectorized over anti-diagonals."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    A = np.fromiter((ord(c) for c in a), dtype=np.int64, count=la)
    B = np.fromiter((ord(c) for c in b), dtype=np.int64, count=lb)
    D = np.zeros((la + 1, lb + 1), dtype=np.int64)
    D[:, 0] = np.arange(la + 1)
    D[0, :] = np.arange(lb + 1)
    for d in range(2, la + lb + 1):
        i = np.arange(max(1, d - lb), min(la, d - 1) + 1)
        j = d - i
        mismatch = (A[i - 1] != B[j - 1]).astype(np.int64)
        D[i, j] = np.minimum(
            np.minimum(D[i - 1, j] + 1, D[i, j - 1] + 1),
            D[i - 1, j - 1] + mismatch,
        )
    return int(D[la, lb])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="pipeline transcript.json")
    parser.add_argument(
        "stems",
        nargs="+",
        help="RAMC TXT stems in concatenation order",
    )
    parser.add_argument(
        "--txt-root",
        type=Path,
        default=Path("artifacts/release_validation_corpus/ramc_core/TXT"),
    )
    parser.add_argument(
        "--block-shift-s",
        type=float,
        default=420.0,
        help="seconds each source recording is shifted by on the global timeline",
    )
    args = parser.parse_args()

    segments, ref_spk, ref_all = load_ref(args.txt_root, args.stems, args.block_shift_s)
    hyp = json.loads(args.transcript.read_text(encoding="utf-8"))
    turns = hyp["turns"]

    hyp_spk = {}
    hyp_parts = []
    for turn in sorted(turns, key=lambda t: t["start_ms"]):
        text = clean(turn["text"])
        hyp_spk.setdefault(str(turn["speaker"]), []).append(text)
        hyp_parts.append(text)
    hyp_spk = {k: "".join(v) for k, v in hyp_spk.items()}
    hyp_all = "".join(hyp_parts)

    total_ref = len(ref_all)
    cer = lev(ref_all, hyp_all) / total_ref

    ref_ids = sorted(ref_spk)
    hyp_ids = sorted(hyp_spk, key=int)
    cost = np.zeros((len(ref_ids), len(hyp_ids)))
    for i, r in enumerate(ref_ids):
        for j, h in enumerate(hyp_ids):
            cost[i, j] = lev(ref_spk[r], hyp_spk[h])
    ri, ci = linear_sum_assignment(cost)
    cp_cer = cost[ri, ci].sum() / total_ref
    mapping = {hyp_ids[j]: ref_ids[i] for i, j in zip(ri, ci)}

    overlap = {}  # hyp speaker -> {ref speaker: seconds}
    for turn in turns:
        t0, t1 = turn["start_ms"] / 1000, turn["end_ms"] / 1000
        per_ref = {}
        for r0, r1, gspk in segments:
            ov = min(t1, r1) - max(t0, r0)
            if ov > 0:
                per_ref[gspk] = per_ref.get(gspk, 0.0) + ov
        bucket = overlap.setdefault(str(turn["speaker"]), {})
        for gspk, ov in per_ref.items():
            bucket[gspk] = bucket.get(gspk, 0.0) + ov
    total_dur = sum(sum(v.values()) for v in overlap.values())
    correct = sum(max(v.values()) for v in overlap.values() if v)
    purity = correct / total_dur

    print(f"ref chars={total_ref}  hyp chars={len(hyp_all)}")
    print(f"CER     = {cer:.4f}")
    print(f"CP-CER  = {cp_cer:.4f}")
    print(f"delta   = {cp_cer - cer:+.4f}")
    print(f"purity  = {purity:.4f} (duration-weighted)")
    print("\noptimal hyp->ref mapping (cost = per-speaker CER):")
    for h, r in sorted(mapping.items(), key=lambda kv: int(kv[0])):
        i, j = ref_ids.index(r), hyp_ids.index(h)
        denom = max(1, len(ref_spk[r]))
        print(f"  spk{h} -> {r}  pair-CER={cost[i, j] / denom:.4f}")
    print("\nper predicted speaker: attributed seconds per ref speaker")
    for h in sorted(overlap, key=int):
        tops = sorted(overlap[h].items(), key=lambda kv: -kv[1])[:3]
        print(f"  spk{h}: " + ", ".join(f"{r}={v:.1f}s" for r, v in tops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
