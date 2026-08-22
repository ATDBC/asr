#!/usr/bin/env python
"""Build synthetic K-speaker conversations by recombining RAMC utterances.

Each conversation interleaves utterances from K distinct (file, speaker) sources
with short silence gaps, producing a 16 kHz mono PCM16 WAV plus exact turn-level
ground truth.  A "fragment" variant gives one speaker only a few short utterances.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
SEED = 20260821

GAP_SAME_SPEAKER_S = 0.25
GAP_SPEAKER_CHANGE_S = 0.45
MIN_UTTERANCE_S = 0.4
MAX_UTTERANCE_S = 10.0
TARGET_CONVERSATION_S = 150.0

LINE = re.compile(r"\[(\d+\.\d+),(\d+\.\d+)\]\t(\S+)\t")

SPECS = [
    ("k2-a", 2), ("k2-b", 2), ("k2-c", 2),
    ("k3-a", 3), ("k3-b", 3), ("k3-c", 3),
    ("k4-a", 4), ("k4-b", 4), ("k4-c", 4),
    ("k5-a", 5), ("k5-b", 5),
    ("k3frag-a", 2), ("k3frag-b", 2),
    ("k4frag-a", 3),
]


def load_speaker_pool(root: Path) -> dict[tuple[str, str], list[tuple[float, float]]]:
    pool: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for txt in sorted((root / "TXT").glob("*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            match = LINE.match(line)
            if not match:
                continue
            start_s, end_s, speaker = float(match[1]), float(match[2]), match[3]
            if not speaker.startswith("G") or speaker == "G00000000":
                continue
            duration = end_s - start_s
            if MIN_UTTERANCE_S <= duration <= MAX_UTTERANCE_S:
                pool.setdefault((txt.stem, speaker), []).append((start_s, end_s))
    return pool


class Source:
    def __init__(self, key, utterances):
        self.file, self.original_speaker = key
        self.utterances = utterances
        self.cursor = 0

    def next_utterance(self):
        if self.cursor >= len(self.utterances):
            return None
        utterance = self.utterances[self.cursor]
        self.cursor += 1
        return utterance


def build_conversation(rng, sources, fragment_source=None, fragment_utterances=3):
    timeline = []  # (source_index, out_start_s, out_end_s, src_start_s, src_end_s)
    cursor_s = 0.0
    order = list(range(len(sources)))
    previous = None
    fragment_inserted = 0
    fragment_at = (
        sorted(rng.sample(range(8, 40), fragment_utterances))
        if fragment_source is not None
        else []
    )
    turn_index = 0
    while cursor_s < TARGET_CONVERSATION_S:
        candidates = [i for i in order if i != previous and sources[i].cursor < len(sources[i].utterances)]
        if not candidates:
            break
        current = rng.choice(candidates)
        run_length = rng.randint(1, 3)
        for _ in range(run_length):
            utterance = sources[current].next_utterance()
            if utterance is None:
                break
            start_s, end_s = utterance
            if cursor_s > 0:
                cursor_s += GAP_SAME_SPEAKER_S if current == previous else GAP_SPEAKER_CHANGE_S
            timeline.append((current, cursor_s, cursor_s + (end_s - start_s), start_s, end_s))
            cursor_s += end_s - start_s
            previous = current
            turn_index += 1
            if fragment_source is not None and fragment_inserted < len(fragment_at) and turn_index >= fragment_at[fragment_inserted]:
                fragment = fragment_source.next_utterance()
                if fragment is not None:
                    fs, fe = fragment
                    cursor_s += GAP_SPEAKER_CHANGE_S
                    timeline.append((len(sources), cursor_s, cursor_s + (fe - fs), fs, fe))
                    cursor_s += fe - fs
                    fragment_inserted += 1
                    previous = None
    return timeline


def render(timeline, all_sources, root: Path, wav_path: Path, truth_path: Path):
    audio_cache: dict[str, np.ndarray] = {}

    def load(file):
        if file not in audio_cache:
            data, rate = sf.read(str(root / "WAV" / f"{file}.wav"), dtype="float32")
            assert rate == SAMPLE_RATE
            audio_cache[file] = data
        return audio_cache[file]

    pieces = []
    turns = []
    for source_index, out_start_s, out_end_s, src_start_s, src_end_s in timeline:
        source = all_sources[source_index]
        data = load(source.file)
        piece = data[int(src_start_s * SAMPLE_RATE):int(src_end_s * SAMPLE_RATE)]
        pieces.append(piece)
        turns.append({
            "speaker_id": source_index + 1,
            "start_ms": int(round(out_start_s * 1000)),
            "end_ms": int(round(out_end_s * 1000)),
        })
    total_ms = int(round(timeline[-1][2] * 1000))
    output = np.zeros(total_ms * SAMPLE_RATE // 1000, dtype="float32")
    for piece, (_, out_start_s, _, _, _) in zip(pieces, timeline):
        start = int(round(out_start_s * SAMPLE_RATE))
        output[start:start + len(piece)] = piece
    sf.write(str(wav_path), output, SAMPLE_RATE, subtype="PCM_16")
    speakers = [
        {"id": index + 1, "source_file": source.file, "source_speaker": source.original_speaker}
        for index, source in enumerate(all_sources)
    ]
    truth_path.write_text(
        json.dumps({"speakers": speakers, "turns": turns}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/release_validation_corpus/ramc_core"),
        help="RAMC corpus root containing WAV/ and TXT/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/recombined_corpus"),
        help="output directory for synthetic conversations",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    pool = load_speaker_pool(args.root)
    keys = sorted(pool)
    rng.shuffle(keys)
    args.out.mkdir(parents=True, exist_ok=True)
    key_cycle = iter(keys * 4)

    def take_sources(count):
        chosen = []
        for _ in range(count):
            key = next(key_cycle)
            chosen.append(Source(key, pool[key]))
        return chosen

    for name, main_count in SPECS:
        sources = take_sources(main_count)
        fragment = None
        if "frag" in name:
            key = next(key_cycle)
            fragment = Source(key, pool[key])
        all_sources = sources + ([fragment] if fragment is not None else [])
        timeline = build_conversation(rng, sources, fragment)
        render(
            timeline,
            all_sources,
            args.root,
            args.out / f"{name}.wav",
            args.out / f"{name}.json",
        )
        print(
            f"{name}: speakers={len(all_sources)} turns={len(timeline)}"
            f" seconds={timeline[-1][2]:.1f}"
            + (f" fragment={fragment.original_speaker}" if fragment else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
