#!/usr/bin/env python
"""Build deterministic two-speaker overlap mixtures from clean source clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from funasr_transcriber.benchmarking import build_overlap_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sources = json.loads(args.sources_json.read_text(encoding="utf-8"))
    manifest = build_overlap_manifest(sources, args.count, args.seed)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        output = audio_dir / f"{item['id']}.wav"
        seconds = float(item["overlap_seconds"])
        second_gain = 10 ** (-float(item["snr_db"]) / 20)
        filter_graph = (
            f"[0:a]atrim=duration={seconds},aresample=16000,"
            "aformat=channel_layouts=mono[a0];"
            f"[1:a]atrim=duration={seconds},volume={second_gain},"
            "aresample=16000,aformat=channel_layouts=mono[a1];"
            "[a0][a1]amix=inputs=2:duration=shortest:normalize=0[out]"
        )
        completed = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(item["first"]["path"]),
                "-i", str(item["second"]["path"]),
                "-filter_complex", filter_graph,
                "-map", "[out]",
                "-acodec", "pcm_s16le",
                str(output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip())
        item["mixture_path"] = str(output.resolve())
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

