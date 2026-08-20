#!/usr/bin/env python
"""Build the portable RAMC release-validation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from funasr_transcriber.corpus_builder import (
    CorpusCounts,
    build_validation_corpus,
    probe_wav,
)


def _check_command(command: str) -> None:
    try:
        completed = subprocess.run(
            [command, "-version"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"required command is unavailable: {command}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"required command failed: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    required = (
        args.source_root / "WAV",
        args.source_root / "TXT",
        args.source_root / "UTTERANCEINFO.txt",
        args.source_root / "SPKINFO.txt",
    )
    if not args.source_root.is_dir() or any(not path.exists() for path in required):
        print(f"source root is not a complete RAMC dataset: {args.source_root}", file=sys.stderr)
        return 2
    try:
        _check_command(args.ffmpeg)
        _check_command(args.ffprobe)
        output = build_validation_corpus(
            args.source_root,
            args.output_dir,
            counts=CorpusCounts(),
            seed=args.seed,
            ffmpeg=args.ffmpeg,
            overwrite=args.overwrite,
        )
        manifest = json.loads((output / "manifest.json").read_text("utf-8"))
        wavs = tuple(output.rglob("*.wav"))
        payload = {
            "output": str(output.resolve()),
            "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
            "audio_duration_ms": sum(probe_wav(path).duration_ms for path in wavs),
            "audio_files": len(wavs),
            "counts": manifest["counts"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
