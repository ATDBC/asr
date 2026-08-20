#!/usr/bin/env python
"""Encode one fixture into the release input-format matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from funasr_transcriber.benchmarking import input_format_matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, item in enumerate(input_format_matrix(), start=1):
        output = args.output_dir / f"case-{index:03d}{item['extension']}"
        completed = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(args.source),
                "-ar", str(item["sample_rate"]),
                "-ac", str(item["channels"]),
                str(output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"unable to build {output.name}: {completed.stderr.strip()}"
            )
        manifest.append({**item, "path": str(output.resolve())})
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

