#!/usr/bin/env python
"""Concatenate PCM WAV files into one file (format must match across inputs)."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import wave


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    params = None
    for path in args.inputs:
        with contextlib.closing(wave.open(str(path))) as handle:
            current = handle.getparams()
        if params is None:
            params = current
        elif (
            current.nchannels,
            current.sampwidth,
            current.framerate,
        ) != (params.nchannels, params.sampwidth, params.framerate):
            raise ValueError(f"format mismatch: {path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(wave.open(str(args.out), "wb")) as out:
        out.setparams(params)
        for path in args.inputs:
            with contextlib.closing(wave.open(str(path))) as handle:
                out.writeframes(handle.readframes(handle.getnframes()))
    with contextlib.closing(wave.open(str(args.out))) as out:
        seconds = out.getnframes() / out.getframerate()
    print(f"{args.out} {seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
