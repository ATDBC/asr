#!/usr/bin/env python
"""Verify a portable release-validation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from funasr_transcriber.corpus_verifier import verify_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    result = verify_corpus(args.corpus)
    print(
        json.dumps(
            {"ok": result.ok, "errors": list(result.errors)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
