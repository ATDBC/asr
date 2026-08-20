#!/usr/bin/env python
"""Verify a portable release-validation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from funasr_transcriber.corpus_verifier import verify_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    result = verify_corpus(args.corpus)
    manifest_path = args.corpus / "manifest.json"
    runner_path = args.corpus / "run_validation_linux.sh"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    print(
        json.dumps(
            {
                "ok": result.ok,
                "errors": list(result.errors),
                "counts": manifest.get("counts"),
                "release_evidence": manifest.get("release_evidence"),
                "runner_sha256": (
                    hashlib.sha256(runner_path.read_bytes()).hexdigest()
                    if runner_path.is_file()
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
