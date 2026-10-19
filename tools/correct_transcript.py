"""CLI tool to run DeepSeek LLM post-correction on an existing transcript.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from funasr_transcriber.config import LLMConfig, load_config
from funasr_transcriber.exporters import export_transcript
from funasr_transcriber.llm_corrector import DeepSeekCorrector
from funasr_transcriber.schemas import AlignmentConfidence, Transcript, Turn


def load_transcript_from_json(path: Path) -> Transcript:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    turns = []
    for t in data.get("turns", []):
        confidence_str = t.get("alignment_confidence", "high").lower()
        confidence = AlignmentConfidence(confidence_str) if confidence_str in AlignmentConfidence.__members__.values() else AlignmentConfidence.HIGH
        turns.append(
            Turn(
                index=t["index"],
                speaker_id=t.get("speaker", 1),
                start_ms=t["start_ms"],
                end_ms=t["end_ms"],
                text=t["text"],
                source=t.get("source", "primary_asr"),
                confidence=confidence,
                overlap=t.get("overlap", False),
            )
        )

    return Transcript(
        language=data.get("language", "zh"),
        speaker_count=data.get("speaker_count", 1),
        turns=tuple(turns),
        warnings=tuple(data.get("warnings", ())),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek LLM post-correction on transcript.json"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input transcript.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to output directory or json file (defaults to <input_dir>/corrected)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to config.toml",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Direct DeepSeek API key (overrides DEEPSEEK_API_KEY environment variable)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: deepseek-v4.1-flash)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Phonetic gate threshold (default: 0.5)",
    )
    parser.add_argument(
        "--nemotron-segments",
        type=Path,
        default=None,
        help="Path to nemotron_segments.json for clause interruption reassembly",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM post-correction (only run Nemotron interruption reassembly if provided)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without calling remote API",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        print(f"Error: input file '{input_path}' not found.", file=sys.stderr)
        return 1

    app_config = load_config(args.config)
    llm_cfg = app_config.llm

    # Overrides
    api_key = args.api_key or llm_cfg.resolve_api_key()
    model = args.model or llm_cfg.model
    threshold = args.threshold if args.threshold is not None else llm_cfg.pinyin_gate_threshold

    effective_config = LLMConfig(
        enabled=not args.no_llm,
        provider="deepseek",
        api_key=api_key,
        api_key_env=llm_cfg.api_key_env,
        base_url=llm_cfg.base_url,
        model=model,
        enable_thinking=False,
        temperature=llm_cfg.temperature,
        timeout_seconds=llm_cfg.timeout_seconds,
        pinyin_gate_threshold=threshold,
        max_context_turns=llm_cfg.max_context_turns,
    )

    print(f"Loading transcript from: {input_path}")
    current_transcript = load_transcript_from_json(input_path)
    print(f"Loaded {len(current_transcript.turns)} turns, {current_transcript.speaker_count} speakers.")

    reassembly_result = None
    if args.nemotron_segments is not None:
        from funasr_transcriber.clause_reassembler import reassemble_transcript_with_nemotron
        from funasr_transcriber.schemas import SpeakerSegment

        nemo_path = args.nemotron_segments.resolve()
        if not nemo_path.is_file():
            print(f"Error: nemotron segments file '{nemo_path}' not found.", file=sys.stderr)
            return 1
        with nemo_path.open("r", encoding="utf-8") as f:
            raw_segs = json.load(f)
        nemo_segments = tuple(
            SpeakerSegment(
                speaker_id=int(s["speaker_id"]),
                start_ms=int(s["start_ms"]),
                end_ms=int(s["end_ms"]),
            )
            for s in raw_segs
        )
        print(f"Loaded {len(nemo_segments)} Nemotron-3 segments from {nemo_path}")
        print("Running Nemotron-3 clause interruption reassembly...")
        reassembly_result = reassemble_transcript_with_nemotron(
            current_transcript,
            nemo_segments,
        )
        current_transcript = reassembly_result.transcript
        print(f"Interruption reassembly complete in {reassembly_result.duration_seconds:.2f}s.")
        print(f"Detected and reattributed {len(reassembly_result.interruption_events)} interruptions.")
        if reassembly_result.interruption_events:
            print("\n--- Nemotron-3 Interruption Events ---")
            for e in reassembly_result.interruption_events[:15]:
                print(
                    f"  Turn {e.turn_index} [{e.start_ms/1000:.2f}s - {e.end_ms/1000:.2f}s]: "
                    f"'{e.clause_text}' Spk {e.original_speaker} -> Spk {e.assigned_speaker} "
                    f"(overlap ratio: {e.interruption_ratio:.2f})"
                )
            if len(reassembly_result.interruption_events) > 15:
                print(f"  ... and {len(reassembly_result.interruption_events) - 15} more.")

    correction_result = None
    if not args.no_llm:
        if not api_key and not args.dry_run:
            print(
                "Error: DEEPSEEK_API_KEY is not set.\n"
                "Please provide it via --api-key, environment variable, or in .env file:\n"
                "  DEEPSEEK_API_KEY=sk-...\n",
                file=sys.stderr,
            )
            return 2

        corrector = DeepSeekCorrector(effective_config)
        print(f"\nCalling DeepSeek corrector (model: {model}, non-thinking mode, gate threshold: {threshold})...")
        correction_result = corrector.correct(current_transcript)
        current_transcript = correction_result.transcript

        print(f"\nCompleted in {correction_result.duration_seconds:.2f}s.")
        print(f"Entities discovered: {list(correction_result.entities_discovered)}")
        print(f"Patches applied: {len(correction_result.patches_applied)}")
        print(f"Patches rejected: {len(correction_result.patches_rejected)}")

        if correction_result.patches_applied:
            print("\n--- Applied Patches ---")
            for p in correction_result.patches_applied:
                print(
                    f"  Turn {p.turn_index}: '{p.original}' -> '{p.corrected}' "
                    f"(sim: {p.similarity:.3f}, pinyin: '{p.original_pinyin}' -> '{p.candidate_pinyin}')\n"
                    f"    Reason: {p.reason}"
                )

        if correction_result.patches_rejected:
            print("\n--- Rejected Patches (Phonetic Gate) ---")
            for p in correction_result.patches_rejected:
                print(
                    f"  Turn {p.turn_index}: '{p.original}' -> '{p.corrected}' "
                    f"(sim: {p.similarity:.3f}, reason: {p.rejection_reason})"
                )

    if args.output is not None:
        out_dir = args.output.resolve()
        if out_dir.suffix == ".json":
            out_dir = out_dir.parent
    else:
        out_dir = input_path.parent / "corrected"

    out_dir.mkdir(parents=True, exist_ok=True)
    export_transcript(current_transcript, out_dir, overwrite=True)

    audit_payload: dict[str, Any] = {}
    if reassembly_result is not None:
        audit_payload["nemotron_reassembly"] = reassembly_result.to_dict()
    if correction_result is not None:
        audit_payload["llm_correction"] = correction_result.to_dict()

    audit_path = out_dir / "correction_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit_payload, f, ensure_ascii=False, indent=2)

    print(f"\nFinal transcript exported to: {out_dir}")
    print(f"Audit log saved to: {audit_path}")
    return 0

    print(f"\nCorrected transcript exported to: {out_dir}")
    print(f"Audit log saved to: {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
