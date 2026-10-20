"""Evaluation tool uniting Qwen-Audio ASR with 3D-Speaker + Nemotron diarization.

Supports evaluating with cached Qwen JSON or live DashScope API.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from funasr_transcriber.config import load_config
from funasr_transcriber.exporters import export_transcript
from funasr_transcriber.qwen_client import QwenTranscriberClient, parse_qwen_transcription
from funasr_transcriber.qwen_hybrid_aligner import align_qwen_with_diarization
from funasr_transcriber.schemas import SpeakerSegment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("eval_qwen_hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align Qwen-Audio ASR with 3D-Speaker + Nemotron diarization",
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=Path,
        required=True,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--qwen-cache",
        type=Path,
        default=None,
        help="Path to cached Qwen transcription JSON (avoids API calls)",
    )
    parser.add_argument(
        "--refine-cache",
        type=Path,
        default=None,
        help="Path to cached refined transcript.json (avoids running diarization)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save aligned transcript.txt and transcript.json",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to config.toml",
    )
    parser.add_argument(
        "--turn-merge-gap-ms",
        type=int,
        default=1500,
        help="Maximum gap to merge same-speaker clauses into turns (default: 1500ms)",
    )
    return parser.parse_args()


def load_segments_from_transcript_json(path: Path) -> list[SpeakerSegment]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    segments = []
    for t in data.get("turns", []):
        spk = t.get("speaker") or t.get("speaker_id") or 1
        s = int(t["start_ms"])
        e = int(t["end_ms"])
        segments.append(SpeakerSegment(int(spk), s, e))
    return segments


def run_live_diarization(audio_path: Path, config_path: Path | None) -> list[SpeakerSegment]:
    """Run 3D-Speaker and Nemotron on GPU0 and refine segments."""
    from funasr_transcriber.nemotron_client import NemotronWorkerClient
    from funasr_transcriber.segment_refiner import (
        align_nemotron_to_3d_speaker,
        refine_segments_with_nemotron,
    )
    from funasr_transcriber.speaker_diarization import SpeakerDiarizer

    cfg = load_config(config_path)
    LOGGER.info("Running 3D-Speaker diarization on GPU0...")
    diarizer = SpeakerDiarizer(cfg)
    base_segments = diarizer.diarize(audio_path)
    LOGGER.info("3D-Speaker generated %d segments", len(base_segments))

    LOGGER.info("Running Nemotron-3 diarization on GPU0...")
    with NemotronWorkerClient(
        worker_python=cfg.nemotron.worker_python,
        model_dir=cfg.paths.model_dir,
        timeout_seconds=cfg.nemotron.timeout_seconds,
        stderr_log_path=cfg.paths.work_dir / "nemotron-eval.log",
    ) as nem_client:
        nem_segments = nem_client.diarize(audio_path)
    LOGGER.info("Nemotron generated %d segments", len(nem_segments))

    LOGGER.info("Aligning Nemotron channels to 3D-Speaker space...")
    aligned_nem, mapping = align_nemotron_to_3d_speaker(base_segments, nem_segments)
    LOGGER.info("Hungarian channel mapping: %s", mapping)

    LOGGER.info("Refining base segments with Nemotron...")
    refined_segs, stats = refine_segments_with_nemotron(base_segments, aligned_nem)
    LOGGER.info("Refined segments: %d (%s)", len(refined_segs), stats)
    return list(refined_segs)


def main() -> int:
    args = parse_args()
    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        LOGGER.error("Audio file not found: %s", audio_path)
        return 1

    # 1. Obtain Qwen transcription result
    if args.qwen_cache and args.qwen_cache.is_file():
        LOGGER.info("Loading cached Qwen ASR from %s", args.qwen_cache)
        with args.qwen_cache.open("r", encoding="utf-8") as f:
            qwen_raw = json.load(f)
        qwen_res = parse_qwen_transcription(qwen_raw)
    else:
        LOGGER.info("Calling DashScope Qwen-Audio API for %s...", audio_path.name)
        client = QwenTranscriberClient()
        qwen_res = client.transcribe(audio_path)

    LOGGER.info(
        "Qwen transcription: %d sentences, %d words",
        len(qwen_res.sentences),
        len(qwen_res.words),
    )

    # 2. Obtain authoritative diarization segments
    if args.refine_cache and args.refine_cache.is_file():
        LOGGER.info("Loading cached refined segments from %s", args.refine_cache)
        diarized_segments = load_segments_from_transcript_json(args.refine_cache)
    else:
        diarized_segments = run_live_diarization(audio_path, args.config)

    LOGGER.info("Using %d authoritative diarization segments", len(diarized_segments))

    # 3. Perform hybrid alignment
    LOGGER.info("Aligning Qwen words/clauses with authoritative diarization...")
    transcript = align_qwen_with_diarization(
        qwen_res,
        diarized_segments,
        turn_merge_gap_ms=args.turn_merge_gap_ms,
    )

    LOGGER.info(
        "Aligned result: %d turns, %d unique speakers",
        len(transcript.turns),
        transcript.speaker_count,
    )

    # 4. Export results
    out_dir = args.output_dir or Path("outputs/runtime/qwen_hybrid") / audio_path.stem
    txt_path, json_path = export_transcript(transcript, out_dir, overwrite=True)
    LOGGER.info("Exported transcript to:")
    LOGGER.info("  TXT : %s", txt_path)
    LOGGER.info("  JSON: %s", json_path)

    print("\n" + "=" * 70)
    print("=== First 10 Conversational Turns ===")
    print("=" * 70)
    for t in transcript.turns[:10]:
        print(f"[{t.start_ms//1000:02d}:{(t.start_ms%1000)//10:02d}] 说话人{t.speaker_id}：{t.text}")

    print("\n" + "=" * 70)
    print("=== Last 5 Conversational Turns ===")
    print("=" * 70)
    for t in transcript.turns[-5:]:
        print(f"[{t.start_ms//1000:02d}:{(t.start_ms%1000)//10:02d}] 说话人{t.speaker_id}：{t.text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
