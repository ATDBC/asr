#!/usr/bin/env python3
"""End-to-end validation of Solution 1 (TSE: Nemotron-3 Diarization + ClearVoice Separation + CAM++ Matching + FireRed ASR)."""

import json
from pathlib import Path
import re
import sys
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path("/mnt/hdd/jiajun/project/standalone_transcriber")
WAV_SOURCE = PROJECT_ROOT / "artifacts/ramc_full/MDT2021S003/WAV/CTS-CN-F2F-2019-11-15-359.wav"
TXT_SOURCE = PROJECT_ROOT / "artifacts/ramc_full/MDT2021S003/TXT/CTS-CN-F2F-2019-11-15-359.txt"
TEST_AUDIO = PROJECT_ROOT / "outputs/test_meeting_overlap.wav"
OUT_DIR = PROJECT_ROOT / "outputs/test_meeting_overlap_out"
CONFIG_PATH = PROJECT_ROOT / "outputs/runtime/nemotron_overlap.toml"


def prepare_synthetic_dialogue():
    audio, sr = sf.read(str(WAV_SOURCE), dtype="float32")
    assert sr == 16000

    def get_chunk(s_sec, e_sec):
        return audio[int(s_sec * sr):int(e_sec * sr)]

    def rms(data):
        return float(np.sqrt(np.mean(np.square(data))))

    def normalize_rms(data, target=0.08):
        current = rms(data)
        if current > 0:
            data = data * (target / current)
        return np.clip(data, -0.99, 0.99)

    # 1. Spk A Solo: [1.232, 5.418] ("爱数智慧语音采集二零一九年十一月九日")
    chunk_a_solo = normalize_rms(get_chunk(1.232, 5.418))
    # 2. Spk B Solo: [9.539, 11.752] ("平常我就玩玩王者荣耀吧")
    chunk_b_solo = normalize_rms(get_chunk(9.539, 11.752))

    # 3. Overlap Mix (3.5s):
    # Spk A: [16.424, 19.168] ("打排位多点还是打匹配多一点")
    # Spk B: [19.305, 22.824] ("匹配相当于练英雄吧，一般都是打排位的")
    dur_overlap = 3.5
    raw_a_over = get_chunk(16.424, 16.424 + dur_overlap)
    raw_b_over = get_chunk(19.305, 19.305 + dur_overlap)
    if len(raw_a_over) < int(dur_overlap * sr):
        raw_a_over = np.pad(raw_a_over, (0, int(dur_overlap * sr) - len(raw_a_over)))
    if len(raw_b_over) < int(dur_overlap * sr):
        raw_b_over = np.pad(raw_b_over, (0, int(dur_overlap * sr) - len(raw_b_over)))

    # Balanced 0.00 dB overlap
    chunk_a_over = normalize_rms(raw_a_over, target=0.08)
    chunk_b_over = normalize_rms(raw_b_over, target=0.08)
    mix_overlap = np.clip(chunk_a_over + chunk_b_over, -0.99, 0.99)

    # Assemble timeline:
    # 0.0s - 4.2s: Spk A Solo
    # 4.2s - 5.0s: 0.8s Silence
    # 5.0s - 7.2s: Spk B Solo
    # 7.2s - 8.2s: 1.0s Silence
    # 8.2s - 11.7s: Overlap (Spk A + Spk B)
    # 11.7s - 12.5s: 0.8s Silence
    silence = lambda sec: np.zeros(int(sec * sr), dtype="float32")

    meeting = np.concatenate([
        chunk_a_solo,
        silence(0.8),
        chunk_b_solo,
        silence(1.0),
        mix_overlap,
        silence(0.8),
    ])

    TEST_AUDIO.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(TEST_AUDIO), (meeting * 32767.0).astype(np.int16), sr, subtype="PCM_16")
    print(f"Created synthetic meeting audio at {TEST_AUDIO} (duration: {len(meeting)/sr:.2f}s)")
    print(f"  • Solo A: 0.0s - {len(chunk_a_solo)/sr:.2f}s")
    print(f"  • Solo B: {(len(chunk_a_solo)/sr + 0.8):.2f}s - {(len(chunk_a_solo)/sr + 0.8 + len(chunk_b_solo)/sr):.2f}s")
    print(f"  • Overlap A+B: 8.2s - 11.7s (3.5s balanced overlap)")


def prepare_config():
    cfg = """[general]
device = "cuda:0"

[paths]
model_dir = "/mnt/hdd/jiajun/project/standalone_transcriber/models"
work_dir = "/mnt/hdd/jiajun/project/standalone_transcriber/outputs/runtime/nemotron-work"

[asr]
worker_python = "/mnt/hdd/jiajun/project/standalone_transcriber/.venv-asr/bin/python"
aed_scheduler = "weighted_duration"
aed_worker_count = 1
use_half = true

[diarization]
backend = "nemotron3"
speaker_count = "auto"

[nemotron]
worker_python = "/mnt/hdd/jiajun/project/standalone_transcriber/.venv-nemotron/bin/python"
activity_threshold = 0.5

[overlap]
enabled = true
worker_python = "/mnt/hdd/jiajun/project/standalone_transcriber/.venv-overlap/bin/python"
detector_model = "pyannote/speaker-diarization-3.1"
hf_token_env = "HF_TOKEN"
timeout_seconds = 1200
speaker_match_min_score = 0.05
speaker_match_min_margin = 0.03
track_min_rms = 0.0005
"""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(cfg, encoding="utf-8")
    print(f"Wrote config to {CONFIG_PATH}")


def main():
    prepare_synthetic_dialogue()
    prepare_config()
    print("\nStarting transcription via funasr_transcriber CLI...")
    cmd = (
        f".venv/bin/python -m funasr_transcriber.cli "
        f"{TEST_AUDIO} "
        f"--config {CONFIG_PATH} "
        f"--output-dir {OUT_DIR} "
        f"--enable-overlap "
        f"--overwrite"
    )
    import subprocess
    ret = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if ret.returncode != 0:
        print(f"Transcription failed with returncode {ret.returncode}", file=sys.stderr)
        return ret.returncode

    print("\n" + "=" * 80)
    print("TRANSCRIPTION COMPLETED SUCCESSFULLY! Inspecting outputs:")
    print("=" * 80)

    txt_file = OUT_DIR / "transcript.txt"
    json_file = OUT_DIR / "transcript.json"
    diag_file = OUT_DIR / "diagnostics.json"

    if txt_file.is_file():
        print(f"\n--- [transcript.txt] ---")
        print(txt_file.read_text(encoding="utf-8"))

    if json_file.is_file():
        data = json.loads(json_file.read_text(encoding="utf-8"))
        print(f"\n--- [transcript.json summary] ---")
        print(f"Speaker count: {data.get('speaker_count')}")
        for idx, turn in enumerate(data.get("turns", []), 1):
            s = turn.get("start_ms", 0) / 1000
            e = turn.get("end_ms", 0) / 1000
            spk = turn.get("speaker")
            ovlp = turn.get("overlap")
            txt = turn.get("text")
            print(f"Turn {idx:02d} [{s:5.2f}s - {e:5.2f}s] (Spk {spk}, overlap={ovlp}): {txt}")

    if diag_file.is_file():
        diag = json.loads(diag_file.read_text(encoding="utf-8"))
        print(f"\n--- [diagnostics.json summary] ---")
        print(f"Warnings: {diag.get('warnings')}")
        if "speaker_crops" in diag.get("scheduler", {}):
            print(f"Speaker crops: {diag['scheduler']['speaker_crops']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
