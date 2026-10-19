import json
import os
import sys
import time
from pathlib import Path
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForAudioFrameClassification

def main():
    wav_path = "/mnt/hdd/jiajun/project/standalone_transcriber/outputs/runtime/audio/对话录音-1_1.wav"
    snapshot = "/mnt/hdd/jiajun/project/standalone_transcriber/models/nemotron3/a435e9867d79e789e90053f9b6d6834053af564a"
    out_path = "/mnt/hdd/jiajun/project/standalone_transcriber/outputs/runtime/nemotron_segments.json"
    
    print(f"Reading audio from {wav_path}...")
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    duration_s = audio.shape[0] / sample_rate
    duration_ms = round(duration_s * 1000)
    print(f"Audio duration: {duration_s:.2f} s ({duration_ms} ms)")
    
    print("Loading Nemotron processor and model on cuda:0...")
    torch.cuda.set_device(0)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(snapshot, local_files_only=True).to("cuda:0")
    model.eval()
    
    print("Processing audio features...")
    t0 = time.time()
    inputs = processor(audio[:, 0], sampling_rate=sample_rate).to(model.device, dtype=model.dtype)
    print(f"Feature extraction took {time.time() - t0:.2f} s")
    
    print("Running model inference...")
    t1 = time.time()
    with torch.inference_mode():
        logits = model(**inputs).logits
    print(f"Model forward pass took {time.time() - t1:.2f} s")
    
    print("Extracting speaker dict...")
    raw = processor.extract_speaker_dict(logits, inputs.attention_mask, threshold=0.5)[0]
    
    segments = []
    for item in raw:
        spk = int(item["Speaker"])
        s_ms = round(float(item["Start"]) * 1000)
        e_ms = round(float(item["End"]) * 1000)
        segments.append({
            "speaker_id": spk + 1,
            "start_ms": s_ms,
            "end_ms": e_ms,
            "start_s": round(s_ms / 1000.0, 3),
            "end_s": round(e_ms / 1000.0, 3),
        })
    segments.sort(key=lambda x: (x["start_ms"], x["speaker_id"], x["end_ms"]))
    print(f"Total Nemotron segments: {len(segments)}")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Saved segments to {out_path}")
    
    print("\n--- Segments around 375s - 400s (06:15 - 06:40) ---")
    for s in segments:
        if 375000 <= s["end_ms"] and s["start_ms"] <= 405000:
            print(f"[{s['start_s']:.2f}s - {s['end_s']:.2f}s] Speaker {s['speaker_id']}")

if __name__ == '__main__':
    main()
