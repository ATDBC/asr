#!/usr/bin/env python3
"""
Waveform-Level Target Speaker Extraction (TSE) Pipeline Evaluation.

Architecture:
1. Stage 1 (ClearVoice MossFormer2-SS):
   - Separates the mixed overlap audio into 2 clean acoustic tracks (Track 1 & Track 2).
2. Stage 2 (CAM++ Speaker Identity Anchoring + FireRed ASR):
   - CAM++ extracts voiceprint embeddings from Track 1 and Track 2.
   - Computes cosine similarity against Reference Voiceprint A and Reference Voiceprint B.
   - Automatically routes:
       Track A = Track with highest affinity to Voiceprint A
       Track B = Track with highest affinity to Voiceprint B
   - Transcribes Track A and Track B using vanilla, unpatched FireRedASR2.
3. Evaluated on the exact same 5-pair 0.00 dB SNR balanced benchmark.
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path("/mnt/hdd/jiajun/project/standalone_transcriber").resolve()
MODEL_DIR = PROJECT_ROOT / "models"
TMP_DIR = PROJECT_ROOT / "outputs" / "tse_eval_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def compute_cer(ref: str, hyp: str) -> float:
    ref = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", ref)
    hyp = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    r_len, h_len = len(ref), len(hyp)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[r_len][h_len] / r_len


def balance_and_mix(wav_a: np.ndarray, wav_b: np.ndarray, target_rms: float = 3000.0):
    length = min(len(wav_a), len(wav_b))
    w_a = wav_a[:length].astype(np.float32)
    w_b = wav_b[:length].astype(np.float32)

    rms_a = np.sqrt(np.mean(w_a ** 2)) + 1e-8
    rms_b = np.sqrt(np.mean(w_b ** 2)) + 1e-8

    w_a_norm = w_a * (target_rms / rms_a)
    w_b_norm = w_b * (target_rms / rms_b)

    mix = w_a_norm + w_b_norm
    max_amp = np.max(np.abs(mix))
    if max_amp > 32000.0:
        scale = 32000.0 / max_amp
        mix = mix * scale
        w_a_norm = w_a_norm * scale
        w_b_norm = w_b_norm * scale

    return mix.astype(np.int16), w_a_norm.astype(np.int16), w_b_norm.astype(np.int16)


# ---------------------------------------------------------------------------
# Stage 1: Speech Separation with ClearVoice MossFormer2
# ---------------------------------------------------------------------------

def run_stage1_separation(test_cases):
    from clearvoice import ClearVoice

    print("\n=== Stage 1: Running MossFormer2 Speech Separation on GPU 0 ===")
    t0 = time.time()
    cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    print(f"  • Loaded ClearVoice MossFormer2 in {time.time() - t0:.2f}s")

    for idx, case in enumerate(test_cases, 1):
        mix_wav = case["mix_int16"]
        mix_float = torch.from_numpy(mix_wav.astype(np.float32) / 32767.0).unsqueeze(0)

        t_sep = time.time()
        with torch.no_grad():
            separated = cv.call_t2t_mode(mix_float)
        dur = len(mix_wav) / 16000.0
        sep_time = time.time() - t_sep
        print(f"  • Case {idx}: Separated {dur:.2f}s mixture into 2 tracks in {sep_time:.3f}s (RTF: {sep_time / dur:.3f})")

        # separated is shape (2, 1, T)
        tr1 = (separated[0, 0] * 32767.0).astype(np.int16)
        tr2 = (separated[1, 0] * 32767.0).astype(np.int16)

        sf.write(str(TMP_DIR / f"case_{idx}_track1.wav"), tr1, 16000)
        sf.write(str(TMP_DIR / f"case_{idx}_track2.wav"), tr2, 16000)


# ---------------------------------------------------------------------------
# Stage 2: Speaker Identity Matching & FireRed ASR Transcription
# ---------------------------------------------------------------------------

def run_stage2_transcription(test_cases):
    FIRERED_SRC = MODEL_DIR / "fireredasr2s" / "source"
    SPEAKER_REPO = MODEL_DIR / "3d-speaker" / "3D-Speaker"
    for path in [FIRERED_SRC, SPEAKER_REPO]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from fireredasr2s.fireredasr2.asr import FireRedAsr2
    from speakerlab.utils.builder import build
    from speakerlab.utils.config import Config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Stage 2: Speaker Matching & FireRed Transcription on {device} ===")

    # 1. Load FireRed ASR
    asr_system = FireRedAsr2.from_pretrained("aed", str(MODEL_DIR / "fireredasr2s" / "FireRedASR2-AED"))
    model = asr_system.model.to(device)
    model.eval()

    # 2. Load CAM++
    spec = {
        "embedding_model": {
            "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
            "args": {"feat_dim": 80, "embedding_size": 192},
        },
        "feature_extractor": {
            "obj": "speakerlab.process.processor.FBank",
            "args": {"n_mels": 80, "sample_rate": 16000, "mean_nor": True},
        },
    }
    ckpt_path = MODEL_DIR / "modelscope" / "iic" / "speech_campplus_sv_zh_en_16k-common_advanced" / "campplus_cn_en_common.pt"
    config = Config(spec)
    feat_extractor = build("feature_extractor", config)
    spk_model = build("embedding_model", config)
    spk_model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu", weights_only=False))
    spk_model.eval().to(device)

    @torch.no_grad()
    def get_embedding(wav_int16: np.ndarray) -> torch.Tensor:
        w = torch.from_numpy(wav_int16.astype(np.float32) / 32767.0).unsqueeze(0).to(device)
        f = feat_extractor(w).unsqueeze(0)
        emb = spk_model(f)
        return F.normalize(emb, p=2, dim=-1).squeeze(0)

    def decode_wav(wav_int16: np.ndarray) -> str:
        feats, lens, _, _, _ = asr_system.feat_extractor([(16000, wav_int16)], ["eval"])
        feats, lens = feats.to(device), lens.to(device)
        with torch.no_grad():
            out, _, mask = model.encoder(feats, lens)
            hyp = model.decoder.batch_beam_search(out, mask, beam_size=1)
            text = asr_system.tokenizer.detokenize(hyp[0][0]["yseq"].tolist())
        return text

    results = []

    print("\n" + "=" * 80)
    print("TSE PIPELINE BENCHMARK RESULTS (0.00 dB SNR Balanced Overlap)")
    print("=" * 80)

    for idx, case in enumerate(test_cases, 1):
        sa, sb = case["spk_a"], case["spk_b"]
        t_a, t_b = case["text_a"], case["text_b"]
        ref_emb_a = case["emb_a"]
        ref_emb_b = case["emb_b"]
        mix_wav = case["mix_int16"]

        # 1. Unconditioned baseline decode (direct mixture)
        text_uncond = decode_wav(mix_wav)
        cer_u_a = compute_cer(t_a, text_uncond)
        cer_u_b = compute_cer(t_b, text_uncond)

        # 2. Read separated tracks
        tr1, _ = sf.read(str(TMP_DIR / f"case_{idx}_track1.wav"), dtype="int16")
        tr2, _ = sf.read(str(TMP_DIR / f"case_{idx}_track2.wav"), dtype="int16")

        # 3. CAM++ Speaker Identity Matching
        emb_tr1 = get_embedding(tr1)
        emb_tr2 = get_embedding(tr2)

        sim_1_a = F.cosine_similarity(emb_tr1, ref_emb_a, dim=-1).item()
        sim_2_a = F.cosine_similarity(emb_tr2, ref_emb_a, dim=-1).item()
        sim_1_b = F.cosine_similarity(emb_tr1, ref_emb_b, dim=-1).item()
        sim_2_b = F.cosine_similarity(emb_tr2, ref_emb_b, dim=-1).item()

        # Assignment matching
        score_1a_2b = sim_1_a + sim_2_b
        score_2a_1b = sim_2_a + sim_1_b

        if score_1a_2b >= score_2a_1b:
            wav_target_a = tr1
            wav_target_b = tr2
            conf_a, conf_b = sim_1_a, sim_2_b
            routing = "Track 1 -> Spk A, Track 2 -> Spk B"
        else:
            wav_target_a = tr2
            wav_target_b = tr1
            conf_a, conf_b = sim_2_a, sim_1_b
            routing = "Track 2 -> Spk A, Track 1 -> Spk B"

        # 4. Transcribe separated target waveforms with vanilla FireRed
        text_target_a = decode_wav(wav_target_a)
        text_target_b = decode_wav(wav_target_b)

        cer_tse_a = compute_cer(t_a, text_target_a)
        cer_tse_b = compute_cer(t_b, text_target_b)

        results.append({
            "cer_u_a": cer_u_a,
            "cer_u_b": cer_u_b,
            "cer_tse_a": cer_tse_a,
            "cer_tse_b": cer_tse_b,
        })

        print(f"\n--- [Case {idx}: {sa} vs {sb}] (0.00 dB SNR) ---")
        print(f"  • Ref A: \"{t_a}\"")
        print(f"  • Ref B: \"{t_b}\"")
        print(f"  • Matching: {routing} (Cosine Sim: A={conf_a:.3f}, B={conf_b:.3f})")
        print(f"  [Unconditioned]: \"{text_uncond}\" | CER vs A: {cer_u_a:.2f}, CER vs B: {cer_u_b:.2f}")
        print(f"  [TSE Target A]:  \"{text_target_a}\" | CER vs A: {cer_tse_a:.2f}")
        print(f"  [TSE Target B]:  \"{text_target_b}\" | CER vs B: {cer_tse_b:.2f}")

    avg_u_a = np.mean([r["cer_u_a"] for r in results])
    avg_u_b = np.mean([r["cer_u_b"] for r in results])
    avg_tse_a = np.mean([r["cer_tse_a"] for r in results])
    avg_tse_b = np.mean([r["cer_tse_b"] for r in results])

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY (TSE Pipeline vs Unconditioned Baseline)")
    print("=" * 80)
    print(f"Speaker A: Baseline CER: {avg_u_a:.3f} -> TSE Extracted CER: {avg_tse_a:.3f} ({'-' if avg_tse_a <= avg_u_a else '+'}{abs(avg_tse_a - avg_u_a):.3f})")
    print(f"Speaker B: Baseline CER: {avg_u_b:.3f} -> TSE Extracted CER: {avg_tse_b:.3f} ({'-' if avg_tse_b <= avg_u_b else '+'}{abs(avg_tse_b - avg_u_b):.3f})")


# ---------------------------------------------------------------------------
# Data Preparation and Multi-Env Coordinator
# ---------------------------------------------------------------------------

def prepare_test_cases():
    corpus_dir = PROJECT_ROOT / "artifacts" / "ramc_full" / "MDT2021S003"
    txt_dir = corpus_dir / "TXT"
    wav_dir = corpus_dir / "WAV"

    wav_files = sorted(list(wav_dir.glob("*.wav")))
    segments_by_spk = {}
    clean_re = re.compile(r"\[([0-9.]+),([0-9.]+)\]\s+([A-Za-z0-9_]+)\s+[^	\n]+\s+(.*)")

    for wav_file in wav_files:
        txt_file = txt_dir / (wav_file.stem + ".txt")
        if not txt_file.is_file():
            continue
        try:
            audio_data, sr = sf.read(str(wav_file), dtype="float32")
        except Exception:
            continue
        if sr != 16000:
            continue
        audio_int16 = (audio_data * 32767.0).astype(np.int16)

        with open(txt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                m = clean_re.match(line)
                if not m:
                    continue
                start_s, end_s, spk_id, text = m.groups()
                start_s = float(start_s)
                end_s = float(end_s)
                dur = end_s - start_s
                text = re.sub(r"\[\+?\]|\(.*?\)|\<.*?\>", "", text).strip()
                if 2.0 <= dur <= 5.0 and len(text) >= 5:
                    start_sample = int(round(start_s * sr))
                    end_sample = int(round(end_s * sr))
                    chunk = audio_int16[start_sample:end_sample]
                    if len(chunk) >= 16000:
                        if spk_id not in segments_by_spk:
                            segments_by_spk[spk_id] = []
                        segments_by_spk[spk_id].append((chunk, text))

    filtered = {k: v for k, v in segments_by_spk.items() if len(v) >= 6}
    speakers = list(filtered.keys())

    test_pairs = [
        (speakers[0], speakers[1]),
        (speakers[2], speakers[3]),
        (speakers[4], speakers[5]),
        (speakers[6], speakers[7]),
        (speakers[8], speakers[9]),
    ]

    # Pre-extract CAM++ embeddings for reference
    SPEAKER_REPO = MODEL_DIR / "3d-speaker" / "3D-Speaker"
    sys.path.insert(0, str(SPEAKER_REPO))
    from speakerlab.utils.builder import build
    from speakerlab.utils.config import Config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = {
        "embedding_model": {
            "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
            "args": {"feat_dim": 80, "embedding_size": 192},
        },
        "feature_extractor": {
            "obj": "speakerlab.process.processor.FBank",
            "args": {"n_mels": 80, "sample_rate": 16000, "mean_nor": True},
        },
    }
    ckpt_path = MODEL_DIR / "modelscope" / "iic" / "speech_campplus_sv_zh_en_16k-common_advanced" / "campplus_cn_en_common.pt"
    config = Config(spec)
    f_ext = build("feature_extractor", config)
    s_model = build("embedding_model", config)
    s_model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu", weights_only=False))
    s_model.eval().to(device)

    def extract_ref_emb(utts):
        embs = []
        for u in utts[:3]:
            w = torch.from_numpy(u[0].astype(np.float32) / 32767.0).unsqueeze(0).to(device)
            f = f_ext(w).unsqueeze(0)
            e = s_model(f)
            embs.append(F.normalize(e, p=2, dim=-1).squeeze(0))
        return F.normalize(torch.stack(embs).mean(dim=0), p=2, dim=-1)

    test_cases = []
    for sa, sb in test_pairs:
        utt_a = filtered[sa][-1]
        utt_b = filtered[sb][-1]
        mix_int16, _, _ = balance_and_mix(utt_a[0], utt_b[0], target_rms=3000.0)
        emb_a = extract_ref_emb(filtered[sa][:-1])
        emb_b = extract_ref_emb(filtered[sb][:-1])

        test_cases.append({
            "spk_a": sa,
            "spk_b": sb,
            "text_a": utt_a[1],
            "text_b": utt_b[1],
            "mix_int16": mix_int16,
            "emb_a": emb_a,
            "emb_b": emb_b,
        })

    return test_cases


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "stage1":
        # Run separation only
        cases = torch.load(str(TMP_DIR / "cases.pt"), weights_only=False)
        run_stage1_separation(cases)
    elif mode == "stage2":
        # Run transcription only
        cases = torch.load(str(TMP_DIR / "cases.pt"), weights_only=False)
        run_stage2_transcription(cases)
    else:
        # Coordinator
        print("=== Preparing Test Cases & Reference Voiceprints ===")
        cases = prepare_test_cases()
        torch.save(cases, str(TMP_DIR / "cases.pt"))
        print(f"  • Saved {len(cases)} test cases to {TMP_DIR / 'cases.pt'}")

        # Run Stage 1 in .venv-overlap
        cmd1 = f"cd {PROJECT_ROOT} && .venv-overlap/bin/python {__file__} stage1"
        os.system(cmd1)

        # Run Stage 2 in .venv-asr
        cmd2 = f"cd {PROJECT_ROOT} && .venv-asr/bin/python {__file__} stage2"
        os.system(cmd2)
