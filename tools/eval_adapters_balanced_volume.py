#!/usr/bin/env python3
"""
Evaluate Single-Layer and Multi-Layer FiLM Adapters under Balanced (0 dB SNR) Volume.

Both speakers are normalized to identical RMS energy so neither has an acoustic
dominance advantage. We test:
1. Baseline (Unconditioned)
2. Single-Layer Input FiLM (speaker_film_adapter_poc.pt)
3. 16-Layer Multi-Layer FiLM (multilayer_film_adapter.pt)
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("/mnt/hdd/jiajun/project/standalone_transcriber").resolve()
MODEL_DIR = PROJECT_ROOT / "models"
FIRERED_SRC = MODEL_DIR / "fireredasr2s" / "source"
SPEAKER_REPO = MODEL_DIR / "3d-speaker" / "3D-Speaker"

for path in [FIRERED_SRC, SPEAKER_REPO]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fireredasr2s.fireredasr2.asr import FireRedAsr2
from speakerlab.utils.builder import build
from speakerlab.utils.config import Config


# ---------------------------------------------------------------------------
# 1. Adapter Architectures
# ---------------------------------------------------------------------------

class SingleLayerFiLMAdapter(nn.Module):
    def __init__(self, spk_dim: int = 192, d_model: int = 1280):
        super().__init__()
        self.proj = nn.Linear(spk_dim, 2 * d_model)

    def forward(self, x: torch.Tensor, spk_emb: torch.Tensor | None = None) -> torch.Tensor:
        if spk_emb is None:
            return x
        if spk_emb.dim() == 1:
            spk_emb = spk_emb.unsqueeze(0)
        spk_emb = F.normalize(spk_emb.to(dtype=x.dtype, device=x.device), p=2, dim=-1)
        scale, shift = self.proj(spk_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return (1.0 + scale) * x + shift


class MultiLayerFiLMAdapter(nn.Module):
    def __init__(self, num_layers: int = 16, spk_dim: int = 192, d_model: int = 1280):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(spk_dim, 2 * d_model) for _ in range(num_layers)
        ])

    def forward(self, layer_idx: int, x: torch.Tensor, spk_emb: torch.Tensor | None = None) -> torch.Tensor:
        if spk_emb is None:
            return x
        if spk_emb.dim() == 1:
            spk_emb = spk_emb.unsqueeze(0)
        spk_emb = F.normalize(spk_emb.to(dtype=x.dtype, device=x.device), p=2, dim=-1)
        scale, shift = self.layers[layer_idx](spk_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return (1.0 + scale) * x + shift


def patch_conformer_universal(encoder):
    """Universal patch that can handle None, single-layer, or multi-layer adapter."""
    def forward_patched(padded_input, input_lengths, pad=True, spk_emb=None):
        if pad:
            padded_input = F.pad(
                padded_input,
                (0, 0, 0, encoder.input_preprocessor.context - 1),
                "constant",
                0.0,
            )
        src_mask = encoder.padding_position_is_0(padded_input, input_lengths)
        embed_output, input_lengths, src_mask = encoder.input_preprocessor(
            padded_input, src_mask
        )

        adapter_type = getattr(encoder, "adapter_mode", None)
        if adapter_type == "single" and getattr(encoder, "single_adapter", None) is not None:
            embed_output = encoder.single_adapter(embed_output, spk_emb)

        enc_output = encoder.dropout(embed_output)
        pos_emb = encoder.dropout(encoder.positional_encoding(embed_output))

        for idx, enc_layer in enumerate(encoder.layer_stack):
            enc_output = enc_layer(
                enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
            )
            if adapter_type == "multi" and getattr(encoder, "multi_adapter", None) is not None:
                enc_output = encoder.multi_adapter(idx, enc_output, spk_emb)

        return enc_output, input_lengths, src_mask

    encoder.forward = forward_patched
    return encoder


# ---------------------------------------------------------------------------
# 2. CAM++ 3D-Speaker Voiceprint Extractor
# ---------------------------------------------------------------------------

class VoiceprintExtractor:
    def __init__(self, device: torch.device):
        self.device = device
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
        self.feature_extractor = build("feature_extractor", config)
        self.model = build("embedding_model", config)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def extract(self, wav_float: np.ndarray) -> torch.Tensor:
        wav_t = torch.from_numpy(wav_float).unsqueeze(0).to(self.device)
        feat = self.feature_extractor(wav_t).unsqueeze(0)
        emb = self.model(feat)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.squeeze(0)


# ---------------------------------------------------------------------------
# 3. Audio Processing & Data Loader
# ---------------------------------------------------------------------------

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
    """
    Truncates to common length, normalizes both to target_rms (strictly 0.0 dB SNR),
    and sums them. Returns (mix, w_a_norm, w_b_norm, actual_snr_db).
    """
    length = min(len(wav_a), len(wav_b))
    w_a = wav_a[:length].astype(np.float32)
    w_b = wav_b[:length].astype(np.float32)

    rms_a = np.sqrt(np.mean(w_a ** 2)) + 1e-8
    rms_b = np.sqrt(np.mean(w_b ** 2)) + 1e-8

    w_a_norm = w_a * (target_rms / rms_a)
    w_b_norm = w_b * (target_rms / rms_b)

    calc_rms_a = np.sqrt(np.mean(w_a_norm ** 2))
    calc_rms_b = np.sqrt(np.mean(w_b_norm ** 2))
    snr_db = 20.0 * np.log10(calc_rms_a / calc_rms_b)

    mix = w_a_norm + w_b_norm
    max_amp = np.max(np.abs(mix))
    if max_amp > 32000.0:
        scale = 32000.0 / max_amp
        mix = mix * scale
        w_a_norm = w_a_norm * scale
        w_b_norm = w_b_norm * scale

    return mix.astype(np.int16), w_a_norm.astype(np.int16), w_b_norm.astype(np.int16), snr_db


def load_ramc_dataset(max_files=16):
    corpus_dir = PROJECT_ROOT / "artifacts" / "cost_model_train_corpus" / "ramc-v5-wave1" / "extracted" / "MDT2021S003"
    txt_dir = corpus_dir / "TXT"
    wav_dir = corpus_dir / "WAV"

    txt_files = sorted(list(txt_dir.glob("*.txt")))[:max_files]
    segments_by_spk = {}
    clean_re = re.compile(r"\[([0-9.]+),([0-9.]+)\]\s+([A-Za-z0-9_]+)\s+[^	\n]+\s+(.*)")

    for txt_file in txt_files:
        wav_file = wav_dir / (txt_file.stem + ".wav")
        if not wav_file.is_file():
            continue
        audio_data, sr = sf.read(str(wav_file), dtype="float32")
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
                    if spk_id not in segments_by_spk:
                        segments_by_spk[spk_id] = []
                    segments_by_spk[spk_id].append((chunk, text))

    filtered = {k: v for k, v in segments_by_spk.items() if len(v) >= 6}
    return filtered


# ---------------------------------------------------------------------------
# 4. Evaluation Execution
# ---------------------------------------------------------------------------

def run_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Balanced Volume (0.0 dB SNR) Overlap Evaluation on {device} ===")

    # 1. Load ASR Model
    print("  • Loading FireRedASR2-AED...")
    asr_system = FireRedAsr2.from_pretrained("aed", str(MODEL_DIR / "fireredasr2s" / "FireRedASR2-AED"))
    model = asr_system.model.to(device)
    model.eval()

    # 2. Patch Conformer
    patch_conformer_universal(model.encoder)

    # 3. Load Single-Layer Adapter
    single_ckpt = PROJECT_ROOT / "outputs" / "speaker_film_adapter_poc.pt"
    single_adapter = SingleLayerFiLMAdapter(spk_dim=192, d_model=model.encoder.odim).to(device)
    if single_ckpt.is_file():
        single_adapter.load_state_dict(torch.load(str(single_ckpt), map_location=device))
        print(f"  • Loaded Single-Layer Adapter ({single_ckpt.name})")
    else:
        print(f"  ! Warning: {single_ckpt} not found")
    single_adapter.eval()
    model.encoder.single_adapter = single_adapter

    # 4. Load Multi-Layer Adapter
    multi_ckpt = PROJECT_ROOT / "outputs" / "multilayer_film_adapter.pt"
    multi_adapter = MultiLayerFiLMAdapter(
        num_layers=model.encoder.n_layers_enc if hasattr(model.encoder, "n_layers_enc") else 16,
        spk_dim=192,
        d_model=model.encoder.odim
    ).to(device)
    if multi_ckpt.is_file():
        multi_adapter.load_state_dict(torch.load(str(multi_ckpt), map_location=device))
        print(f"  • Loaded 16-Layer Adapter ({multi_ckpt.name})")
    else:
        print(f"  ! Warning: {multi_ckpt} not found")
    multi_adapter.eval()
    model.encoder.multi_adapter = multi_adapter

    # 5. Load CAM++
    print("  • Loading CAM++ 3D-Speaker...")
    vp_extractor = VoiceprintExtractor(device)

    # 6. Load RAMC Dataset
    print("  • Loading RAMC multi-speaker dataset...")
    spk_data = load_ramc_dataset(max_files=16)
    speakers = list(spk_data.keys())
    print(f"  • Loaded {len(speakers)} qualified speakers.")

    # Compute reference voiceprints
    spk_embeddings = {}
    for spk in speakers[:12]:
        utts = spk_data[spk]
        embs = [vp_extractor.extract(u[0].astype(np.float32) / 32767.0) for u in utts[:4]]
        spk_embeddings[spk] = F.normalize(torch.stack(embs).mean(dim=0), p=2, dim=-1)

    # Form 5 distinct speaker pairs
    test_pairs = [
        (speakers[0], speakers[1]),
        (speakers[2], speakers[3]),
        (speakers[4], speakers[5]),
        (speakers[6], speakers[7]),
        (speakers[8], speakers[9]),
    ]

    def decode_audio(wav_int16, mode=None, spk_emb=None):
        model.encoder.adapter_mode = mode
        feats, lens, _, _, _ = asr_system.feat_extractor([(16000, wav_int16)], ["eval"])
        feats, lens = feats.to(device), lens.to(device)
        with torch.no_grad():
            out, _, mask = model.encoder(feats, lens, spk_emb=spk_emb)
            hyp = model.decoder.batch_beam_search(out, mask, beam_size=1)
            text = asr_system.tokenizer.detokenize(hyp[0][0]["yseq"].tolist())
        return text

    results = []

    print("\n" + "=" * 80)
    print("STARTING BALANCED OVERLAP EVALUATION (0.0 dB SNR - NO VOLUME ADVANTAGE)")
    print("=" * 80)

    for idx, (sa, sb) in enumerate(test_pairs, 1):
        utt_a = spk_data[sa][-1]
        utt_b = spk_data[sb][-1]
        w_a, t_a = utt_a
        w_b, t_b = utt_b

        # Perfectly balance RMS energy
        mix_int16, w_a_norm, w_b_norm, snr_db = balance_and_mix(w_a, w_b, target_rms=3000.0)

        # Baseline Clean decodes (for ground-truth validation)
        clean_dec_a = decode_audio(w_a_norm, mode=None)
        clean_dec_b = decode_audio(w_b_norm, mode=None)

        # 1. Unconditioned Mixture Decode (Baseline)
        uncond_dec = decode_audio(mix_int16, mode=None)

        # 2. Single-Layer Adapter Decodes
        emb_a = spk_embeddings[sa].unsqueeze(0)
        emb_b = spk_embeddings[sb].unsqueeze(0)

        single_dec_a = decode_audio(mix_int16, mode="single", spk_emb=emb_a)
        single_dec_b = decode_audio(mix_int16, mode="single", spk_emb=emb_b)

        # 3. Multi-Layer Adapter Decodes
        multi_dec_a = decode_audio(mix_int16, mode="multi", spk_emb=emb_a)
        multi_dec_b = decode_audio(mix_int16, mode="multi", spk_emb=emb_b)

        # Compute CERs
        r = {
            "pair": f"{sa} vs {sb}",
            "t_a": t_a,
            "t_b": t_b,
            "snr_db": snr_db,
            "clean_a": clean_dec_a,
            "clean_b": clean_dec_b,
            "uncond": uncond_dec,
            "cer_uncond_a": compute_cer(t_a, uncond_dec),
            "cer_uncond_b": compute_cer(t_b, uncond_dec),
            "single_a": single_dec_a,
            "cer_s_a_vs_a": compute_cer(t_a, single_dec_a),
            "cer_s_a_vs_b": compute_cer(t_b, single_dec_a),
            "single_b": single_dec_b,
            "cer_s_b_vs_a": compute_cer(t_a, single_dec_b),
            "cer_s_b_vs_b": compute_cer(t_b, single_dec_b),
            "multi_a": multi_dec_a,
            "cer_m_a_vs_a": compute_cer(t_a, multi_dec_a),
            "cer_m_a_vs_b": compute_cer(t_b, multi_dec_a),
            "multi_b": multi_dec_b,
            "cer_m_b_vs_a": compute_cer(t_a, multi_dec_b),
            "cer_m_b_vs_b": compute_cer(t_b, multi_dec_b),
        }
        results.append(r)

        print(f"\n--- [Case {idx}: {sa} & {sb}] (SNR: {snr_db:+.2f} dB - Exactly Balanced) ---")
        print(f"  • Ref A: \"{t_a}\" (Clean: \"{clean_dec_a}\")")
        print(f"  • Ref B: \"{t_b}\" (Clean: \"{clean_dec_b}\")")
        print(f"  [Unconditioned]:")
        print(f"      Decoded: \"{uncond_dec}\" | CER vs A: {r['cer_uncond_a']:.2f}, CER vs B: {r['cer_uncond_b']:.2f}")
        print(f"  [Single-Layer FiLM]:")
        print(f"      Cond on A: \"{single_dec_a}\" | CER vs A: {r['cer_s_a_vs_a']:.2f} (vs B: {r['cer_s_a_vs_b']:.2f})")
        print(f"      Cond on B: \"{single_dec_b}\" | CER vs B: {r['cer_s_b_vs_b']:.2f} (vs A: {r['cer_s_b_vs_a']:.2f})")
        print(f"  [Multi-Layer 16L FiLM]:")
        print(f"      Cond on A: \"{multi_dec_a}\" | CER vs A: {r['cer_m_a_vs_a']:.2f} (vs B: {r['cer_m_a_vs_b']:.2f})")
        print(f"      Cond on B: \"{multi_dec_b}\" | CER vs B: {r['cer_m_b_vs_b']:.2f} (vs A: {r['cer_m_b_vs_a']:.2f})")

    # Aggregate Statistics
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS SUMMARY (5 Pairs, 0.0 dB SNR)")
    print("=" * 80)
    avg_uncond_a = np.mean([r["cer_uncond_a"] for r in results])
    avg_uncond_b = np.mean([r["cer_uncond_b"] for r in results])

    avg_s_a = np.mean([r["cer_s_a_vs_a"] for r in results])
    avg_s_b = np.mean([r["cer_s_b_vs_b"] for r in results])

    avg_m_a = np.mean([r["cer_m_a_vs_a"] for r in results])
    avg_m_b = np.mean([r["cer_m_b_vs_b"] for r in results])

    print(f"Target Speaker A Performance:")
    print(f"  • Baseline Unconditioned CER: {avg_uncond_a:.3f}")
    print(f"  • Single-Layer FiLM CER:      {avg_s_a:.3f} ({'-' if avg_s_a <= avg_uncond_a else '+'}{abs(avg_s_a - avg_uncond_a):.3f})")
    print(f"  • Multi-Layer 16L FiLM CER:   {avg_m_a:.3f} ({'-' if avg_m_a <= avg_uncond_a else '+'}{abs(avg_m_a - avg_uncond_a):.3f})")

    print(f"\nTarget Speaker B Performance:")
    print(f"  • Baseline Unconditioned CER: {avg_uncond_b:.3f}")
    print(f"  • Single-Layer FiLM CER:      {avg_s_b:.3f} ({'-' if avg_s_b <= avg_uncond_b else '+'}{abs(avg_s_b - avg_uncond_b):.3f})")
    print(f"  • Multi-Layer 16L FiLM CER:   {avg_m_b:.3f} ({'-' if avg_m_b <= avg_uncond_b else '+'}{abs(avg_m_b - avg_uncond_b):.3f})")


if __name__ == "__main__":
    run_evaluation()
