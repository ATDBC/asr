#!/usr/bin/env python3
"""
Speaker Cross-Attention (SCA) Adapter with Robust Non-Collapsing Architecture.

Key Optimizations:
1. Robust Cross-Attention:
   - Removes the learnable scalar gate that causes shortcut collapse to zero.
   - Uses fixed residual scale (x + 0.5 * delta) with LayerNorm and standard Xavier initialization.
   - Audio representations X (B, T, 1280) actively query M speaker tokens S (B, M, 1280).
2. Guards against zero-length snippets (length >= 1600 samples / 0.1s).
3. 1,000 Steps Scale Training on 268 RAMC Conversations (515 Speakers, 56,000+ Turns).
4. Benchmark evaluation on the 5 balanced 0.0 dB SNR test pairs.
"""

import math
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
# 1. Speaker Cross-Attention (SCA) Architecture
# ---------------------------------------------------------------------------

class SpeakerCrossAttentionAdapter(nn.Module):
    """
    Multi-Head Speaker Cross-Attention Adapter.

    Audio representations X (B, T, d_model) query M speaker tokens S (B, M, d_model):
        S = MLP(spk_emb)
        Q = X W_q,  K = S W_k,  V = S W_v
        Attn = Softmax(Q K^T / sqrt(head_dim)) V
        X_out = LayerNorm(X + 0.5 * W_out(Attn))
    """
    def __init__(self, spk_dim: int = 192, d_model: int = 1280, n_heads: int = 8, num_tokens: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_tokens = num_tokens

        # Speaker projection into M tokens
        self.spk_proj = nn.Sequential(
            nn.Linear(spk_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, num_tokens * d_model),
            nn.LayerNorm(num_tokens * d_model)
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Standard Xavier initialization ensures active gradient updates
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, spk_emb: torch.Tensor | None = None) -> torch.Tensor:
        if spk_emb is None:
            return x
        if spk_emb.dim() == 1:
            spk_emb = spk_emb.unsqueeze(0)

        B, T, D = x.shape
        spk_emb = F.normalize(spk_emb.to(dtype=x.dtype, device=x.device), p=2, dim=-1)

        # Generate M speaker acoustic tokens: (B, num_tokens, D)
        spk_tokens = self.spk_proj(spk_emb).view(B, self.num_tokens, D)

        # Multi-Head Projections
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(spk_tokens).view(B, self.num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(spk_tokens).view(B, self.num_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (B, H, T, num_tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)

        context = context.transpose(1, 2).contiguous().view(B, T, D)
        delta = self.out_proj(context)

        return self.norm(x + 0.5 * delta)


def patch_conformer_encoder_cross_attn(encoder, adapter: SpeakerCrossAttentionAdapter):
    """Injects Speaker Cross-Attention Adapter at the Conformer Encoder output."""
    encoder.speaker_adapter = adapter

    def forward_with_adapter(padded_input, input_lengths, pad=True, spk_emb=None):
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

        enc_output = encoder.dropout(embed_output)
        pos_emb = encoder.dropout(encoder.positional_encoding(embed_output))

        for enc_layer in encoder.layer_stack:
            enc_output = enc_layer(
                enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
            )

        # Apply Speaker Cross-Attention at Encoder Output
        if getattr(encoder, "speaker_adapter", None) is not None:
            enc_output = encoder.speaker_adapter(enc_output, spk_emb)

        return enc_output, input_lengths, src_mask

    encoder.forward = forward_with_adapter
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
# 3. Audio & Dataset Loader
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


def load_ramc_dataset():
    corpus_dir = PROJECT_ROOT / "artifacts" / "ramc_full" / "MDT2021S003"
    txt_dir = corpus_dir / "TXT"
    wav_dir = corpus_dir / "WAV"

    wav_files = sorted(list(wav_dir.glob("*.wav")))
    segments_by_spk = {}
    clean_re = re.compile(r"\[([0-9.]+),([0-9.]+)\]\s+([A-Za-z0-9_]+)\s+[^	\n]+\s+(.*)")

    total_turns = 0
    loaded_files = 0
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
        loaded_files += 1

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
                        total_turns += 1

    filtered = {k: v for k, v in segments_by_spk.items() if len(v) >= 6}
    return filtered, loaded_files, total_turns


# ---------------------------------------------------------------------------
# 4. Training and Evaluation Pipeline
# ---------------------------------------------------------------------------

def run_cross_attn_training(num_steps: int = 1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Speaker Cross-Attention (SCA) Non-Collapsing Training on {device} ===")

    # 1. Load FireRed ASR Model
    print("  • Loading FireRedASR2-AED...")
    asr_system = FireRedAsr2.from_pretrained("aed", str(MODEL_DIR / "fireredasr2s" / "FireRedASR2-AED"))
    model = asr_system.model.to(device)
    model.eval()

    # Freeze entire backbone
    for p in model.parameters():
        p.requires_grad = False

    # 2. Instantiate and Patch Cross-Attention Adapter
    adapter = SpeakerCrossAttentionAdapter(
        spk_dim=192,
        d_model=model.encoder.odim,
        n_heads=8,
        num_tokens=4
    ).to(device)
    patch_conformer_encoder_cross_attn(model.encoder, adapter)

    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  • Frozen Backbone Parameters: {frozen_params:,}")
    print(f"  • Cross-Attention Adapter Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")

    # 3. Load CAM++ Voiceprint Extractor
    print("  • Loading CAM++ 3D-Speaker...")
    vp_extractor = VoiceprintExtractor(device)

    # 4. Load RAMC Dataset
    print("  • Loading RAMC multi-speaker dataset...")
    spk_data, loaded_files, total_turns = load_ramc_dataset()
    speakers = list(spk_data.keys())
    print(f"  • Loaded {loaded_files} conversations, {len(speakers)} qualified speakers, {total_turns} clean turns.")

    # Pre-compute speaker embeddings
    print("  • Pre-computing speaker voiceprints...")
    spk_embeddings = {}
    for spk in speakers:
        utts = spk_data[spk]
        embs = [vp_extractor.extract(u[0].astype(np.float32) / 32767.0) for u in utts[:3]]
        spk_embeddings[spk] = F.normalize(torch.stack(embs).mean(dim=0), p=2, dim=-1)

    # 5. Fixed 5 Test Pairs (0.0 dB SNR Balanced Benchmark)
    test_pairs = [
        (speakers[0], speakers[1]),
        (speakers[2], speakers[3]),
        (speakers[4], speakers[5]),
        (speakers[6], speakers[7]),
        (speakers[8], speakers[9]),
    ]
    test_cases = []
    for sa, sb in test_pairs:
        test_cases.append({
            "spk_a": sa,
            "utt_a": spk_data[sa][-1],
            "spk_b": sb,
            "utt_b": spk_data[sb][-1],
        })

    # 6. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=3e-5)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"\n=== Training Speaker Cross-Attention Adapter ({num_steps} Steps) ===")
    start_time = time.time()
    loss_history = []

    for step in range(1, num_steps + 1):
        is_clean_anchor = (step % 4 == 0)

        if is_clean_anchor:
            s_clean = np.random.choice(speakers)
            pool = spk_data[s_clean][:-1]
            wav_c, text_c = pool[np.random.randint(len(pool))]
            if len(wav_c) < 16000:
                continue
            _, ids_c = asr_system.tokenizer.tokenize(text_c)
            if len(ids_c) == 0:
                continue

            feats_pad, lengths, _, _, _ = asr_system.feat_extractor([(16000, wav_c)], ["clean"])
            feats_pad, lengths = feats_pad.to(device), lengths.to(device)
            target = torch.tensor(ids_c, dtype=torch.long, device=device)
            target_len = torch.tensor([len(ids_c)], dtype=torch.long, device=device)

            emb = spk_embeddings[s_clean].unsqueeze(0)
            enc_out, enc_len, _ = model.encoder(feats_pad, lengths, spk_emb=emb)
            logits = model.ctc(enc_out).transpose(0, 1)
            total_loss = ctc_loss_fn(logits, target, enc_len, target_len)
        else:
            s_a, s_b = np.random.choice(speakers, 2, replace=False)
            pool_a = spk_data[s_a][:-1]
            pool_b = spk_data[s_b][:-1]
            utt_a = pool_a[np.random.randint(len(pool_a))]
            utt_b = pool_b[np.random.randint(len(pool_b))]

            wav_a, text_a = utt_a
            wav_b, text_b = utt_b

            length = min(len(wav_a), len(wav_b))
            if length < 16000:
                continue
            w_a = wav_a[:length].astype(np.float32)
            w_b = wav_b[:length].astype(np.float32)

            rms_a = np.sqrt(np.mean(w_a ** 2)) + 1e-8
            rms_b = np.sqrt(np.mean(w_b ** 2)) + 1e-8
            snr_db = np.random.uniform(-4.0, 4.0)
            target_rms_b = rms_a * (10.0 ** (-snr_db / 20.0))
            w_b = w_b * (target_rms_b / rms_b)

            mix = w_a + w_b
            max_val = np.max(np.abs(mix))
            if max_val > 32000.0:
                mix = mix / max_val * 32000.0
            mix_int16 = mix.astype(np.int16)

            _, ids_a = asr_system.tokenizer.tokenize(text_a)
            _, ids_b = asr_system.tokenizer.tokenize(text_b)
            if len(ids_a) == 0 or len(ids_b) == 0:
                continue

            feats_pad, lengths, _, _, _ = asr_system.feat_extractor([(16000, mix_int16)], ["mix"])
            feats_pad, lengths = feats_pad.to(device), lengths.to(device)

            # Pass 1: Conditioned on Speaker A
            emb_a = spk_embeddings[s_a].unsqueeze(0)
            enc_out_a, enc_len_a, _ = model.encoder(feats_pad, lengths, spk_emb=emb_a)
            logits_a = model.ctc(enc_out_a).transpose(0, 1)
            target_a = torch.tensor(ids_a, dtype=torch.long, device=device)
            target_len_a = torch.tensor([len(ids_a)], dtype=torch.long, device=device)
            loss_a = ctc_loss_fn(logits_a, target_a, enc_len_a, target_len_a)

            # Pass 2: Conditioned on Speaker B
            emb_b = spk_embeddings[s_b].unsqueeze(0)
            enc_out_b, enc_len_b, _ = model.encoder(feats_pad, lengths, spk_emb=emb_b)
            logits_b = model.ctc(enc_out_b).transpose(0, 1)
            target_b = torch.tensor(ids_b, dtype=torch.long, device=device)
            target_len_b = torch.tensor([len(ids_b)], dtype=torch.long, device=device)
            loss_b = ctc_loss_fn(logits_b, target_b, enc_len_b, target_len_b)

            total_loss = (loss_a + loss_b) / 2.0

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=5.0)
        optimizer.step()
        lr_scheduler.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            lr_cur = optimizer.param_groups[0]["lr"]
            tag = "Clean Anchor" if is_clean_anchor else "Overlap Dual"
            print(f"  • Step {step:4d}/{num_steps} | Loss: {loss_val:.4f} | [{tag}] | LR: {lr_cur:.2e} | Time: {elapsed:.1f}s")

    print(f"\n  • Initial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")

    # 7. Save Checkpoint
    out_ckpt = PROJECT_ROOT / "outputs" / "speaker_cross_attn_robust.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), str(out_ckpt))
    print(f"  • Checkpoint saved: {out_ckpt}")

    # ---------------------------------------------------------------------------
    # 8. Evaluate under Balanced (0.0 dB SNR) Benchmark
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATING SPEAKER CROSS-ATTENTION UNDER BALANCED (0.0 dB SNR) BENCHMARK")
    print("=" * 80)
    adapter.eval()

    def decode_audio(wav_int16, spk_emb=None):
        feats, lens, _, _, _ = asr_system.feat_extractor([(16000, wav_int16)], ["eval"])
        feats, lens = feats.to(device), lens.to(device)
        with torch.no_grad():
            out, _, mask = model.encoder(feats, lens, spk_emb=spk_emb)
            hyp = model.decoder.batch_beam_search(out, mask, beam_size=1)
            text = asr_system.tokenizer.detokenize(hyp[0][0]["yseq"].tolist())
        return text

    results = []
    for idx, case in enumerate(test_cases, 1):
        sa, sb = case["spk_a"], case["spk_b"]
        w_a, t_a = case["utt_a"]
        w_b, t_b = case["utt_b"]

        mix_int16, _, _ = balance_and_mix(w_a, w_b, target_rms=3000.0)

        dec_uncond = decode_audio(mix_int16, spk_emb=None)

        emb_a = spk_embeddings[sa].unsqueeze(0)
        dec_cond_a = decode_audio(mix_int16, spk_emb=emb_a)

        emb_b = spk_embeddings[sb].unsqueeze(0)
        dec_cond_b = decode_audio(mix_int16, spk_emb=emb_b)

        cer_u_a = compute_cer(t_a, dec_uncond)
        cer_u_b = compute_cer(t_b, dec_uncond)
        cer_c_a_vs_a = compute_cer(t_a, dec_cond_a)
        cer_c_b_vs_b = compute_cer(t_b, dec_cond_b)

        results.append({
            "cer_u_a": cer_u_a,
            "cer_u_b": cer_u_b,
            "cer_a": cer_c_a_vs_a,
            "cer_b": cer_c_b_vs_b,
        })

        print(f"\n--- [Case {idx}: {sa} vs {sb}] (0.00 dB SNR) ---")
        print(f"  • Ref A: \"{t_a}\"")
        print(f"  • Ref B: \"{t_b}\"")
        print(f"  [Unconditioned]: \"{dec_uncond}\" | CER vs A: {cer_u_a:.2f}, CER vs B: {cer_u_b:.2f}")
        print(f"  [Cond on A]:     \"{dec_cond_a}\" | CER vs A: {cer_c_a_vs_a:.2f}")
        print(f"  [Cond on B]:     \"{dec_cond_b}\" | CER vs B: {cer_c_b_vs_b:.2f}")

    avg_u_a = np.mean([r["cer_u_a"] for r in results])
    avg_u_b = np.mean([r["cer_u_b"] for r in results])
    avg_c_a = np.mean([r["cer_a"] for r in results])
    avg_c_b = np.mean([r["cer_b"] for r in results])

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS SUMMARY (Cross-Attention Adapter vs Baseline)")
    print("=" * 80)
    print(f"Speaker A: Baseline CER: {avg_u_a:.3f} -> Cross-Attention CER: {avg_c_a:.3f} ({'-' if avg_c_a <= avg_u_a else '+'}{abs(avg_c_a - avg_u_a):.3f})")
    print(f"Speaker B: Baseline CER: {avg_u_b:.3f} -> Cross-Attention CER: {avg_c_b:.3f} ({'-' if avg_c_b <= avg_u_b else '+'}{abs(avg_c_b - avg_u_b):.3f})")


if __name__ == "__main__":
    run_cross_attn_training(num_steps=1000)
