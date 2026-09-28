#!/usr/bin/env python3
"""Scheme 1: Multi-Layer Deep-FiLM Adapter for Conformer Speaker-Conditioned ASR.

1. Injects 16 independent FiLM modulation layers into the 16 Conformer blocks.
2. Caches multi-speaker conversational recordings from RAMC.
3. Pre-computes reference speaker voiceprints via 3D-Speaker (CAM++).
4. Trains via dual-pass CTC Loss on dynamically synthesized multi-talker overlaps.
5. Evaluates beam-search decoding on unseen test overlap mixtures.
"""

import os
import sys
import time
import re
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
# 1. Architecture: MultiLayerFiLMAdapter & Conformer Patching
# ---------------------------------------------------------------------------

class MultiLayerFiLMAdapter(nn.Module):
    """Deep FiLM modulation across all Conformer encoder layers."""

    def __init__(self, num_layers: int = 16, spk_dim: int = 192, d_model: int = 1280):
        super().__init__()
        self.num_layers = num_layers
        self.spk_dim = spk_dim
        self.d_model = d_model

        # 16 distinct projection layers (one per Conformer block)
        self.layers = nn.ModuleList([
            nn.Linear(spk_dim, 2 * d_model) for _ in range(num_layers)
        ])

        # Strict zero-initialization: starts as exact bit-exact identity
        for layer in self.layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

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


def patch_conformer_encoder_multilayer(encoder, adapter: MultiLayerFiLMAdapter):
    """Injects layer-wise FiLM adapter into ConformerEncoder.forward."""
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

        for idx, enc_layer in enumerate(encoder.layer_stack):
            enc_output = enc_layer(
                enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
            )
            # Apply layer-specific FiLM modulation
            if getattr(encoder, "speaker_adapter", None) is not None:
                enc_output = encoder.speaker_adapter(idx, enc_output, spk_emb)

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
# 3. Fast Data Loader for RAMC
# ---------------------------------------------------------------------------

def load_ramc_dataset(max_files=16):
    corpus_dir = PROJECT_ROOT / "artifacts" / "cost_model_train_corpus" / "ramc-v5-wave1" / "extracted" / "MDT2021S003"
    txt_dir = corpus_dir / "TXT"
    wav_dir = corpus_dir / "WAV"

    txt_files = sorted(list(txt_dir.glob("*.txt")))[:max_files]
    segments_by_spk = {}
    clean_re = re.compile(r"\[([0-9.]+),([0-9.]+)\]\s+([A-Za-z0-9_]+)\s+[^	\n]+\s+(.*)")

    loaded_count = 0
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
                if 1.5 <= dur <= 5.5 and len(text) >= 4:
                    start_sample = int(round(start_s * sr))
                    end_sample = int(round(end_s * sr))
                    chunk = audio_int16[start_sample:end_sample]
                    if spk_id not in segments_by_spk:
                        segments_by_spk[spk_id] = []
                    segments_by_spk[spk_id].append((chunk, text))
                    loaded_count += 1

    # Keep speakers with at least 5 utterances
    filtered = {k: v for k, v in segments_by_spk.items() if len(v) >= 5}
    return filtered


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


# ---------------------------------------------------------------------------
# 4. Main Training and Evaluation Routine
# ---------------------------------------------------------------------------

def run_multilayer_adapter_training(num_steps: int = 300):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Initializing Multi-Layer FiLM Adapter Training on {device} ===")

    # 1. Load FireRed ASR Model
    print("  • Loading FireRedASR2-AED...")
    asr_system = FireRedAsr2.from_pretrained("aed", str(MODEL_DIR / "fireredasr2s" / "FireRedASR2-AED"))
    model = asr_system.model.to(device)
    model.eval()

    # Freeze entire backbone
    for p in model.parameters():
        p.requires_grad = False

    # Initialize 16-Layer FiLM Adapter
    adapter = MultiLayerFiLMAdapter(
        num_layers=model.encoder.n_layers_enc if hasattr(model.encoder, "n_layers_enc") else 16,
        spk_dim=192,
        d_model=model.encoder.odim
    ).to(device)
    patch_conformer_encoder_multilayer(model.encoder, adapter)

    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  • Frozen Backbone Parameters: {frozen_params:,}")
    print(f"  • 16-Layer FiLM Adapter Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")

    # 2. Load CAM++ Voiceprint Extractor
    print("  • Loading CAM++ 3D-Speaker...")
    vp_extractor = VoiceprintExtractor(device)

    # 3. Load RAMC Dataset
    print("  • Loading RAMC multi-speaker dataset (16 conversations)...")
    spk_data = load_ramc_dataset(max_files=16)
    speakers = list(spk_data.keys())
    total_utts = sum(len(v) for v in spk_data.values())
    print(f"  • Loaded {len(speakers)} speakers with {total_utts} clean utterances.")

    # Compute global reference voiceprint for each speaker
    print("  • Pre-computing reference voiceprints for each speaker...")
    spk_embeddings = {}
    for spk, utts in spk_data.items():
        sorted_utts = sorted(utts, key=lambda x: len(x[0]), reverse=True)
        # Average top 3 longest utterances
        embs = [vp_extractor.extract(u[0].astype(np.float32) / 32767.0) for u in sorted_utts[:3]]
        spk_embeddings[spk] = F.normalize(torch.stack(embs).mean(dim=0), p=2, dim=-1)

    # Create fixed test pairs from held-out utterances
    test_spk_pairs = [
        (speakers[0], speakers[1]),
        (speakers[2], speakers[3]),
        (speakers[4], speakers[5]),
    ]
    test_cases = []
    for sa, sb in test_spk_pairs:
        test_cases.append({
            "spk_a": sa,
            "utt_a": spk_data[sa][-1],
            "spk_b": sb,
            "utt_b": spk_data[sb][-1],
        })

    # 4. Training Setup
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=5e-5)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"\n=== Training 16-Layer FiLM Adapter on Synthetic Overlaps ({num_steps} Steps) ===")
    start_time = time.time()
    loss_history = []

    for step in range(1, num_steps + 1):
        # Pick 2 distinct speakers at random
        s_a, s_b = np.random.choice(speakers, 2, replace=False)
        # Avoid held-out test utterances
        pool_a = spk_data[s_a][:-1]
        pool_b = spk_data[s_b][:-1]
        utt_a = pool_a[np.random.randint(len(pool_a))]
        utt_b = pool_b[np.random.randint(len(pool_b))]

        wav_a, text_a = utt_a
        wav_b, text_b = utt_b

        # Synchronize overlap lengths
        length = min(len(wav_a), len(wav_b))
        w_a = wav_a[:length].astype(np.float32)
        w_b = wav_b[:length].astype(np.float32)

        # Contrastive SNR sampling: -4 dB to +4 dB
        snr_db = np.random.uniform(-4.0, 4.0)
        scale_b = 10 ** (-snr_db / 20.0)
        mix = w_a + scale_b * w_b
        max_val = np.max(np.abs(mix))
        if max_val > 32767.0:
            mix = mix / max_val * 32700.0
        mix_int16 = mix.astype(np.int16)

        # Tokenize targets
        _, ids_a = asr_system.tokenizer.tokenize(text_a)
        _, ids_b = asr_system.tokenizer.tokenize(text_b)
        if len(ids_a) == 0 or len(ids_b) == 0:
            continue

        # Extract acoustic features
        feats_pad, lengths, _, _, _ = asr_system.feat_extractor([(16000, mix_int16)], ["utt_mix"])
        feats_pad = feats_pad.to(device)
        lengths = lengths.to(device)

        # Forward pass 1: Conditioned on Speaker A
        emb_a = spk_embeddings[s_a].unsqueeze(0)
        enc_out_a, enc_len_a, _ = model.encoder(feats_pad, lengths, spk_emb=emb_a)
        logits_a = model.ctc(enc_out_a).transpose(0, 1)
        target_a = torch.tensor(ids_a, dtype=torch.long, device=device)
        target_len_a = torch.tensor([len(ids_a)], dtype=torch.long, device=device)
        loss_a = ctc_loss_fn(logits_a, target_a, enc_len_a, target_len_a)

        # Forward pass 2: Conditioned on Speaker B
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

        if step % 20 == 0 or step == 1:
            elapsed = time.time() - start_time
            lr_cur = optimizer.param_groups[0]["lr"]
            print(f"  • Step {step:3d}/{num_steps} | Loss: {loss_val:.4f} | LR: {lr_cur:.2e} | Time: {elapsed:.1f}s")

    print(f"\n  • Initial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")

    # ---------------------------------------------------------------------------
    # 5. Evaluate Steering Across Held-out Test Overlaps
    # ---------------------------------------------------------------------------
    print("\n=== Evaluating Speaker Steering on 3 Held-out Overlap Test Cases ===")
    adapter.eval()

    total_cer_uncond_a = []
    total_cer_uncond_b = []
    total_cer_cond_a = []
    total_cer_cond_b = []

    for i, case in enumerate(test_cases, 1):
        sa, sb = case["spk_a"], case["spk_b"]
        w_a, t_a = case["utt_a"]
        w_b, t_b = case["utt_b"]
        t_len = min(len(w_a), len(w_b))
        test_mix = ((w_a[:t_len].astype(np.float32) + w_b[:t_len].astype(np.float32)) / 2.0).astype(np.int16)

        feats, lens, _, _, _ = asr_system.feat_extractor([(16000, test_mix)], ["test_mix"])
        feats, lens = feats.to(device), lens.to(device)

        with torch.no_grad():
            # 1. Unconditioned
            out_u, _, mask_u = model.encoder(feats, lens, spk_emb=None)
            hyp_u = model.decoder.batch_beam_search(out_u, mask_u, beam_size=1)
            text_u = asr_system.tokenizer.detokenize(hyp_u[0][0]["yseq"].tolist())

            # 2. Conditioned on A
            out_a, _, mask_a = model.encoder(feats, lens, spk_emb=spk_embeddings[sa].unsqueeze(0))
            hyp_a = model.decoder.batch_beam_search(out_a, mask_a, beam_size=1)
            text_a_dec = asr_system.tokenizer.detokenize(hyp_a[0][0]["yseq"].tolist())

            # 3. Conditioned on B
            out_b, _, mask_b = model.encoder(feats, lens, spk_emb=spk_embeddings[sb].unsqueeze(0))
            hyp_b = model.decoder.batch_beam_search(out_b, mask_b, beam_size=1)
            text_b_dec = asr_system.tokenizer.detokenize(hyp_b[0][0]["yseq"].tolist())

        cer_u_a = compute_cer(t_a, text_u)
        cer_u_b = compute_cer(t_b, text_u)
        cer_c_a = compute_cer(t_a, text_a_dec)
        cer_c_b = compute_cer(t_b, text_b_dec)

        total_cer_uncond_a.append(cer_u_a)
        total_cer_uncond_b.append(cer_u_b)
        total_cer_cond_a.append(cer_c_a)
        total_cer_cond_b.append(cer_c_b)

        print(f"\n--- [Test Case {i}] ---")
        print(f"  • Reference A ({sa}): \"{t_a}\"")
        print(f"  • Reference B ({sb}): \"{t_b}\"")
        print(f"  [Unconditioned]:     \"{text_u}\" (CER vs A: {cer_u_a:.2f}, vs B: {cer_u_b:.2f})")
        print(f"  [Conditioned on A]:  \"{text_a_dec}\" (CER vs A: {cer_c_a:.2f})")
        print(f"  [Conditioned on B]:  \"{text_b_dec}\" (CER vs B: {cer_c_b:.2f})")

    avg_u_a = np.mean(total_cer_uncond_a)
    avg_u_b = np.mean(total_cer_uncond_b)
    avg_c_a = np.mean(total_cer_cond_a)
    avg_c_b = np.mean(total_cer_cond_b)

    print("\n=== Aggregate Benchmark Metrics ===")
    print(f"  • Speaker A: Unconditioned CER: {avg_u_a:.3f} -> Conditioned CER: {avg_c_a:.3f} ({'-' if avg_c_a <= avg_u_a else '+'}{abs(avg_c_a - avg_u_a):.3f})")
    print(f"  • Speaker B: Unconditioned CER: {avg_u_b:.3f} -> Conditioned CER: {avg_c_b:.3f} ({'-' if avg_c_b <= avg_u_b else '+'}{abs(avg_c_b - avg_u_b):.3f})")

    # 6. Save Checkpoint
    out_ckpt = PROJECT_ROOT / "outputs" / "multilayer_film_adapter.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), str(out_ckpt))
    print(f"\n  • Checkpoint saved: {out_ckpt}")


if __name__ == "__main__":
    run_multilayer_adapter_training(num_steps=300)
