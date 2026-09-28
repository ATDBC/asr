#!/usr/bin/env python3
"""Mini-Training Spike: Speaker-Conditioned FiLM Adapter for FireRedASR2-AED.

1. Extracts clean speaker segments and voiceprint embeddings from RAMC recordings.
2. Synthesizes on-the-fly multi-talker overlapping audio pairs.
3. Fine-tunes the lightweight SpeakerFiLMAdapter using CTC Loss on GPU 0.
4. Evaluates whether the adapter steers ASR attention to the target speaker.
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
# 1. Architecture: SpeakerFiLMAdapter & Patching
# ---------------------------------------------------------------------------

class SpeakerFiLMAdapter(nn.Module):
    def __init__(self, spk_dim: int = 192, d_model: int = 1280):
        super().__init__()
        self.spk_dim = spk_dim
        self.d_model = d_model
        # Linear layer projecting 192 -> 2 * 1280 (scale gamma, shift beta)
        self.proj = nn.Linear(spk_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

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


def patch_conformer_encoder(encoder, adapter: SpeakerFiLMAdapter):
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
        if getattr(encoder, "speaker_adapter", None) is not None:
            embed_output = encoder.speaker_adapter(embed_output, spk_emb)

        enc_output = encoder.dropout(embed_output)
        pos_emb = encoder.dropout(encoder.positional_encoding(embed_output))

        for enc_layer in encoder.layer_stack:
            enc_output = enc_layer(
                enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
            )
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
        # wav_float: (samples,) 16kHz in [-1.0, 1.0]
        wav_t = torch.from_numpy(wav_float).unsqueeze(0).to(self.device)
        feat = self.feature_extractor(wav_t).unsqueeze(0)  # (1, T, 80)
        emb = self.model(feat)  # (1, 192)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.squeeze(0)  # (192,)


# ---------------------------------------------------------------------------
# 3. Data Loading & Parsing from RAMC
# ---------------------------------------------------------------------------

def load_ramc_segments(max_files=4):
    corpus_dir = PROJECT_ROOT / "artifacts" / "cost_model_train_corpus" / "ramc-v5-wave1" / "extracted" / "MDT2021S003"
    txt_dir = corpus_dir / "TXT"
    wav_dir = corpus_dir / "WAV"

    txt_files = sorted(list(txt_dir.glob("*.txt")))[:max_files]
    segments_by_spk = {}  # spk_id -> list of (wav_data_int16, text)

    clean_re = re.compile(r"\[([0-9.]+),([0-9.]+)\]\s+([A-Za-z0-9_]+)\s+[^	\n]+\s+(.*)")

    for txt_file in txt_files:
        wav_file = wav_dir / (txt_file.stem + ".wav")
        if not wav_file.is_file():
            continue
        audio_data, sr = sf.read(str(wav_file), dtype="float32")
        if sr != 16000:
            continue
        # Scale to Kaldi PCM 16-bit integer scale
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
                # Keep utterances between 1.5s and 6.0s with clean text
                text = re.sub(r"\[\+?\]|\(.*?\)|\<.*?\>", "", text).strip()
                if 1.5 <= dur <= 6.0 and len(text) >= 4:
                    start_sample = int(round(start_s * sr))
                    end_sample = int(round(end_s * sr))
                    chunk = audio_int16[start_sample:end_sample]
                    if spk_id not in segments_by_spk:
                        segments_by_spk[spk_id] = []
                    segments_by_spk[spk_id].append((chunk, text))

    filtered = {k: v for k, v in segments_by_spk.items() if len(v) >= 4}
    return filtered


# ---------------------------------------------------------------------------
# 4. Main Training and Evaluation Routine
# ---------------------------------------------------------------------------

def run_adapter_spike():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Initializing Speaker-Conditioned Adapter Spike on {device} ===")

    # 1. Load ASR Model
    print("  • Loading FireRedASR2-AED...")
    asr_system = FireRedAsr2.from_pretrained("aed", str(MODEL_DIR / "fireredasr2s" / "FireRedASR2-AED"))
    model = asr_system.model.to(device)
    model.eval()

    # Freeze all backbone weights
    for p in model.parameters():
        p.requires_grad = False

    # Initialize Adapter
    adapter = SpeakerFiLMAdapter(spk_dim=192, d_model=model.encoder.odim).to(device)
    patch_conformer_encoder(model.encoder, adapter)

    # 2. Load 3D-Speaker Voiceprint Extractor
    print("  • Loading CAM++ 3D-Speaker...")
    vp_extractor = VoiceprintExtractor(device)

    # 3. Load Dataset
    print("  • Loading RAMC conversation segments...")
    spk_data = load_ramc_segments(max_files=4)
    speakers = list(spk_data.keys())
    print(f"  • Extracted {len(speakers)} speakers with {sum(len(v) for v in spk_data.values())} utterances.")

    # Compute reference voiceprint for each speaker
    spk_embeddings = {}
    for spk, utts in spk_data.items():
        sorted_utts = sorted(utts, key=lambda x: len(x[0]), reverse=True)
        # Convert int16 chunk to float32 [-1, 1] for CAM++
        embs = [vp_extractor.extract(u[0].astype(np.float32) / 32767.0) for u in sorted_utts[:2]]
        avg_emb = F.normalize(torch.stack(embs).mean(dim=0), p=2, dim=-1)
        spk_embeddings[spk] = avg_emb

    # Split train and test
    test_spk_a, test_spk_b = speakers[0], speakers[1]
    test_pair_a = spk_data[test_spk_a][-1]
    test_pair_b = spk_data[test_spk_b][-1]

    train_speakers = speakers

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    print("\n=== Training SpeakerFiLMAdapter on Synthetic Overlaps (100 Steps) ===")
    start_time = time.time()
    num_steps = 100
    loss_history = []

    for step in range(1, num_steps + 1):
        # Pick 2 distinct speakers
        s_a, s_b = np.random.choice(train_speakers, 2, replace=False)
        utt_a = spk_data[s_a][np.random.randint(len(spk_data[s_a]))]
        utt_b = spk_data[s_b][np.random.randint(len(spk_data[s_b]))]

        wav_a, text_a = utt_a
        wav_b, text_b = utt_b

        # Mix: crop/pad to match length
        length = min(len(wav_a), len(wav_b))
        w_a = wav_a[:length].astype(np.float32)
        w_b = wav_b[:length].astype(np.float32)

        # Random SNR between -2 dB and +2 dB
        snr_db = np.random.uniform(-2.0, 2.0)
        scale_b = 10 ** (-snr_db / 20.0)
        mix = w_a + scale_b * w_b
        max_val = np.max(np.abs(mix))
        if max_val > 32767.0:
            mix = mix / max_val * 32700.0
        mix_int16 = mix.astype(np.int16)

        # Target token IDs
        tokens_a, ids_a = asr_system.tokenizer.tokenize(text_a)
        tokens_b, ids_b = asr_system.tokenizer.tokenize(text_b)

        if len(ids_a) == 0 or len(ids_b) == 0:
            continue

        # Extract features for mixed audio
        feats_pad, lengths, _, _, _ = asr_system.feat_extractor([(16000, mix_int16)], ["utt_mix"])
        feats_pad = feats_pad.to(device)
        lengths = lengths.to(device)

        # Forward with Speaker A condition
        emb_a = spk_embeddings[s_a].unsqueeze(0)
        enc_out_a, enc_len_a, _ = model.encoder(feats_pad, lengths, spk_emb=emb_a)
        logits_a = model.ctc(enc_out_a).transpose(0, 1)  # (T, N, C)

        target_a = torch.tensor(ids_a, dtype=torch.long, device=device)
        target_len_a = torch.tensor([len(ids_a)], dtype=torch.long, device=device)
        loss_a = ctc_loss_fn(logits_a, target_a, enc_len_a, target_len_a)

        # Forward with Speaker B condition
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

        loss_val = total_loss.item()
        loss_history.append(loss_val)

        if step % 10 == 0 or step == 1:
            print(f"  • Step {step:3d}/100 | Loss: {loss_val:.4f} | Time: {time.time() - start_time:.1f}s")

    print(f"\n  • Initial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")

    # ---------------------------------------------------------------------------
    # 5. Evaluate Steering on Held-out Overlap Test Case
    # ---------------------------------------------------------------------------
    print("\n=== Evaluating Speaker Steering on Test Overlap ===")
    wav_ta, text_ta = test_pair_a
    wav_tb, text_tb = test_pair_b
    t_len = min(len(wav_ta), len(wav_tb))
    test_mix = (wav_ta[:t_len].astype(np.float32) + wav_tb[:t_len].astype(np.float32)) / 2.0
    test_mix_int16 = test_mix.astype(np.int16)

    print(f"  • Ground Truth A ({test_spk_a}): \"{text_ta}\"")
    print(f"  • Ground Truth B ({test_spk_b}): \"{text_tb}\"")

    feats_test, lens_test, _, _, _ = asr_system.feat_extractor([(16000, test_mix_int16)], ["test_mix"])
    feats_test = feats_test.to(device)
    lens_test = lens_test.to(device)

    # 1. Unconditioned baseline (spk_emb = None)
    with torch.no_grad():
        out_uncond, _, mask_uncond = model.encoder(feats_test, lens_test, spk_emb=None)
        hyp_uncond = model.decoder.batch_beam_search(out_uncond, mask_uncond, beam_size=1)
        text_uncond = asr_system.tokenizer.detokenize(hyp_uncond[0][0]["yseq"].tolist())
    print(f"\n  [Unconditioned FireRed]: \"{text_uncond}\"")

    # 2. Conditioned on Speaker A
    with torch.no_grad():
        out_ca, _, mask_ca = model.encoder(feats_test, lens_test, spk_emb=spk_embeddings[test_spk_a].unsqueeze(0))
        hyp_ca = model.decoder.batch_beam_search(out_ca, mask_ca, beam_size=1)
        text_ca = asr_system.tokenizer.detokenize(hyp_ca[0][0]["yseq"].tolist())
    print(f"  [Conditioned on Speaker A]: \"{text_ca}\"")

    # 3. Conditioned on Speaker B
    with torch.no_grad():
        out_cb, _, mask_cb = model.encoder(feats_test, lens_test, spk_emb=spk_embeddings[test_spk_b].unsqueeze(0))
        hyp_cb = model.decoder.batch_beam_search(out_cb, mask_cb, beam_size=1)
        text_cb = asr_system.tokenizer.detokenize(hyp_cb[0][0]["yseq"].tolist())
    print(f"  [Conditioned on Speaker B]: \"{text_cb}\"")

    # Save adapter checkpoint
    save_path = PROJECT_ROOT / "outputs" / "speaker_film_adapter_poc.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), str(save_path))
    print(f"\n  • Saved Adapter checkpoint to {save_path}")


if __name__ == "__main__":
    run_adapter_spike()
