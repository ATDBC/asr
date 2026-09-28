#!/usr/bin/env python3
"""Spike: Speaker-conditioned FiLM Adapter for FireRedASR2-AED."""

import os
import sys
from pathlib import Path
import numpy as np
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


class SpeakerFiLMAdapter(nn.Module):
    """Feature-wise Linear Modulation (FiLM) adapter for Conformer acoustic features."""

    def __init__(self, spk_dim: int = 192, d_model: int = 1280):
        super().__init__()
        self.spk_dim = spk_dim
        self.d_model = d_model
        # Linear layer projecting 192 -> 2 * 1280 (scale gamma, shift beta)
        self.proj = nn.Linear(spk_dim, 2 * d_model)
        # Zero-initialize weight and bias for exact identity at step 0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, spk_emb: torch.Tensor | None = None) -> torch.Tensor:
        if spk_emb is None:
            return x
        if spk_emb.dim() == 1:
            spk_emb = spk_emb.unsqueeze(0)
        # Normalize embedding to unit sphere
        spk_emb = F.normalize(spk_emb.to(dtype=x.dtype, device=x.device), p=2, dim=-1)
        scale, shift = self.proj(spk_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return (1.0 + scale) * x + shift


def patch_conformer_encoder(encoder, adapter: SpeakerFiLMAdapter):
    """Non-invasively inject adapter into ConformerEncoder.forward."""
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

        # Injected FiLM Adapter
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


def test_identity_and_gradients():
    print("=== [1] Testing Adapter Architecture & Identity ===")
    adapter = SpeakerFiLMAdapter(spk_dim=192, d_model=1280)

    # 1. Identity test
    dummy_x = torch.randn(2, 50, 1280)
    dummy_spk = torch.randn(2, 192)
    out_identity = adapter(dummy_x, dummy_spk)
    max_diff = torch.max(torch.abs(out_identity - dummy_x)).item()
    print(f"  • Zero-init identity max abs diff: {max_diff:.8e} (expected: 0.0)")
    assert max_diff == 0.0, "Zero-init FiLM adapter must be bit-exact identity!"

    # 2. Gradient flow test
    loss = out_identity.sum()
    loss.backward()
    assert adapter.proj.weight.grad is not None, "Adapter must receive gradients!"
    print(f"  • Gradient check passed: proj.weight.grad norm = {adapter.proj.weight.grad.norm().item():.4f}")


def test_campplus_embedding():
    print("\n=== [2] Testing 3D-Speaker (CAM++) Embedding Extraction ===")
    from speakerlab.utils.builder import build
    from speakerlab.utils.config import Config

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
    feature_extractor = build("feature_extractor", config)
    embedding_model = build("embedding_model", config)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    embedding_model.load_state_dict(state)
    embedding_model.eval()

    # Create 1s dummy 16kHz speech waveform
    t = torch.linspace(0, 1, 16000)
    dummy_wav = (torch.sin(2 * np.pi * 200 * t) + 0.5 * torch.sin(2 * np.pi * 500 * t)).unsqueeze(0)
    feat = feature_extractor(dummy_wav).unsqueeze(0)
    with torch.no_grad():
        emb = embedding_model(feat)
    print(f"  • CAM++ Output Embedding shape: {emb.shape} (expected: [1, 192])")
    assert emb.shape == (1, 192), f"Expected (1, 192), got {emb.shape}"
    print("  • 3D-Speaker CAM++ verified!")


def test_firered_integration():
    print("\n=== [3] Testing FireRedASR2-AED Integration ===")
    from fireredasr2s.fireredasr2.asr import load_fireredasr_aed_model
    model_path = MODEL_DIR / "fireredasr2s" / "FireRedASR2-AED" / "model.pth.tar"
    print(f"  • Loading FireRedASR2-AED from {model_path}...")
    model = load_fireredasr_aed_model(str(model_path))
    model.eval()

    # Freeze entire FireRed backbone
    for p in model.parameters():
        p.requires_grad = False

    adapter = SpeakerFiLMAdapter(spk_dim=192, d_model=model.encoder.odim)
    patch_conformer_encoder(model.encoder, adapter)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  • Frozen backbone parameters: {frozen_params:,}")
    print(f"  • Trainable adapter parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")

    # Forward pass comparison (spk_emb=None vs spk_emb=random with zero init)
    dummy_input = torch.randn(1, 100, 80)
    input_lengths = torch.tensor([100])
    with torch.no_grad():
        out_none, _, _ = model.encoder(dummy_input, input_lengths, spk_emb=None)
        dummy_spk = torch.randn(1, 192)
        out_spk_eval, _, _ = model.encoder(dummy_input, input_lengths, spk_emb=dummy_spk)

    max_diff = torch.max(torch.abs(out_none - out_spk_eval)).item()
    print(f"  • FireRed Encoder output max diff (None vs spk_emb): {max_diff:.8e}")
    assert max_diff == 0.0, "FireRed encoder with zero-init adapter must match unconditioned forward!"

    # CTC Loss computation check with gradients enabled for adapter
    print("  • Testing CTC Loss with Adapter...")
    out_spk, _, _ = model.encoder(dummy_input, input_lengths, spk_emb=dummy_spk)
    ctc_logits = model.ctc(out_spk)  # (N, T_sub, odim)
    print(f"  • CTC Logits shape: {ctc_logits.shape}")

    # Dummy CTC target
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    target = torch.tensor([[10, 25, 30, 45, 50]], dtype=torch.long)
    target_lengths = torch.tensor([5])
    input_lengths_sub = torch.tensor([ctc_logits.size(1)])

    # Transpose for PyTorch CTCLoss: (T, N, C)
    log_probs = ctc_logits.transpose(0, 1)
    loss = ctc_loss_fn(log_probs, target, input_lengths_sub, target_lengths)
    print(f"  • CTC Loss computed successfully: {loss.item():.4f}")

    loss.backward()
    assert adapter.proj.weight.grad is not None, "Adapter must receive CTC gradients!"
    print(f"  • Adapter gradient norm from CTC Loss: {adapter.proj.weight.grad.norm().item():.4f}")


if __name__ == "__main__":
    test_identity_and_gradients()
    test_campplus_embedding()
    test_firered_integration()
    print("\n>>> [SUCCESS] All Adapter Spike verifications passed flawlessly! <<<")
