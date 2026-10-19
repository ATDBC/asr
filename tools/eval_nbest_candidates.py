import re
import sys
from pathlib import Path
import soundfile as sf
import torch

from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
from fireredasr2s.fireredasr2 import FireRedAsr2Config
from fireredasr2s.fireredpunc import FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVadConfig

def main():
    wav_path = "/mnt/hdd/jiajun/project/standalone_transcriber/outputs/runtime/audio/对话录音-1_1.wav"
    model_root = "/mnt/hdd/jiajun/project/standalone_transcriber/models/fireredasr2s"
    
    print("Loading audio...")
    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    assert sample_rate == 16000

    print("Initializing FireRedAsr2System with beam_size=10, nbest=10...")
    torch.cuda.set_device(0)
    
    config = FireRedAsr2SystemConfig(
        vad_model_dir=str(Path(model_root) / "FireRedVAD" / "VAD"),
        asr_type="aed",
        asr_model_dir=str(Path(model_root) / "FireRedASR2-AED"),
        punc_model_dir=str(Path(model_root) / "FireRedPunc"),
        vad_config=FireRedVadConfig(use_gpu=False),
        asr_config=FireRedAsr2Config(
            use_gpu=True,
            use_half=True,
            beam_size=10,
            nbest=10,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=False,
        ),
        punc_config=FireRedPuncConfig(use_gpu=True),
        asr_batch_size=1,
        punc_batch_size=16,
        enable_vad=False,
        enable_lid=False,
    )
    system = FireRedAsr2System(config)
    asr = system.asr
    
    test_segments = [
        ("Segment 1: 丹丹酒 vs 丹丹主任 (16.44s - 22.51s)", 16.44, 22.51),
        ("Segment 2: 丹丹酒局部聚焦 (19.0s - 22.0s)", 19.0, 22.0),
        ("Segment 3: 公屏六度 vs 宫廷六度 (381.0s - 386.0s)", 381.0, 386.0),
        ("Segment 4: 之前 vs 周年庆 (386.0s - 389.0s)", 386.0, 389.0),
    ]

    for title, start_s, end_s in test_segments:
        print("\n" + "="*70)
        print(f"=== {title} ===")
        print("="*70)
        s_idx = int(start_s * sample_rate)
        e_idx = int(end_s * sample_rate)
        sub_wav = wav_np[s_idx:e_idx]
        dur = len(sub_wav) / sample_rate
        
        batch_uttid = [f"test_s{int(start_s*1000)}_e{int(end_s*1000)}"]
        batch_wav = [(sample_rate, sub_wav)]
        
        feats, lengths, durs, _, _ = asr.feat_extractor(batch_wav, batch_uttid)
        feats, lengths = feats.cuda(), lengths.cuda()
        if torch.cuda.is_bf16_supported():
            feats = feats.bfloat16()
        else:
            feats = feats.half()
            
        with torch.no_grad():
            hyps = asr.model.transcribe(
                feats, lengths,
                beam_size=10,
                nbest=10,
                decode_max_len=0,
                softmax_smoothing=1.25,
                length_penalty=0.6,
                eos_penalty=1.0,
                return_timestamp=False,
                elm=asr.elm,
                elm_weight=0.0
            )
            
        cand_list = hyps[0]
        print(f"Extracted {len(cand_list)} candidate hypotheses:")
        for rank, hyp in enumerate(cand_list, start=1):
            hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
            text = asr.tokenizer.detokenize(hyp_ids)
            text = re.sub(r"(<blank>)|(<sil>)", "", text)
            score = float(hyp.get("score", 0.0))
            conf = float(hyp.get("confidence", 0.0))
            print(f"  [Rank {rank:2d}] score: {score:.4f}, conf: {conf:.4f} -> \"{text}\"")

if __name__ == '__main__':
    main()
