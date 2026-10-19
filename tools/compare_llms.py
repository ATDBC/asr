import json
import time
import os
from funasr_transcriber.config import load_config, LLMConfig
from funasr_transcriber.llm_corrector import DeepSeekCorrector
from funasr_transcriber.schemas import Transcript, Turn, AlignmentConfidence

# 1. Load config and .env
load_config()

# 2. Reconstruct raw uncorrected transcript of 1-hour audio (212 turns)
with open("/mnt/hdd/jiajun/project/standalone_transcriber/outputs/runtime/results/刘洋洋_nemotron_llm/transcript.json") as f:
    t_data = json.load(f)
with open("/mnt/hdd/jiajun/project/standalone_transcriber/outputs/runtime/results/刘洋洋_nemotron_llm/diagnostics.json") as f:
    diag = json.load(f)

applied_patches = diag["runtime"]["llm_correction"]["patches_applied"]
text_map = {t["index"]: t["text"] for t in t_data["turns"]}
for p in reversed(applied_patches):
    idx = p["turn_index"]
    orig = p["original"]
    corr = p["corrected"]
    text_map[idx] = text_map[idx].replace(corr, orig, 1)

raw_turns = []
for t in t_data["turns"]:
    raw_turns.append(
        Turn(
            index=t["index"],
            speaker_id=t.get("speaker", 1),
            start_ms=t["start_ms"],
            end_ms=t["end_ms"],
            text=text_map[t["index"]],
            source="primary_asr",
            confidence=AlignmentConfidence(t.get("alignment_confidence", "high")),
            overlap=t.get("overlap", False),
        )
    )
raw_transcript = Transcript(
    language=t_data["language"],
    speaker_count=t_data["speaker_count"],
    turns=tuple(raw_turns),
    warnings=(),
)
print(f"Loaded raw transcript: {len(raw_transcript.turns)} turns, {sum(len(t.text) for t in raw_transcript.turns)} characters.")

# Instrument corrector to capture token usage
class DetailedCorrector(DeepSeekCorrector):
    def __init__(self, config):
        super().__init__(config)
        self.last_usage = {}

    def _call_deepseek_api(self, prompt, api_key):
        resp_data = super()._call_deepseek_api(prompt, api_key)
        self.last_usage = resp_data.get("usage", {})
        return resp_data

models = [
    {
        "name": "DeepSeek (deepseek-v4.1-flash)",
        "provider": "deepseek",
        "model": "deepseek-v4.1-flash",
        "base_url": "https://api.deepseek.com",
        "env_var": "DEEPSEEK_API_KEY",
    },
    {
        "name": "Xiaomi MiMo (mimo-v2.6-flash)",
        "provider": "mimo",
        "model": "mimo-v2.6-flash",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "env_var": "MIMO_API_KEY",
    },
    {
        "name": "Alibaba DashScope (qwen3.7-flash)",
        "provider": "dashscope",
        "model": "qwen3.7-flash",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_var": "DASHSCOPE_API_KEY",
    },
]

results = []

for m in models:
    print("\n" + "="*50)
    print("Running evaluation on: " + m["name"])
    print("="*50)
    cfg = LLMConfig(
        enabled=True,
        provider=m["provider"],
        api_key="",
        api_key_env=m["env_var"],
        base_url=m["base_url"],
        model=m["model"],
        enable_thinking=False,
        temperature=0.1,
        timeout_seconds=90,
        pinyin_gate_threshold=0.5,
        max_context_turns=250,
    )
    corrector = DetailedCorrector(cfg)
    t0 = time.perf_counter()
    res = corrector.correct(raw_transcript)
    dur = time.perf_counter() - t0

    usage = corrector.last_usage
    p_tok = usage.get("prompt_tokens", 0)
    c_tok = usage.get("completion_tokens", 0)
    t_tok = usage.get("total_tokens", 0)
    cache_hit = usage.get("prompt_cache_hit_tokens", 0)

    print(f"Finished in {dur:.2f}s!")
    print(f"Token stats -> Prompt: {p_tok} (Cache hit: {cache_hit}), Completion: {c_tok}, Total: {t_tok}")
    print(f"Entities discovered count: {len(res.entities_discovered)}")
    print(f"Patches applied (passed gate): {len(res.patches_applied)}")
    print(f"Patches rejected (gate blocked): {len(res.patches_rejected)}")
    if res.warnings:
        print(f"Warnings: {res.warnings}")

    results.append({
        "info": m,
        "duration_seconds": dur,
        "usage": usage,
        "result": res,
    })

# Save comparison data to json
out_summary = []
for r in results:
    m = r["info"]
    res = r["result"]
    out_summary.append({
        "name": m["name"],
        "provider": m["provider"],
        "model": m["model"],
        "duration_seconds": round(r["duration_seconds"], 2),
        "usage": r["usage"],
        "entities_count": len(res.entities_discovered),
        "entities": list(res.entities_discovered),
        "patches_applied_count": len(res.patches_applied),
        "patches_rejected_count": len(res.patches_rejected),
        "patches_applied": [p.to_dict() for p in res.patches_applied],
        "patches_rejected": [p.to_dict() for p in res.patches_rejected],
        "warnings": list(res.warnings),
    })

with open("/mnt/hdd/jiajun/project/standalone_transcriber/outputs/llm_comparison_1h.json", "w", encoding="utf-8") as f:
    json.dump(out_summary, f, ensure_ascii=False, indent=2)

print("\nSaved full comparison report to outputs/llm_comparison_1h.json")
