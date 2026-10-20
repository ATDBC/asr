"""LLM-based post-correction and quality inspection using DeepSeek with phonetic gating."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from time import perf_counter
from typing import Any, Mapping

import httpx

from .config import LLMConfig
from .diagnostics import Diagnostics
from .pinyin_gate import PhoneticGateResult, verify_phonetic_gate
from .schemas import AlignmentConfidence, Transcript, Turn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个极高精度的中文语音识别（ASR）质检与纠错专家。你的任务是对对话录音的ASR初步识别结果进行近音纠错与专有名词校正。

【背景与输入特点】
输入的对话录音包含多人真实对话。语音识别模型在面对快语速、发音含糊、方言口音、连音吞音时，可能会将医学项目、医生姓名或专有名词误识别为读音相近的常见词汇。

【纠错原则与严格纪律】
1. 严禁自由改写：绝不可将口语改写为书面语，不可删除或修改语气词（如'嗯'、'啊'、'对'、'那个'等），不可润色句式。
2. 全局实体自发现（Session-level Entity Discovery）：仔细通读整篇对话，自动发现会话中高频、清晰提及的核心实体（如医生姓名、手术项目、专有名词、机构名称）。如果全篇某些轮次清晰提及了正确名称（例如'陈丹丹主任'、'宫廷六度'、'下至'），而其他轮次出现了发音相近的误识词，必须依据全局上下文进行纠偏。
3. 声学近音约束：每一个修改后的词必须与原词在中文发音上高度吻合（同音字、前后鼻音、平翘舌音、快语速吞音滑音）。严禁任何与原词发音完全无关的纯语义臆测。
4. 最小替换原则：original必须是原句中实际出现的词片段，corrected是纠正后的词。
5. 结构化JSON输出：你必须且只能输出合法的JSON对象，包含两个字段：
   - entities: 字符串数组，记录在全局对话中识别出的专有名词实体。
   - patches: 纠错补丁数组，每一项必须包含：
     - turn_index: 对应的轮次序号（整数）
     - original: 原句中被误识的字词（必须完全匹配原句内容）
     - corrected: 修正后的正确字词
     - reason: 纠错理由（说明发音相似性与上下文证据）

若通篇识别准确，无须纠错，patches请返回空数组 []。"""


@dataclass(frozen=True, slots=True)
class CorrectionAudit:
    turn_index: int
    original: str
    corrected: str
    reason: str
    similarity: float
    passed: bool
    original_pinyin: str
    candidate_pinyin: str
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "turn_index": self.turn_index,
            "original": self.original,
            "corrected": self.corrected,
            "reason": self.reason,
            "similarity": self.similarity,
            "passed": self.passed,
            "original_pinyin": self.original_pinyin,
            "candidate_pinyin": self.candidate_pinyin,
        }
        if self.rejection_reason is not None:
            data["rejection_reason"] = self.rejection_reason
        return data


@dataclass(frozen=True, slots=True)
class LLMCorrectionResult:
    transcript: Transcript
    patches_applied: tuple[CorrectionAudit, ...]
    patches_rejected: tuple[CorrectionAudit, ...]
    entities_discovered: tuple[str, ...]
    model: str
    duration_seconds: float
    warnings: tuple[str, ...]
    provider: str = "deepseek"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "duration_seconds": round(self.duration_seconds, 3),
            "entities_discovered": list(self.entities_discovered),
            "patches_applied_count": len(self.patches_applied),
            "patches_rejected_count": len(self.patches_rejected),
            "patches_applied": [p.to_dict() for p in self.patches_applied],
            "patches_rejected": [p.to_dict() for p in self.patches_rejected],
            "warnings": list(self.warnings),
        }


def format_turns_prompt(turns: tuple[Turn, ...]) -> str:
    """Format dialogue turns into a numbered context block for LLM prompt."""
    lines = []
    for turn in turns:
        lines.append(f"[{turn.index}] 说话人{turn.speaker_id}：{turn.text}")
    return "\n".join(lines)


def normalize_deepseek_model(model: str) -> str:
    name = model.strip()
    if name in {"deepseek-v4.1-flash", "deepseek-v4-flash"}:
        return "deepseek-flash"
    return name


def normalize_model_name(provider: str, model: str) -> str:
    name = model.strip()
    if provider == "deepseek" or "deepseek" in name.lower():
        return normalize_deepseek_model(name)
    return name


class DeepSeekCorrector:
    """Orchestrates LLM (DeepSeek / MiMo / DashScope Qwen / OpenAI-compatible) dialogue inspection and phonetic-gated correction."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def _resolve_endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _call_deepseek_api(self, prompt: str, api_key: str) -> dict[str, Any]:
        url = self._resolve_endpoint()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        target_model = normalize_model_name(self.config.provider, self.config.model)
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": 8192,
        }
        if not self.config.enable_thinking:
            is_mimo = self.config.provider == "mimo" or "mimo" in target_model.lower()
            is_deepseek = self.config.provider == "deepseek" or "deepseek" in target_model.lower()
            is_dashscope = self.config.provider in {"dashscope", "qwen"} or "qwen" in target_model.lower()
            if is_deepseek or is_mimo:
                payload["thinking"] = {"type": "disabled"}
            elif is_dashscope:
                payload["enable_thinking"] = False

        with httpx.Client(timeout=float(self.config.timeout_seconds)) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data

    def correct(
        self,
        transcript: Transcript,
        diagnostics: Diagnostics | None = None,
        dry_run_patches: list[dict[str, Any]] | None = None,
    ) -> LLMCorrectionResult:
        """Execute post-correction over transcript turns.

        If dry_run_patches is provided, skips the network API call and applies
        the provided patches through the full phonetic gate and validation flow.
        """
        start_time = perf_counter()
        warnings: list[str] = []

        api_key = self.config.resolve_api_key()
        if not api_key and dry_run_patches is None:
            msg = f"LLM post-correction enabled but API key ({self.config.api_key_env}) is not configured."
            if self.config.enabled:
                warnings.append(msg)
                if diagnostics is not None:
                    diagnostics.warn(msg)
            return LLMCorrectionResult(
                transcript=transcript,
                patches_applied=(),
                patches_rejected=(),
                entities_discovered=(),
                model=self.config.model,
                duration_seconds=perf_counter() - start_time,
                warnings=tuple(warnings),
                provider=self.config.provider,
            )

        if not transcript.turns:
            return LLMCorrectionResult(
                transcript=transcript,
                patches_applied=(),
                patches_rejected=(),
                entities_discovered=(),
                model=self.config.model,
                duration_seconds=perf_counter() - start_time,
                warnings=(),
                provider=self.config.provider,
            )

        # Call API or use provided dry run patches
        entities_discovered: list[str] = []
        raw_patches: list[dict[str, Any]] = []

        if dry_run_patches is not None:
            raw_patches = dry_run_patches
        else:
            prompt = format_turns_prompt(transcript.turns)
            try:
                api_response = self._call_deepseek_api(prompt, api_key)
                content = (
                    api_response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "{}")
                )
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "invalid structured response: root must be an object"
                    )
                raw_entities = parsed.get("entities", [])
                raw_patches = parsed.get("patches", [])
                if not isinstance(raw_entities, list) or not all(
                    isinstance(item, str) for item in raw_entities
                ):
                    raise ValueError(
                        "invalid structured response: entities must be strings"
                    )
                if not isinstance(raw_patches, list):
                    raise ValueError(
                        "invalid structured response: patches must be a list"
                    )
                entities_discovered = raw_entities
            except Exception as exc:
                err_msg = f"{self.config.provider} post-correction failed: {exc}"
                logger.warning(err_msg)
                warnings.append(err_msg)
                if diagnostics is not None:
                    diagnostics.warn(err_msg)
                return LLMCorrectionResult(
                    transcript=transcript,
                    patches_applied=(),
                    patches_rejected=(),
                    entities_discovered=(),
                    model=self.config.model,
                    duration_seconds=perf_counter() - start_time,
                    warnings=tuple(warnings),
                    provider=self.config.provider,
                )

        # Apply patches with phonetic gate
        applied_audits: list[CorrectionAudit] = []
        rejected_audits: list[CorrectionAudit] = []

        turns_by_index = {t.index: t for t in transcript.turns}
        updated_texts: dict[int, str] = {t.index: t.text for t in transcript.turns}

        for patch in raw_patches:
            if not isinstance(patch, dict):
                continue
            turn_idx = patch.get("turn_index")
            original = str(patch.get("original", "")).strip()
            corrected = str(patch.get("corrected", "")).strip()
            reason = str(patch.get("reason", "")).strip()

            if turn_idx not in turns_by_index:
                rejected_audits.append(
                    CorrectionAudit(
                        turn_index=turn_idx or 0,
                        original=original,
                        corrected=corrected,
                        reason=reason,
                        similarity=0.0,
                        passed=False,
                        original_pinyin="",
                        candidate_pinyin="",
                        rejection_reason=f"turn_index {turn_idx} does not exist",
                    )
                )
                continue

            current_text = updated_texts[turn_idx]
            if original not in current_text:
                rejected_audits.append(
                    CorrectionAudit(
                        turn_index=turn_idx,
                        original=original,
                        corrected=corrected,
                        reason=reason,
                        similarity=0.0,
                        passed=False,
                        original_pinyin="",
                        candidate_pinyin="",
                        rejection_reason=f"original term '{original}' not found in turn text",
                    )
                )
                continue

            gate_result: PhoneticGateResult = verify_phonetic_gate(
                original,
                corrected,
                threshold=self.config.pinyin_gate_threshold,
            )

            if not gate_result.passed:
                rejected_audits.append(
                    CorrectionAudit(
                        turn_index=turn_idx,
                        original=original,
                        corrected=corrected,
                        reason=reason,
                        similarity=gate_result.similarity,
                        passed=False,
                        original_pinyin=gate_result.original_pinyin,
                        candidate_pinyin=gate_result.candidate_pinyin,
                        rejection_reason=gate_result.rejection_reason,
                    )
                )
                continue

            # Accepted: perform in-place replacement
            updated_texts[turn_idx] = current_text.replace(original, corrected, 1)
            applied_audits.append(
                CorrectionAudit(
                    turn_index=turn_idx,
                    original=original,
                    corrected=corrected,
                    reason=reason,
                    similarity=gate_result.similarity,
                    passed=True,
                    original_pinyin=gate_result.original_pinyin,
                    candidate_pinyin=gate_result.candidate_pinyin,
                    rejection_reason=None,
                )
            )

        # Build modified Transcript
        new_turns: list[Turn] = []
        for t in transcript.turns:
            new_text = updated_texts[t.index]
            if new_text != t.text:
                new_turns.append(
                    Turn(
                        index=t.index,
                        speaker_id=t.speaker_id,
                        start_ms=t.start_ms,
                        end_ms=t.end_ms,
                        text=new_text,
                        source="llm_corrected",
                        confidence=t.confidence,
                        overlap=t.overlap,
                    )
                )
            else:
                new_turns.append(t)

        all_warnings = list(transcript.warnings) + warnings
        new_transcript = Transcript(
            language=transcript.language,
            speaker_count=transcript.speaker_count,
            turns=tuple(new_turns),
            warnings=tuple(all_warnings),
        )

        total_duration = perf_counter() - start_time
        res = LLMCorrectionResult(
            transcript=new_transcript,
            patches_applied=tuple(applied_audits),
            patches_rejected=tuple(rejected_audits),
            entities_discovered=tuple(entities_discovered),
            model=self.config.model,
            duration_seconds=total_duration,
            warnings=tuple(warnings),
            provider=self.config.provider,
        )

        if diagnostics is not None:
            diagnostics.stage_seconds["llm_correction"] = total_duration
            diagnostics.runtime["llm_correction"] = res.to_dict()

        return res


LLMCorrector = DeepSeekCorrector
