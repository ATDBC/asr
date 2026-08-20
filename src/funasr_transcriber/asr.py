"""FunASR adapter that preserves word-level timestamps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .errors import TranscriptionError
from .schemas import AsrResult, SentenceSpan, TimedToken


PUNCTUATION = set("，。！？；：、,.!?;:…")


class AsrModel(Protocol):
    def generate(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Generate FunASR-style results."""


def _punctuate_raw_tokens(
    raw_tokens: list[str],
    sentence_text: str,
) -> list[str]:
    plain_text = "".join(raw_tokens)
    char_to_token: list[int] = []
    for token_index, token in enumerate(raw_tokens):
        char_to_token.extend([token_index] * len(token))
    output = [""] * len(raw_tokens)
    raw_index = 0
    for character in sentence_text:
        if character.isspace():
            continue
        if raw_index < len(plain_text) and character == plain_text[raw_index]:
            output[char_to_token[raw_index]] += character
            raw_index += 1
            continue
        if character in PUNCTUATION and output:
            target = char_to_token[max(0, raw_index - 1)]
            output[target] += character
            continue
        raise ValueError(
            "sentence text cannot be aligned with raw timestamp tokens"
        )
    if raw_index != len(plain_text):
        raise ValueError("sentence text is missing raw timestamp tokens")
    return output


def parse_funasr_result(raw: dict[str, Any]) -> AsrResult:
    full_text = str(raw.get("text") or "")
    sentence_info = raw.get("sentence_info") or []
    tokens: list[TimedToken] = []
    untimed: list[SentenceSpan] = []
    for sentence in sentence_info:
        text = str(sentence.get("text") or "")
        start = int(sentence.get("start", 0))
        end = int(sentence.get("end", 0))
        raw_text = str(sentence.get("raw_text") or "").strip()
        timestamps = sentence.get("timestamp") or []
        if not raw_text or not timestamps:
            if text and end > start:
                untimed.append(SentenceSpan(text, start, end))
            continue
        raw_tokens = raw_text.split()
        if len(raw_tokens) != len(timestamps):
            raise ValueError(
                "raw token and timestamp count mismatch: "
                f"{len(raw_tokens)} != {len(timestamps)}"
            )
        punctuated = _punctuate_raw_tokens(raw_tokens, text)
        for token_text, timestamp in zip(punctuated, timestamps):
            if (
                not isinstance(timestamp, (list, tuple))
                or len(timestamp) != 2
            ):
                raise ValueError("timestamp must contain start and end")
            tokens.append(
                TimedToken(
                    text=token_text,
                    start_ms=int(timestamp[0]),
                    end_ms=int(timestamp[1]),
                )
            )
    tokens.sort(key=lambda item: (item.start_ms, item.end_ms))
    untimed.sort(key=lambda item: (item.start_ms, item.end_ms))
    return AsrResult(full_text, tuple(tokens), tuple(untimed))


class FunAsrEngine:
    def __init__(self, model: AsrModel):
        self.model = model

    def transcribe(self, path: Path) -> AsrResult:
        results = self.model.generate(
            input=str(path),
            batch_size_s=300,
            return_raw_text=True,
            sentence_timestamp=True,
        )
        if not results:
            raise TranscriptionError("FunASR returned an empty result")
        result = parse_funasr_result(results[0])
        if not result.text.strip():
            raise TranscriptionError("FunASR returned empty text")
        if not result.tokens and not result.untimed_sentences:
            raise TranscriptionError("FunASR returned text without timing data")
        return result

