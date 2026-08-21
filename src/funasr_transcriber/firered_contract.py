"""Pure normalization for the FireRedASR2S worker boundary."""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable
import unicodedata


class FireRedContractError(ValueError):
    """FireRed returned data that cannot satisfy the stable ASR contract."""


class NoSpeechError(FireRedContractError):
    """FireRedVAD did not find any speech."""


def normalize_text(text: str) -> str:
    """Normalize compatibility characters and remove all whitespace."""

    normalized = unicodedata.normalize("NFKC", str(text))
    return "".join(character for character in normalized if not character.isspace())


def _is_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith("P")


def _same_character(left: str, right: str) -> bool:
    return left == right or left.casefold() == right.casefold()


def attach_sentence_punctuation(
    words: list[dict[str, object]],
    sentence_text: str,
) -> list[str]:
    """Project punctuated sentence characters back onto timestamped words."""

    raw_parts = [normalize_text(str(word.get("text") or "")) for word in words]
    if not raw_parts or any(not part for part in raw_parts):
        raise FireRedContractError("sentence text cannot be aligned without words")
    raw_text = "".join(raw_parts)
    char_to_word = [
        index
        for index, part in enumerate(raw_parts)
        for _ in part
    ]
    output = ["" for _ in words]
    raw_index = 0
    leading_punctuation = ""
    for character in str(sentence_text):
        if character.isspace():
            continue
        comparable = normalize_text(character)
        expected = raw_text[raw_index:raw_index + len(comparable)]
        if (
            comparable
            and len(expected) == len(comparable)
            and all(
                _same_character(left, right)
                for left, right in zip(comparable, expected)
            )
        ):
            word_index = char_to_word[raw_index]
            if leading_punctuation:
                output[word_index] += leading_punctuation
                leading_punctuation = ""
            output[word_index] += character
            raw_index += len(comparable)
            continue
        if _is_punctuation(character):
            if raw_index == 0:
                leading_punctuation += character
            else:
                output[char_to_word[raw_index - 1]] += character
            continue
        raise FireRedContractError(
            "sentence text cannot be aligned with timestamp words"
        )
    if raw_index != len(raw_text) or leading_punctuation:
        raise FireRedContractError(
            "sentence text cannot be aligned with all timestamp words"
        )
    if any(not token for token in output):
        raise FireRedContractError("sentence produced an empty timestamp token")
    return output


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise FireRedContractError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FireRedContractError(f"{name} must be numeric") from exc
    if not isfinite(number):
        raise FireRedContractError(f"{name} must be finite")
    return number


def _milliseconds(value: object, name: str) -> int:
    return int(round(_number(value, name)))


def _confidence(value: object, name: str) -> float:
    result = _number(value, name)
    if not 0.0 <= result <= 1.0:
        raise FireRedContractError(f"{name} must stay within 0..1")
    return result


def _validate_ranges(
    ranges: Iterable[tuple[int, int]],
    duration_ms: int,
    name: str,
) -> None:
    previous_start = -1
    previous_end = -1
    for start_ms, end_ms in ranges:
        if not 0 <= start_ms < end_ms <= duration_ms:
            raise FireRedContractError(f"{name} timestamp bounds are invalid")
        if start_ms < previous_start or end_ms < previous_end:
            raise FireRedContractError(f"{name} timestamps are not monotonic")
        previous_start = start_ms
        previous_end = end_ms


def _sentence_owners(
    tokens: list[dict[str, object]],
    sentences: list[dict[str, object]],
) -> list[int]:
    owners: list[int] = []
    for token in tokens:
        midpoint = (
            int(token["start_ms"]) + int(token["end_ms"])
        ) / 2
        matching = [
            index
            for index, sentence in enumerate(sentences)
            if int(sentence["start_ms"]) <= midpoint
            and (
                midpoint < int(sentence["end_ms"])
                or (
                    index == len(sentences) - 1
                    and midpoint == int(sentence["end_ms"])
                )
            )
        ]
        if len(matching) != 1:
            raise FireRedContractError(
                "timestamp word must belong to exactly one sentence"
            )
        owners.append(matching[0])
    return owners


def validate_stable_response(response: dict[str, object]) -> None:
    """Validate the JSON-serializable response exposed to the main process."""

    if response.get("ok") is not True:
        raise FireRedContractError("stable response must be successful")
    text = str(response.get("text") or "")
    tokens = response.get("tokens")
    sentences = response.get("sentences")
    vad_segments = response.get("vad_segments_ms")
    if not text.strip():
        raise FireRedContractError("stable response text must not be empty")
    if not isinstance(tokens, list) or not tokens:
        raise FireRedContractError("stable response text is without words")
    if not isinstance(sentences, list) or not sentences:
        raise FireRedContractError("stable response requires sentences")
    if not isinstance(vad_segments, list) or not vad_segments:
        raise NoSpeechError("FireRedVAD found no speech")
    duration_ms = _milliseconds(response.get("duration_ms"), "duration_ms")
    if duration_ms <= 0:
        raise FireRedContractError("duration_ms must be positive")

    token_ranges: list[tuple[int, int]] = []
    for token in tokens:
        if not isinstance(token, dict) or not str(token.get("text") or ""):
            raise FireRedContractError("token text must not be empty")
        token_ranges.append(
            (
                _milliseconds(token.get("start_ms"), "token start_ms"),
                _milliseconds(token.get("end_ms"), "token end_ms"),
            )
        )
        _confidence(token.get("confidence"), "token confidence")
    _validate_ranges(token_ranges, duration_ms, "token")

    sentence_ranges: list[tuple[int, int]] = []
    for sentence in sentences:
        if not isinstance(sentence, dict):
            raise FireRedContractError("sentence must be an object")
        if not str(sentence.get("text") or ""):
            raise FireRedContractError("sentence text must not be empty")
        sentence_ranges.append(
            (
                _milliseconds(sentence.get("start_ms"), "sentence start_ms"),
                _milliseconds(sentence.get("end_ms"), "sentence end_ms"),
            )
        )
        _confidence(
            sentence.get("asr_confidence"),
            "sentence asr_confidence",
        )
    _validate_ranges(sentence_ranges, duration_ms, "sentence")

    normalized_vad: list[tuple[int, int]] = []
    for segment in vad_segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise FireRedContractError("VAD segment must contain two timestamps")
        start_ms = _milliseconds(segment[0], "VAD start_ms")
        end_ms = _milliseconds(segment[1], "VAD end_ms")
        if end_ms - start_ms > 60_000:
            raise FireRedContractError("VAD segment exceeds the 60 second AED limit")
        normalized_vad.append((start_ms, end_ms))
    _validate_ranges(normalized_vad, duration_ms, "VAD")

    token_text = "".join(str(token["text"]) for token in tokens)
    if normalize_text(token_text) != normalize_text(text):
        raise FireRedContractError("tokens do not preserve the full text")
    sentence_text = "".join(str(item["text"]) for item in sentences)
    if normalize_text(sentence_text) != normalize_text(text):
        raise FireRedContractError("sentences do not preserve the full text")

    owners = _sentence_owners(tokens, sentences)
    for index, sentence in enumerate(sentences):
        covered = "".join(
            str(token["text"])
            for token, owner in zip(tokens, owners)
            if owner == index
        )
        if normalize_text(covered) != normalize_text(str(sentence["text"])):
            raise FireRedContractError(
                "sentence text does not match its timestamp tokens"
            )


def normalize_official_result(
    request_id: str,
    raw: dict[str, object],
) -> dict[str, object]:
    """Convert one pinned FireRedASR2S result to the stable worker schema."""

    duration_ms = int(round(_number(raw.get("dur_s"), "dur_s") * 1000))
    if duration_ms <= 0:
        raise FireRedContractError("audio duration must be positive")

    raw_vad = raw.get("vad_segments_ms")
    if not isinstance(raw_vad, (list, tuple)) or not raw_vad:
        raise NoSpeechError("FireRedVAD found no speech")
    vad_segments: list[list[int]] = []
    for segment in raw_vad:
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise FireRedContractError("VAD segment must contain two timestamps")
        vad_segments.append(
            [
                _milliseconds(segment[0], "VAD start_ms"),
                _milliseconds(segment[1], "VAD end_ms"),
            ]
        )

    raw_sentences = raw.get("sentences")
    if not isinstance(raw_sentences, (list, tuple)) or not raw_sentences:
        raise FireRedContractError("FireRed returned text without sentences")
    sentences: list[dict[str, object]] = []
    for raw_sentence in raw_sentences:
        if not isinstance(raw_sentence, dict):
            raise FireRedContractError("official sentence must be an object")
        sentences.append(
            {
                "text": str(raw_sentence.get("text") or ""),
                "start_ms": _milliseconds(
                    raw_sentence.get("start_ms"),
                    "sentence start_ms",
                ),
                "end_ms": _milliseconds(
                    raw_sentence.get("end_ms"),
                    "sentence end_ms",
                ),
                "asr_confidence": _confidence(
                    raw_sentence.get("asr_confidence"),
                    "sentence asr_confidence",
                ),
            }
        )

    raw_words = raw.get("words")
    if not isinstance(raw_words, (list, tuple)) or not raw_words:
        raise FireRedContractError("FireRed returned text without words")
    words: list[dict[str, object]] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            raise FireRedContractError("official word must be an object")
        words.append(
            {
                "text": str(raw_word.get("text") or ""),
                "start_ms": _milliseconds(
                    raw_word.get("start_ms"),
                    "word start_ms",
                ),
                "end_ms": _milliseconds(
                    raw_word.get("end_ms"),
                    "word end_ms",
                ),
            }
        )
    _validate_ranges(
        [
            (int(word["start_ms"]), int(word["end_ms"]))
            for word in words
        ],
        duration_ms,
        "word",
    )
    _validate_ranges(
        [
            (int(sentence["start_ms"]), int(sentence["end_ms"]))
            for sentence in sentences
        ],
        duration_ms,
        "sentence",
    )
    owners = _sentence_owners(words, sentences)

    punctuated: list[str] = ["" for _ in words]
    for sentence_index, sentence in enumerate(sentences):
        indexes = [
            index
            for index, owner in enumerate(owners)
            if owner == sentence_index
        ]
        if not indexes:
            raise FireRedContractError("sentence text is without words")
        attached = attach_sentence_punctuation(
            [words[index] for index in indexes],
            str(sentence["text"]),
        )
        for word_index, token_text in zip(indexes, attached):
            punctuated[word_index] = token_text

    tokens = [
        {
            "text": punctuated[index],
            "start_ms": word["start_ms"],
            "end_ms": word["end_ms"],
            "confidence": sentences[owners[index]]["asr_confidence"],
        }
        for index, word in enumerate(words)
    ]
    response: dict[str, object] = {
        "id": str(request_id),
        "ok": True,
        "text": str(raw.get("text") or ""),
        "tokens": tokens,
        "sentences": sentences,
        "vad_segments_ms": vad_segments,
        "duration_ms": duration_ms,
    }
    validate_stable_response(response)
    return response
