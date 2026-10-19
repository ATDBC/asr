"""Phonetic distance gate for validating Chinese ASR error corrections."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata

try:
    import pypinyin
except ImportError:  # pragma: no cover
    pypinyin = None


INITIALS_GROUPS: list[set[str]] = [
    {"z", "zh"},
    {"c", "ch"},
    {"s", "sh"},
    {"l", "n", "r"},
    {"f", "h"},
    {"b", "p"},
    {"d", "t"},
    {"g", "k"},
    {"j", "q", "x"},
    {"j", "zh"},
    {"q", "ch"},
    {"x", "sh"},
]

FINALS_GROUPS: list[set[str]] = [
    {"in", "ing"},
    {"en", "eng"},
    {"an", "ang"},
    {"ian", "iang"},
    {"uan", "uang"},
    {"u", "ou"},
    {"o", "ou"},
    {"e", "ei"},
    {"ai", "ei"},
    {"i", "iu"},
    {"u", "iu"},
    {"ie", "e"},
]

INITIALS_LIST: list[str] = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t",
    "n", "l", "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s", "y", "w",
]


@dataclass(frozen=True, slots=True)
class PhoneticGateResult:
    original: str
    candidate: str
    similarity: float
    passed: bool
    original_pinyin: str
    candidate_pinyin: str
    threshold: float
    rejection_reason: str | None = None


def normalize_chinese_text(text: str) -> str:
    """Normalize text by NFKC, removing whitespace and common punctuation."""
    normalized = unicodedata.normalize("NFKC", text)
    filtered = []
    for char in normalized:
        if char.isspace() or char in "，。！？；：、“”‘’（）()[]{}<>,.!?;:-_":
            continue
        filtered.append(char)
    return "".join(filtered)


def get_pinyin_syllables(text: str) -> list[str]:
    """Convert Chinese text to a list of pinyin syllables without tone."""
    clean = normalize_chinese_text(text)
    if not clean:
        return []
    if pypinyin is None:
        raise RuntimeError("pypinyin is required for phonetic gate")
    raw = pypinyin.pinyin(clean, style=pypinyin.Style.NORMAL)
    return [item[0].lower().strip() for item in raw if item and item[0].strip()]


def split_syllable(s: str) -> tuple[str, str]:
    """Split a pinyin syllable into initial consonant and final vowel."""
    s = s.lower().strip()
    for init in INITIALS_LIST:
        if s.startswith(init):
            return init, s[len(init):]
    return "", s


SIBILANTS: set[str] = {"z", "zh", "c", "ch", "s", "sh"}


def syllable_distance(s1: str, s2: str) -> float:
    """Compute acoustic phonetic distance between two pinyin syllables (0.0 to 1.0)."""
    if s1 == s2:
        return 0.0

    i1, f1 = split_syllable(s1)
    i2, f2 = split_syllable(s2)

    i_match = (i1 == i2)
    i_near = any(i1 in g and i2 in g for g in INITIALS_GROUPS)
    i_sibilant = (i1 in SIBILANTS and i2 in SIBILANTS)

    f_match = (f1 == f2)
    f_near = any(f1 in g and f2 in g for g in FINALS_GROUPS)

    if i_match and f_near:
        return 0.15
    if i_near and f_match:
        return 0.15
    if i_sibilant and f_match:
        return 0.22
    if i_near and f_near:
        return 0.30
    if i_sibilant and f_near:
        return 0.35

    ratio = SequenceMatcher(None, s1, s2).ratio()
    return max(0.0, min(1.0, 1.0 - ratio * 0.8))


def sequence_phonetic_similarity(text1: str, text2: str) -> tuple[float, str, str]:
    """Compute normalized sequence phonetic similarity between two Chinese strings."""
    p1 = get_pinyin_syllables(text1)
    p2 = get_pinyin_syllables(text2)

    str1 = " ".join(p1)
    str2 = " ".join(p2)

    m, n = len(p1), len(p2)
    if m == 0 and n == 0:
        return 1.0, str1, str2
    if m == 0 or n == 0:
        return 0.0, str1, str2
    if p1 == p2:
        return 1.0, str1, str2

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = syllable_distance(p1[i - 1], p2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost,
            )
            # Support 2-to-1 swallowed/contracted syllable matching in fast speech
            if i >= 2:
                combined_cost = 0.8 + 0.5 * min(
                    syllable_distance(p1[i - 2], p2[j - 1]),
                    syllable_distance(p1[i - 1], p2[j - 1]),
                )
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 1] + combined_cost)
            if j >= 2:
                combined_cost = 0.8 + 0.5 * min(
                    syllable_distance(p1[i - 1], p2[j - 2]),
                    syllable_distance(p1[i - 1], p2[j - 1]),
                )
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 2] + combined_cost)

    max_len = max(m, n)
    sim = max(0.0, 1.0 - dp[m][n] / max_len)
    return round(sim, 3), str1, str2


def verify_phonetic_gate(
    original: str,
    candidate: str,
    threshold: float = 0.5,
) -> PhoneticGateResult:
    """Verify whether a candidate correction satisfies the phonetic gate threshold."""
    if not original.strip():
        return PhoneticGateResult(
            original=original,
            candidate=candidate,
            similarity=0.0,
            passed=False,
            original_pinyin="",
            candidate_pinyin="",
            threshold=threshold,
            rejection_reason="original text is empty",
        )
    if not candidate.strip():
        return PhoneticGateResult(
            original=original,
            candidate=candidate,
            similarity=0.0,
            passed=False,
            original_pinyin="",
            candidate_pinyin="",
            threshold=threshold,
            rejection_reason="candidate text is empty",
        )

    similarity, p1_str, p2_str = sequence_phonetic_similarity(original, candidate)
    passed = similarity >= threshold
    reason = None if passed else f"similarity {similarity:.3f} below threshold {threshold:.3f}"

    return PhoneticGateResult(
        original=original,
        candidate=candidate,
        similarity=similarity,
        passed=passed,
        original_pinyin=p1_str,
        candidate_pinyin=p2_str,
        threshold=threshold,
        rejection_reason=reason,
    )
