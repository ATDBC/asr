import importlib.util
from pathlib import Path

from funasr_transcriber.benchmarking import (
    build_overlap_manifest,
    input_format_matrix,
    score_overlap_cases,
    stress_durations_minutes,
)


_BENCHMARK_PATH = Path(__file__).parents[2] / "tools" / "benchmark_ramc_e2e.py"
_SPEC = importlib.util.spec_from_file_location("ramc_benchmark_tool", _BENCHMARK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
reference_for = _MODULE.reference_for


def test_ramc_benchmark_has_explicit_text_retention_gate():
    source = _BENCHMARK_PATH.read_text(encoding="utf-8")

    assert '"text_retention_rate"' in source
    assert '"text_retention": summary["text_retention_rate"] == 1.0' in source


def test_overlap_manifest_is_deterministic_and_cross_speaker():
    sources = [
        {
            "path": f"/audio/{index}.wav",
            "speaker": f"S{index % 4}",
            "text": f"文本{index}",
            "duration_seconds": 10.0,
        }
        for index in range(20)
    ]

    first = build_overlap_manifest(sources, count=50, seed=0)
    second = build_overlap_manifest(sources, count=50, seed=0)

    assert first == second
    assert len(first) == 50
    assert all(item["first"]["speaker"] != item["second"]["speaker"] for item in first)
    assert {item["snr_db"] for item in first} == {-5, 0, 5}
    assert all(1 <= item["overlap_seconds"] <= 8 for item in first)


def test_stress_manifest_has_required_lengths():
    assert stress_durations_minutes() == (30, 60, 120)


def test_input_matrix_covers_formats_rates_and_channels():
    matrix = input_format_matrix()

    assert {item["extension"] for item in matrix} == {
        ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"
    }
    assert {item["sample_rate"] for item in matrix} == {
        8000, 16000, 44100, 48000
    }
    assert {item["channels"] for item in matrix} == {1, 2}


def test_overlap_scoring_separates_synthetic_real_and_negative_gates():
    cases = [
        {
            "kind": "synthetic",
            "reference_turns": [
                {"speaker": "A", "text": "甲方完整内容"},
                {"speaker": "B", "text": "乙方完整内容"},
            ],
            "hypothesis_turns": [
                {"speaker": 1, "text": "甲方完整内容", "overlap": True},
                {"speaker": 2, "text": "乙方完整内容", "overlap": True},
            ],
        },
        {
            "kind": "real_overlap",
            "reference_turns": [],
            "hypothesis_turns": [],
        },
        {
            "kind": "speaker_change_negative",
            "reference_turns": [],
            "hypothesis_turns": [
                {"speaker": 1, "text": "正常内容", "overlap": True}
            ],
        },
    ]

    result = score_overlap_cases(cases)

    assert result["synthetic"] == {
        "total": 1,
        "recovered": 1,
        "recovery_rate": 1.0,
        "passed": True,
    }
    assert result["real_overlap"]["total"] == 1
    assert result["speaker_change_negative"]["false_positive_rate"] == 1.0
    assert result["passed"] is False


def test_preclipped_ramc_reference_uses_zero_origin(tmp_path):
    txt = tmp_path / "TXT"
    txt.mkdir()
    (txt / "sample.txt").write_text(
        "[0.000,1.000]\tS1\t女,普通话\t开头[+]\n"
        "[1.000,2.000]\tG00000000\t无,普通话\t[*]\n"
        "[2.000,3.000]\tS1\t女,普通话\t[LAUGHTER]\n"
        "[419.000,420.000]\tS2\t男,普通话\t结尾\n",
        encoding="utf-8",
    )

    intervals, by_speaker, full_text = reference_for(
        tmp_path,
        "sample",
        preclipped=True,
    )

    assert intervals[0]["start"] == 0.0
    assert intervals[-1]["end"] == 420.0
    assert by_speaker == {"S1": "开头", "S2": "结尾"}
    assert full_text == "开头结尾"
