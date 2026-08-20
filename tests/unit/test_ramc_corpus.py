from pathlib import Path

import pytest

from funasr_transcriber.ramc_corpus import (
    CoreCandidate,
    OverlapPlan,
    RecordingInfo,
    SpeakerInfo,
    TranscriptRow,
    analyze_core_candidate,
    parse_speaker_info,
    parse_transcript,
    parse_utterance_info,
    select_core_candidates,
    select_overlap_material,
)


def test_parse_transcript_preserves_timing_speaker_and_markers(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "[1.250,4.500]\tS1\t女,普通话\t你好[+]世界\n",
        encoding="utf-8",
    )

    assert parse_transcript(path) == (
        TranscriptRow(1250, 4500, "S1", "女", "普通话", "你好[+]世界"),
    )


def test_parse_utterance_info_indexes_by_wav_name(tmp_path: Path):
    path = tmp_path / "UTTERANCEINFO.txt"
    path.write_text(
        "CHANNEL\tUTTRANS_ID\tSPEAKER_ID-1\tSPEAKER_ID-2\tTOPIC\t"
        "VALID(min)\tTOTAL(min)\tEnvironment\n"
        "C0\ta.wav\tS1\tS2\t科技\t28.0\t31.0\t室内\n",
        encoding="utf-8",
    )

    assert parse_utterance_info(path) == {
        "a.wav": RecordingInfo(
            "a.wav", ("S1", "S2"), "科技", 28.0, 31.0, "室内"
        )
    }


def test_parse_speaker_info_indexes_gender_and_region(tmp_path: Path):
    path = tmp_path / "SPKINFO.txt"
    path.write_text(
        "CHANNEL\tSPEAKER_ID\tGENDER\tAGE\tREGION,CITY\tDEVICE\n"
        "C0\tS1\tF\t青年\t四川\t手机\n",
        encoding="utf-8",
    )

    assert parse_speaker_info(path) == {
        "S1": SpeakerInfo("S1", "F", "四川")
    }


@pytest.mark.parametrize(
    "content",
    [
        "bad row\n",
        "[4.0,1.0]\tS1\t女,普通话\t倒序\n",
        "[1.0,2.0]\t\t女,普通话\t缺少说话人\n",
    ],
)
def test_parse_transcript_reports_path_and_line(content: str, tmp_path: Path):
    path = tmp_path / "broken.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.txt:1"):
        parse_transcript(path)


def test_metadata_parser_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "SPKINFO.txt"
    path.write_text(
        "CHANNEL\tSPEAKER_ID\tGENDER\tAGE\tREGION,CITY\tDEVICE\n"
        "C0\tS1\tF\t青年\t四川\t手机\n"
        "C0\tS1\tF\t青年\t四川\t手机\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate speaker"):
        parse_speaker_info(path)


def candidate_rows(character_count: int = 500):
    text = "测" * character_count
    return (
        TranscriptRow(60_000, 120_000, "S1", "女", "普通话", text),
        TranscriptRow(120_000, 180_000, "S2", "男", "普通话", "试" * 20),
    )


def test_core_candidate_requires_two_balanced_speakers_and_500_characters(
    tmp_path: Path,
):
    info = RecordingInfo("a.wav", ("S1", "S2"), "科技", 28, 31, "室内")
    speakers = {
        "S1": SpeakerInfo("S1", "F", "四川"),
        "S2": SpeakerInfo("S2", "M", "北京"),
    }

    candidate = analyze_core_candidate(
        tmp_path / "a.wav",
        tmp_path / "a.txt",
        candidate_rows(),
        info,
        speakers,
        duration_ms=1_800_000,
    )

    assert candidate is not None
    assert candidate.character_count == 520
    assert candidate.speaker_speech_ms == (("S1", 60_000), ("S2", 60_000))
    assert candidate.gender_pair == ("F", "M")


@pytest.mark.parametrize(
    ("rows", "duration_ms"),
    [
        (candidate_rows(), 479_999),
        ((TranscriptRow(60_000, 120_000, "S1", "女", "普通话", "测" * 600),), 1_800_000),
        (
            (
                TranscriptRow(60_000, 120_000, "S1", "女", "普通话", "测" * 500),
                TranscriptRow(120_000, 149_999, "S2", "男", "普通话", "试" * 20),
            ),
            1_800_000,
        ),
        (candidate_rows(character_count=400), 1_800_000),
    ],
)
def test_ineligible_core_candidate_returns_none(rows, duration_ms, tmp_path: Path):
    info = RecordingInfo("a.wav", ("S1", "S2"), "科技", 28, 31, "室内")
    speakers = {
        "S1": SpeakerInfo("S1", "F", "四川"),
        "S2": SpeakerInfo("S2", "M", "北京"),
    }

    assert analyze_core_candidate(
        tmp_path / "a.wav",
        tmp_path / "a.txt",
        rows,
        info,
        speakers,
        duration_ms=duration_ms,
    ) is None


def test_core_candidate_rejects_an_unexpected_third_speaker(tmp_path: Path):
    info = RecordingInfo("a.wav", ("S1", "S2"), "topic", 28, 31, "inside")
    speakers = {
        "S1": SpeakerInfo("S1", "F", "A"),
        "S2": SpeakerInfo("S2", "M", "B"),
    }
    rows = candidate_rows() + (
        TranscriptRow(180_000, 181_000, "S3", "M", "Mandarin", "unexpected"),
    )

    assert analyze_core_candidate(
        tmp_path / "a.wav", tmp_path / "a.txt", rows, info, speakers, 1_800_000
    ) is None


def make_core(name, topic, gender_pair, regions):
    return CoreCandidate(
        wav_name=name,
        wav_path=Path(name),
        transcript_path=Path(name).with_suffix(".txt"),
        rows=candidate_rows(),
        speaker_ids=("S1", "S2"),
        speaker_speech_ms=(("S1", 60_000), ("S2", 60_000)),
        character_count=520,
        topic=topic,
        gender_pair=gender_pair,
        regions=regions,
    )


def test_core_selection_is_order_independent_and_covers_topics_first():
    candidates = (
        make_core("z.wav", "科技", ("F", "M"), ("北京", "四川")),
        make_core("a.wav", "科技", ("F", "F"), ("北京", "上海")),
        make_core("b.wav", "教育", ("M", "M"), ("广东", "浙江")),
    )

    first = select_core_candidates(candidates, count=2)
    second = select_core_candidates(tuple(reversed(candidates)), count=2)

    assert [item.wav_name for item in first] == ["a.wav", "b.wav"]
    assert first == second


def test_core_selection_rejects_insufficient_candidates():
    with pytest.raises(ValueError, match="requested 2"):
        select_core_candidates(
            (make_core("a.wav", "科技", ("F", "M"), ("北京", "四川")),),
            count=2,
        )


def overlap_candidate(name: str, rows: tuple[TranscriptRow, ...]):
    return CoreCandidate(
        wav_name=name,
        wav_path=Path(name),
        transcript_path=Path(name).with_suffix(".txt"),
        rows=rows,
        speaker_ids=("S1", "S2"),
        speaker_speech_ms=(("S1", 60_000), ("S2", 60_000)),
        character_count=1000,
        topic="测试",
        gender_pair=("F", "M"),
        regions=("北京", "四川"),
    )


def overlap_fixture_candidates():
    clean_a = overlap_candidate(
        "clean-a.wav",
        (
            TranscriptRow(60_000, 65_000, "A1", "女", "普通话", "甲" * 12),
            TranscriptRow(70_000, 75_000, "A2", "男", "普通话", "含有[LAUGHTER]"),
        ),
    )
    clean_b = overlap_candidate(
        "clean-b.wav",
        (TranscriptRow(80_000, 86_000, "B1", "男", "普通话", "乙" * 12),),
    )
    real = overlap_candidate(
        "real.wav",
        (
            TranscriptRow(100_000, 106_000, "R1", "女", "普通话", "真实重叠[+]第一路"),
            TranscriptRow(106_000, 112_000, "R2", "男", "普通话", "真实重叠第二路"),
        ),
    )
    negative = overlap_candidate(
        "negative.wav",
        (
            TranscriptRow(200_000, 204_000, "N1", "女", "普通话", "正常换人第一句话"),
            TranscriptRow(204_500, 209_000, "N2", "男", "普通话", "正常换人第二句话"),
        ),
    )
    return clean_a, clean_b, real, negative


def test_overlap_material_selects_clean_real_and_negative_cases():
    candidates = overlap_fixture_candidates()

    plan = select_overlap_material(
        candidates,
        synthetic_count=1,
        real_count=1,
        negative_count=1,
    )

    assert isinstance(plan, OverlapPlan)
    assert len(plan.synthetic) == 1
    assert plan.synthetic[0].first.speaker_id != plan.synthetic[0].second.speaker_id
    assert "[" not in plan.synthetic[0].first.text
    assert len(plan.real) == 1
    assert plan.real[0].kind == "real_overlap"
    assert all("[+]" not in row.text for row in plan.real[0].reference_rows)
    assert len(plan.negative) == 1
    assert plan.negative[0].kind == "speaker_change_negative"
    assert plan.negative[0].gap_ms == 500
    assert plan.real[0].wav_name != plan.negative[0].wav_name


def test_overlap_material_selection_is_order_independent():
    candidates = overlap_fixture_candidates()

    first = select_overlap_material(candidates, 1, 1, 1)
    second = select_overlap_material(tuple(reversed(candidates)), 1, 1, 1)

    assert first == second


def test_overlap_material_rejects_insufficient_clean_clips():
    with pytest.raises(ValueError, match="synthetic"):
        select_overlap_material(
            (overlap_fixture_candidates()[0],),
            synthetic_count=1,
            real_count=0,
            negative_count=0,
        )


def test_real_overlap_ignores_control_speaker_and_keeps_marker_evidence():
    candidate = overlap_candidate(
        "real-control.wav",
        (
            TranscriptRow(100_000, 104_000, "S1", "F", "Mandarin", "marked[+]"),
            TranscriptRow(104_000, 104_100, "G00000000", "U", "Mandarin", "[*]"),
            TranscriptRow(104_500, 108_000, "S2", "M", "Mandarin", "second"),
        ),
    )

    plan = select_overlap_material((candidate,), 0, 1, 0)

    assert plan.real[0].source_marker == candidate.rows[0]
    assert {row.speaker_id for row in plan.real[0].reference_rows} == {"S1", "S2"}


def test_negative_rejects_overlap_marker_within_two_second_buffer():
    candidate = overlap_candidate(
        "negative-near-marker.wav",
        (
            TranscriptRow(100_000, 104_000, "S1", "F", "Mandarin", "first words"),
            TranscriptRow(104_500, 108_000, "S2", "M", "Mandarin", "second words"),
            TranscriptRow(108_800, 109_500, "S1", "F", "Mandarin", "nearby[+]"),
        ),
    )

    with pytest.raises(ValueError, match="speaker-change negatives"):
        select_overlap_material((candidate,), 0, 0, 1)


def test_synthetic_selection_rotates_sources_and_leading_track():
    candidates = tuple(
        overlap_candidate(
            f"source-{index}.wav",
            (
                TranscriptRow(
                    60_000,
                    65_000 + index,
                    f"S{index}",
                    "F",
                    "Mandarin",
                    "测试内容" * 3,
                ),
            ),
        )
        for index in range(4)
    )

    mixes = select_overlap_material(candidates, 6, 0, 0).synthetic

    assert len({item.second.wav_name for item in mixes}) > 1
    assert len({(item.first.wav_name, item.second.wav_name) for item in mixes}) == 6
    assert {item.first.wav_name for item in mixes} == {
        "source-0.wav", "source-1.wav", "source-2.wav", "source-3.wav"
    }
    assert any(item.first_offset_ms > item.second_offset_ms for item in mixes)
    assert any(item.second_offset_ms > item.first_offset_ms for item in mixes)
