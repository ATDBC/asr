from pathlib import Path
import json
import wave

import funasr_transcriber.corpus_builder as corpus_builder
from funasr_transcriber.corpus_builder import (
    CorpusCounts,
    CorpusPlan,
    SeedMaterial,
    build_validation_corpus,
    clip_audio,
    probe_wav,
    shift_rows,
    write_transcript,
)
from funasr_transcriber.ramc_corpus import (
    CleanClip,
    CoreCandidate,
    OverlapPlan,
    SyntheticMix,
    TranscriptRow,
    WindowClip,
    parse_transcript,
)


def write_pcm_fixture(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        stream.writeframes(b"\0\0\0\0" * 16_000)


def write_normalized_fixture(path: Path, duration_ms: int = 8_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * (duration_ms * 16))


def test_clip_audio_normalizes_to_16k_mono_pcm(tmp_path: Path):
    source = tmp_path / "source.wav"
    target = tmp_path / "clip.wav"
    write_pcm_fixture(source)

    clip_audio(source, target, 500, 1_500)
    properties = probe_wav(target)

    assert properties.sample_rate == 16_000
    assert properties.channels == 1
    assert properties.sample_width == 2
    assert 990 <= properties.duration_ms <= 1_010


def test_shift_rows_clips_intersections_and_moves_origin():
    rows = (
        TranscriptRow(500, 1_500, "S1", "女", "普通话", "第一句"),
        TranscriptRow(2_500, 3_500, "S2", "男", "普通话", "第二句"),
        TranscriptRow(4_000, 5_000, "S1", "女", "普通话", "窗口外"),
    )
    shifted = shift_rows(rows, 1_000, 3_000)

    assert shifted == (
        TranscriptRow(0, 500, "S1", "女", "普通话", "第一句"),
        TranscriptRow(1_500, 2_000, "S2", "男", "普通话", "第二句"),
    )


def test_write_transcript_round_trips_parser(tmp_path: Path):
    path = tmp_path / "sample.txt"
    rows = (
        TranscriptRow(0, 1_250, "S1", "女", "普通话", "你好"),
        TranscriptRow(1_250, 2_500, "S2", "男", "普通话", "世界"),
    )

    write_transcript(path, rows)

    assert parse_transcript(path) == rows


def test_build_validation_corpus_writes_complete_relative_layout(
    monkeypatch,
    tmp_path: Path,
):
    source_wav = tmp_path / "source.wav"
    write_pcm_fixture(source_wav)
    rows = (
        TranscriptRow(0, 1_000, "S1", "女", "普通话", "第一句话"),
        TranscriptRow(1_000, 2_000, "S2", "男", "普通话", "第二句话"),
    )
    core_rows = (
        TranscriptRow(60_000, 61_000, "S1", "女", "普通话", "第一句话"),
        TranscriptRow(61_000, 62_000, "S2", "男", "普通话", "第二句话"),
    )
    core = CoreCandidate(
        "core.wav",
        source_wav,
        tmp_path / "core.txt",
        core_rows,
        ("S1", "S2"),
        (("S1", 1_000), ("S2", 1_000)),
        500,
        "测试",
        ("F", "M"),
        ("北京", "四川"),
    )
    first = CleanClip("a.wav", source_wav, "A", 0, 1_000, "甲方内容")
    second = CleanClip("b.wav", source_wav, "B", 0, 1_000, "乙方内容")
    synthetic = SyntheticMix("synthetic-0001", first, second, 0, 1.0, 0)
    real = WindowClip(
        "real-overlap-0001",
        "real_overlap",
        "real.wav",
        source_wav,
        0,
        2_000,
        rows,
        source_marker=TranscriptRow(
            0, 1_000, "S1", "F", "Mandarin", "marker[+]"
        ),
    )
    negative = WindowClip(
        "speaker-change-negative-0001",
        "speaker_change_negative",
        "negative.wav",
        source_wav,
        0,
        2_000,
        rows,
        0,
    )
    seed = SeedMaterial(source_wav, 0, 2_000, rows)
    plan = CorpusPlan(
        core=(core,),
        overlap=OverlapPlan((synthetic,), (real,), (negative,)),
        format_fixture=seed,
        stress_source=seed,
        doctor_smoke=seed,
    )
    monkeypatch.setattr(
        corpus_builder,
        "create_corpus_plan",
        lambda source, counts, seed: plan,
    )

    def fake_clip(source, target, start, end, ffmpeg="ffmpeg"):
        write_normalized_fixture(Path(target))

    def fake_mix(item, target, work, ffmpeg="ffmpeg"):
        write_normalized_fixture(Path(target), duration_ms=1_000)

    monkeypatch.setattr(corpus_builder, "clip_audio", fake_clip)
    monkeypatch.setattr(
        corpus_builder,
        "mix_synthetic",
        fake_mix,
    )
    output = tmp_path / "artifact"

    build_validation_corpus(
        tmp_path / "dataset",
        output,
        counts=CorpusCounts(1, 1, 1, 1),
    )

    expected = {
        "ramc_core/WAV/core.wav",
        "ramc_core/TXT/core.txt",
        "ramc_core/test.tsv",
        "overlap/audio/synthetic-0001.wav",
        "overlap/audio/real-overlap-0001.wav",
        "overlap/audio/speaker-change-negative-0001.wav",
        "overlap/manifest.json",
        "seeds/format_fixture.wav",
        "seeds/stress_source.wav",
        "seeds/doctor_smoke.wav",
        "manifest.json",
        "checksums.sha256",
        "SOURCE_NOTICE.txt",
        "run_validation_linux.sh",
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert expected <= actual
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    assert manifest["counts"] == {
        "core": 1,
        "synthetic": 1,
        "real_overlap": 1,
        "speaker_change_negative": 1,
        "seeds": 3,
    }
    assert manifest["release_evidence"] == {
        "schema_version": "1.0",
        "required": [
            "firered_modules",
            "lid_coverage",
            "timestamp_integrity",
            "gpu_stages",
            "deterministic_hashes",
        ],
        "firered_modules": ["vad", "lid", "aed", "punc"],
        "stress_durations_minutes": [30, 60, 120],
    }
    assert "G:\\" not in (output / "manifest.json").read_text("utf-8")
