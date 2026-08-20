from funasr_transcriber import overlap_worker


def test_detect_uses_diarization_overlap_timeline(monkeypatch):
    class Segment:
        start = 1.25
        end = 2.75

    class Timeline:
        def support(self):
            return [Segment()]

    class Annotation:
        def get_overlap(self):
            return Timeline()

    monkeypatch.setattr(
        overlap_worker,
        "_load_detector",
        lambda request: (lambda path: Annotation(), "detector"),
    )

    result = overlap_worker._detect({"audio_path": "recording.wav"})

    assert result["windows"] == [
        {
            "id": "overlap-0001",
            "start_ms": 1250,
            "end_ms": 2750,
        }
    ]


def test_worker_configures_shared_model_cache_before_imports(
    monkeypatch,
    tmp_path,
):
    changed_to = []
    monkeypatch.setattr(overlap_worker.os, "chdir", changed_to.append)

    overlap_worker._configure_cache({"cache_dir": str(tmp_path)})

    assert overlap_worker.os.environ["HF_HOME"] == str(tmp_path.resolve())
    assert changed_to == [tmp_path.resolve()]
