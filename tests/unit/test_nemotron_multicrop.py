from pathlib import Path

import pytest

from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.firered_parallel import (
    WeightedAedPuncWorkerClient,
)
from funasr_transcriber.schemas import AsrResult, TimedToken
from funasr_transcriber.speaker_crops import CropPlan, SpeakerCrop


def _plan(tmp_path: Path) -> CropPlan:
    crops = (
        SpeakerCrop(
            0, 1, 0, 6000, tmp_path / "s1.wav",
            "a" * 64, ((0, 6000),)
        ),
        SpeakerCrop(
            1, 2, 5000, 8000, tmp_path / "s2.wav",
            "b" * 64, ((0, 3000),)
        ),
    )
    return CropPlan(crops, 8000, 0, 2, 1000, ())


class Aed:
    def __init__(self, *, empty=False):
        self.paths = []
        self.empty = empty

    def run_aed_plan(self, path, batch_plan, vad):
        self.paths.append(Path(path))
        return {
            "aed_results": []
            if self.empty
            else [
                {
                    "index": 0,
                    "result": {
                        "uttid": "job_s0_e1000",
                        "text": "甲",
                        "confidence": 0.9,
                        "timestamp": [("甲", 0.0, 0.5)],
                    },
                }
            ],
            "duration_ms": max(end for _, end in vad),
            "batch_metrics": [],
        }


class Punc:
    def __init__(self):
        self.paths = []

    def punctuate(self, path, records, duration_ms, vad):
        self.paths.append(Path(path))
        return AsrResult(
            "甲",
            (TimedToken("甲", 0, 500, 0.9),),
            vad_segments_ms=tuple(tuple(item) for item in vad),
            duration_ms=duration_ms,
        )


def test_multicrop_uses_one_audio_path_per_aed_and_punc_request(tmp_path):
    aed = Aed()
    punc = Punc()
    owner = WeightedAedPuncWorkerClient(aed, punc)
    plan = _plan(tmp_path)

    results = owner.transcribe_crops(plan)

    assert aed.paths == [crop.path for crop in plan.crops]
    assert punc.paths == [crop.path for crop in plan.crops]
    assert [item.crop.speaker_id for item in results] == [1, 2]
    parallel = owner.last_response["metrics"]["parallel"]
    assert parallel["aed_ended_at_monotonic"] >= parallel["aed_started_at_monotonic"]


def test_two_lanes_keep_each_crop_on_one_path(tmp_path):
    first, second = Aed(), Aed()
    owner = WeightedAedPuncWorkerClient(first, Punc(), worker1=second)

    results = owner.transcribe_crops(_plan(tmp_path))

    assert len(results) == 2
    assert sorted(first.paths + second.paths) == sorted(
        crop.path for crop in _plan(tmp_path).crops
    )
    assert len(first.paths) == len(second.paths) == 1


def test_all_no_speech_fails_explicitly(tmp_path):
    owner = WeightedAedPuncWorkerClient(Aed(empty=True), Punc())

    with pytest.raises(TranscriptionError, match="no speech"):
        owner.transcribe_crops(_plan(tmp_path))
    assert owner.last_response["metrics"]["parallel"]["no_speech_crop_count"] == 2


def test_all_crop_plans_are_frozen_before_first_aed_request(
    tmp_path, monkeypatch
):
    import funasr_transcriber.firered_parallel as parallel
    original = parallel.plan_aed_batches
    planned = []

    def spy(*args, **kwargs):
        planned.append(1)
        return original(*args, **kwargs)

    class GuardedAed(Aed):
        def run_aed_plan(self, path, batch_plan, vad):
            assert len(planned) == 2
            return super().run_aed_plan(path, batch_plan, vad)

    monkeypatch.setattr(parallel, "plan_aed_batches", spy)
    owner = WeightedAedPuncWorkerClient(GuardedAed(), Punc())
    owner.transcribe_crops(_plan(tmp_path))


def test_one_no_speech_crop_does_not_discard_other_crop(tmp_path):
    class OneEmpty(Aed):
        def run_aed_plan(self, path, batch_plan, vad):
            response = super().run_aed_plan(path, batch_plan, vad)
            if Path(path).name == "s2.wav":
                response["aed_results"] = []
            return response

    owner = WeightedAedPuncWorkerClient(OneEmpty(), Punc())
    results = owner.transcribe_crops(_plan(tmp_path))

    assert [result.result.text for result in results] == ["甲", ""]
    assert owner.last_response["metrics"]["parallel"][
        "no_speech_crop_count"
    ] == 1


def test_lane_error_propagates_after_other_lane_finishes(tmp_path):
    from threading import Event

    completed = Event()

    class Failing(Aed):
        def run_aed_plan(self, path, batch_plan, vad):
            raise RuntimeError("AED lane failed")

    class Successful(Aed):
        def run_aed_plan(self, path, batch_plan, vad):
            response = super().run_aed_plan(path, batch_plan, vad)
            completed.set()
            return response

    owner = WeightedAedPuncWorkerClient(
        Failing(), Punc(), worker1=Successful()
    )
    with pytest.raises(RuntimeError, match="AED lane failed"):
        owner.transcribe_crops(_plan(tmp_path))
    assert completed.is_set()
