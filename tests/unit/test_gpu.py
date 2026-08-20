from __future__ import annotations

import pytest

from funasr_transcriber.errors import EnvironmentError
from funasr_transcriber.gpu import (
    GpuSample,
    build_gpu_stage_record,
    query_gpu,
    require_gpu_capacity,
)


class Result:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


GPU_OUTPUT = "\n".join(
    (
        "0, NVIDIA GeForce RTX 3090, 24576, 12288, 580.142",
        "1, NVIDIA GeForce RTX 3090, 24576, 12000, 580.142",
    )
)


def _runner(output: str = GPU_OUTPUT, returncode: int = 0):
    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return Result(output, returncode, "nvidia-smi failed")

    return run


def test_query_gpu_selects_explicit_physical_device_zero():
    sample = query_gpu("cuda:0", runner=_runner())

    assert sample == GpuSample(
        index=0,
        name="NVIDIA GeForce RTX 3090",
        total_mib=24576,
        free_mib=12288,
        driver_version="580.142",
    )


def test_capacity_accepts_exact_total_and_free_thresholds():
    sample = require_gpu_capacity("cuda:0", runner=_runner())

    assert sample.free_mib == 12 * 1024


@pytest.mark.parametrize(
    ("output", "message"),
    (
        ("0, Small GPU, 23551, 22000, 580.142", "total memory"),
        ("0, RTX 3090, 24576, 12287, 580.142", "free memory"),
    ),
)
def test_capacity_rejects_insufficient_memory(output: str, message: str):
    with pytest.raises(EnvironmentError, match=message):
        require_gpu_capacity("cuda:0", runner=_runner(output))


@pytest.mark.parametrize("device", ("cpu", "cuda", "cuda:x", "cuda:-1"))
def test_query_rejects_invalid_device(device: str):
    with pytest.raises(EnvironmentError, match="CUDA device"):
        query_gpu(device, runner=_runner())


def test_query_failure_is_fatal():
    with pytest.raises(EnvironmentError, match="nvidia-smi"):
        query_gpu("cuda:0", runner=_runner("", returncode=1))


def test_query_rejects_unparseable_or_missing_device():
    with pytest.raises(EnvironmentError, match="device 0"):
        query_gpu("cuda:0", runner=_runner("1, malformed"))


def test_gpu_stage_record_marks_release_with_small_sampling_tolerance():
    before = GpuSample(0, "RTX 3090", 24576, 22000, "580.142")
    after = GpuSample(0, "RTX 3090", 24576, 21488, "580.142")

    record = build_gpu_stage_record(
        before,
        after,
        peak_gpu_bytes=123456,
        load_seconds=1.25,
        inference_seconds=2.5,
    )

    assert record == {
        "device_index": 0,
        "before_free_mib": 22000,
        "after_free_mib": 21488,
        "peak_gpu_bytes": 123456,
        "load_seconds": 1.25,
        "inference_seconds": 2.5,
        "released": True,
    }


def test_gpu_stage_record_rejects_more_than_cuda_context_tolerance():
    before = GpuSample(0, "RTX 3090", 24576, 22000, "580.142")
    after = GpuSample(0, "RTX 3090", 24576, 21487, "580.142")

    record = build_gpu_stage_record(before, after, None, None, None)

    assert record["released"] is False
