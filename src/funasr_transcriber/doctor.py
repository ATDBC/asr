"""Linux runtime preflight checks for isolated GPU model phases."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Callable

from .config import AppConfig
from .errors import EnvironmentError as TranscriberEnvironmentError
from .errors import ModelError
from .firered_client import FireRedWorkerClient
from .gpu import (
    GpuSample,
    build_gpu_stage_record,
    query_gpu,
    require_gpu_capacity,
)
from .model_registry import resolve_firered_paths, verify_models
from .overlap import JsonOverlapWorkerClient
from .schemas import AsrResult
from .speaker_diarization import SpeakerDiarizer


def _require_released(before: GpuSample, after: GpuSample, stage: str) -> None:
    record = build_gpu_stage_record(before, after, None, None, None)
    if not record["released"]:
        raise TranscriberEnvironmentError(
            f"GPU memory was not released after {stage}: "
            f"before={before.free_mib} MiB, after={after.free_mib} MiB"
        )


def _validate_firered_smoke(result: Any) -> None:
    if not isinstance(result, AsrResult) or not (
        result.text.strip()
        and result.tokens
        and result.sentences
        and result.vad_segments_ms
        and result.duration_ms > 0
    ):
        raise ModelError(
            "FireRed worker verification returned incomplete response"
        )


def _default_asr_client(config: AppConfig) -> FireRedWorkerClient:
    return FireRedWorkerClient(
        worker_python=config.asr.worker_python,
        model_root=config.paths.model_dir,
        device="cuda:0",
        timeout_seconds=config.asr.timeout_seconds,
        stderr_log_path=config.paths.work_dir / "doctor-firered-worker.log",
        segment_batch_size=config.asr.segment_batch_size,
    )


def _default_overlap_verifier(config: AppConfig) -> None:
    worker = JsonOverlapWorkerClient(
        worker_python=config.overlap.worker_python,
        worker_script=Path(__file__).with_name("overlap_worker.py"),
        work_dir=config.paths.work_dir / "doctor-overlap-worker",
        timeout_seconds=config.overlap.timeout_seconds,
    )
    worker.call(
        "verify",
        {
            "model_id": config.overlap.detector_model,
            "hf_token_env": config.overlap.hf_token_env,
            "device": "cuda:0",
            "model": "MossFormer2_SS_16K",
            "cache_dir": str(
                (config.paths.model_dir / "huggingface").resolve()
            ),
        },
    )


def run_doctor(
    config: AppConfig,
    check_cuda: bool = True,
    smoke_test: Callable[[], None] | None = None,
    *,
    asr_client_factory: Callable[[], Any] | None = None,
    speaker_factory: Callable[[], Any] | None = None,
    overlap_verifier: Callable[[], None] | None = None,
    gpu_capacity_check: Callable[[str], GpuSample] = require_gpu_capacity,
    gpu_sampler: Callable[[str], GpuSample] = query_gpu,
) -> tuple[str, ...]:
    checks: list[str] = []
    required_commands = ["ffmpeg", "ffprobe"]
    if check_cuda:
        required_commands.append("nvidia-smi")
    for command in required_commands:
        if shutil.which(command) is None:
            raise TranscriberEnvironmentError(
                f"required command is missing: {command}"
            )
        checks.append(f"{command}: ok")

    for directory in (config.paths.model_dir, config.paths.work_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)
    test_file = config.paths.work_dir / ".doctor-write-test"
    try:
        test_file.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise TranscriberEnvironmentError(
            f"directory is not writable: {config.paths.work_dir}"
        ) from exc
    finally:
        test_file.unlink(missing_ok=True)
    checks.append("work_dir: writable")

    minimum_free_bytes = 2 * 1024**3
    free_bytes = shutil.disk_usage(config.paths.work_dir).free
    if free_bytes < minimum_free_bytes:
        raise TranscriberEnvironmentError(
            "work_dir requires at least 2 GiB free space"
        )
    checks.append(f"work_dir_free_bytes: {free_bytes}")

    if not config.asr.worker_python.is_file():
        raise TranscriberEnvironmentError(
            f"ASR worker Python is missing: {config.asr.worker_python}"
        )
    checks.append("firered_venv: ok")
    if config.overlap.enabled:
        if not config.overlap.worker_python.is_file():
            raise TranscriberEnvironmentError(
                "overlap worker Python is missing: "
                f"{config.overlap.worker_python}"
            )
        checks.append("overlap_venv: ok")

    verification = verify_models(config.paths.model_dir)
    if not verification.ok:
        raise ModelError(
            "model verification failed: " + ", ".join(verification.missing)
        )
    checks.extend(
        (
            "firered_source: verified",
            "firered_models: verified",
        )
    )

    if not check_cuda:
        checks.append("model_smoke: skipped (--skip-cuda)")
        return tuple(checks)
    if config.general.device != "cuda:0":
        raise TranscriberEnvironmentError(
            "doctor is restricted to physical GPU 0 (cuda:0)"
        )

    doctor_audio = (
        resolve_firered_paths(
            config.paths.model_dir,
            require_existing=False,
        ).source
        / "assets"
        / "hello_zh.wav"
    ).resolve()
    asr_factory = asr_client_factory or (
        lambda: _default_asr_client(config)
    )
    before = gpu_capacity_check("cuda:0")
    try:
        with asr_factory() as client:
            _validate_firered_smoke(client.verify(doctor_audio))
    except (ModelError, TranscriberEnvironmentError):
        raise
    except Exception as exc:
        raise ModelError(
            f"FireRed worker verification failed: {exc}"
        ) from exc
    _require_released(before, gpu_sampler("cuda:0"), "firered_worker")
    checks.append("firered_worker: verified")

    speaker_owner_factory = speaker_factory or (
        lambda: SpeakerDiarizer.from_config(config)
    )
    before = gpu_capacity_check("cuda:0")
    try:
        with speaker_owner_factory():
            pass
    except Exception as exc:
        raise ModelError(f"speaker model verification failed: {exc}") from exc
    _require_released(before, gpu_sampler("cuda:0"), "speaker_models")
    checks.append("speaker_models: verified")

    if config.overlap.enabled:
        verify_overlap = overlap_verifier or (
            lambda: _default_overlap_verifier(config)
        )
        before = gpu_capacity_check("cuda:0")
        try:
            verify_overlap()
        except Exception as exc:
            raise ModelError(
                f"overlap model verification failed: {exc}"
            ) from exc
        _require_released(before, gpu_sampler("cuda:0"), "overlap_models")
        checks.append("overlap_models: verified")

    if smoke_test is not None:
        smoke_test()
        checks.append("smoke_test: ok")
    checks.append("gpu_release: verified")
    return tuple(checks)
