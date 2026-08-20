"""Linux runtime preflight checks."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

from .config import AppConfig
from .errors import EnvironmentError as TranscriberEnvironmentError
from .errors import ModelError
from .model_registry import verify_models


def run_doctor(
    config: AppConfig,
    check_cuda: bool = True,
    smoke_test: Callable[[], None] | None = None,
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
            "missing models: " + ", ".join(verification.missing)
        )
    checks.append("models: verified")

    if check_cuda:
        try:
            import torch
        except ImportError as exc:
            raise TranscriberEnvironmentError(
                "PyTorch is not installed"
            ) from exc
        if not torch.cuda.is_available():
            raise TranscriberEnvironmentError("CUDA is unavailable in PyTorch")
        checks.append(f"cuda: {torch.cuda.get_device_name(0)}")
    if smoke_test is not None:
        smoke_test()
        checks.append("smoke_test: ok")
    return tuple(checks)
