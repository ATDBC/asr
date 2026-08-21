"""Command-line entry point for standalone transcription."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Sequence
import uuid

from .audio import normalize_audio, probe_audio
from .config import load_config
from .diagnostics import export_diagnostics
from .doctor import run_doctor
from .errors import EnvironmentError, OutputError, TranscriberError
from .exporters import export_transcript, publish_output_directory
from .firered_client import FireRedWorkerClient
from .gpu import build_gpu_stage_record, query_gpu, require_gpu_capacity
from .locking import ServiceStartupLock, TaskLock
from .model_registry import (
    acquire_overlap_assets,
    download_models,
    verify_models,
)
from .overlap import (
    JsonOverlapWorkerClient,
    OverlapEnhancement,
    OverlapPreparation,
    SubprocessOverlapPreparer,
    apply_prepared_overlap,
)
from .pipeline import TranscriptionPipeline, attribute_primary
from .speaker_diarization import SpeakerDiarizer
from .service_protocol import (
    ServiceClient,
    ServicePaths,
    ServiceProtocolError,
    ServiceRequest,
    config_fingerprint,
)


def default_output_dir(input_path: Path) -> Path:
    source = Path(input_path)
    return source.with_name(f"{source.stem}_transcript")


def staging_output_dir(output_dir: Path, request_id: str) -> Path:
    target = Path(output_dir)
    return target.with_name(f".{target.name}.staging-{request_id}")


def launch_service(config, paths: ServicePaths, config_path: Path | None) -> None:
    """Start only this project's daemon, pinned to physical GPU 0."""

    command = [sys.executable, "-m", "funasr_transcriber.service"]
    if config_path is not None:
        command.extend(["--config", str(Path(config_path).expanduser().resolve())])
    paths.directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    with paths.log.open("a", encoding="utf-8", buffering=1) as log:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            close_fds=True,
        )


def _service_client(config, paths: ServicePaths) -> ServiceClient:
    return ServiceClient(
        paths,
        timeout_seconds=config.service.startup_timeout_seconds,
    )


def _wait_for_service_ready(config, paths: ServicePaths) -> ServiceClient:
    deadline = time.monotonic() + config.service.startup_timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client = _service_client(config, paths)
        try:
            client.health(uuid.uuid4().hex)
            return client
        except ServiceProtocolError as exc:
            last_error = exc
            time.sleep(0.1)
    raise ServiceProtocolError(
        "local service did not become ready before startup timeout"
        + (f": {last_error}" if last_error is not None else "")
    )


def submit_to_service(args: argparse.Namespace, config) -> dict[str, object]:
    """Submit one CLI request, auto-starting a compatible local daemon."""

    input_audio = Path(args.input_audio).expanduser().resolve()
    output_dir = Path(
        args.output_dir or default_output_dir(input_audio)
    ).expanduser().resolve()
    if not args.overwrite and output_dir.exists():
        raise OutputError(f"output already exists: {output_dir}")
    paths = ServicePaths.from_work_dir(config.paths.work_dir)
    client = _service_client(config, paths)
    try:
        client.health(uuid.uuid4().hex)
    except ServiceProtocolError:
        with ServiceStartupLock(paths.directory):
            try:
                client = _service_client(config, paths)
                client.health(uuid.uuid4().hex)
            except ServiceProtocolError:
                launch_service(config, paths, getattr(args, "config", None))
                client = _wait_for_service_ready(config, paths)
    overlap_enabled = (
        True
        if getattr(args, "enable_overlap", False)
        else False
        if getattr(args, "disable_overlap", False)
        else config.overlap.enabled
    )
    request = ServiceRequest(
        identifier=uuid.uuid4().hex,
        action="submit",
        input_audio=input_audio,
        output_dir=output_dir,
        config_fingerprint=config_fingerprint(config),
        overwrite=bool(args.overwrite),
        keep_intermediate=bool(args.keep_intermediate),
        overlap_enabled=overlap_enabled,
        profile_resources=bool(getattr(args, "profile_resources", False)),
    )
    return client.submit_and_wait(request)


def require_physical_gpu_zero(device: str) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    first_visible = (
        visible.split(",", 1)[0].strip()
        if visible is not None
        else "0"
    )
    if device != "cuda:0" or first_visible != "0":
        raise EnvironmentError(
            "this deployment is restricted to physical GPU 0; "
            f"device={device}, CUDA_VISIBLE_DEVICES={visible!r}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="funasr-transcribe",
        description="Offline Mandarin transcription with speaker diarization",
    )
    parser.add_argument("input_audio", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    overlap_group = parser.add_mutually_exclusive_group()
    overlap_group.add_argument("--enable-overlap", action="store_true")
    overlap_group.add_argument("--disable-overlap", action="store_true")
    parser.add_argument("--profile-resources", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _models_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="funasr-transcribe models")
    parser.add_argument("action", choices=("download", "verify"))
    parser.add_argument("--config", type=Path)
    return parser


def _doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="funasr-transcribe doctor")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--skip-cuda", action="store_true")
    return parser


def _service_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="funasr-transcribe service")
    parser.add_argument("action", choices=("status", "stop"))
    parser.add_argument("--config", type=Path)
    return parser


def execute_model_phases(
    normalized: Path,
    config,
    overlap_enabled: bool,
    asr_client_factory: Callable[[str], Any],
    diarizer_factory: Callable[[], Any],
    overlap_preparer_factory: Callable[[Any], Any],
    gpu_check: Callable[[str], Any],
    gpu_after: Callable[[str, Any, Any, Any], None] | None = None,
    result_consumer: Callable[[Any], None] | None = None,
):
    """Run ASR, speaker, and optional overlap ASR with no GPU co-residency."""

    pipeline = TranscriptionPipeline(config)
    diagnostics = pipeline.new_diagnostics()

    primary_before = gpu_check("primary-asr")
    primary_owner = None
    with diagnostics.stage("asr"):
        with asr_client_factory("primary-asr") as primary_owner:
            asr_result = primary_owner.transcribe(Path(normalized))
    if gpu_after is not None:
        gpu_after("primary_asr", primary_before, primary_owner, diagnostics)

    speaker_before = gpu_check("speaker")
    speaker_owner = None
    preparation = OverlapPreparation((), ())
    with diagnostics.stage("diarization"):
        with diarizer_factory() as speaker_owner:
            speaker_segments = speaker_owner.diarize(Path(normalized))
            if not speaker_segments:
                from .errors import TranscriptionError

                raise TranscriptionError(
                    "speaker diarization returned no segments"
                )
            if overlap_enabled:
                with diagnostics.stage("overlap_preparation"):
                    preparation = overlap_preparer_factory(
                        speaker_owner
                    ).prepare(Path(normalized), speaker_segments)
    if gpu_after is not None:
        gpu_after("speaker", speaker_before, speaker_owner, diagnostics)

    with diagnostics.stage("alignment"):
        attributed = attribute_primary(asr_result, speaker_segments, config)

    enhancement: OverlapEnhancement | None = None
    if overlap_enabled:
        if any(window.tracks for window in preparation.windows):
            overlap_before = gpu_check("overlap-asr")
            overlap_owner = None
            with diagnostics.stage("overlap_asr"):
                with asr_client_factory("overlap-asr") as overlap_owner:
                    enhancement = apply_prepared_overlap(
                        preparation,
                        attributed,
                        overlap_owner,
                        min_score=config.overlap.speaker_match_min_score,
                        min_margin=config.overlap.speaker_match_min_margin,
                    )
            if gpu_after is not None:
                gpu_after(
                    "overlap_asr",
                    overlap_before,
                    overlap_owner,
                    diagnostics,
                )
        else:
            warnings = preparation.warnings + tuple(
                "overlap window "
                f"{window.id} kept original: track_count_not_two"
                for window in preparation.windows
            )
            enhancement = OverlapEnhancement(
                attributed,
                warnings,
                False,
            )
    result = pipeline.finish(
        asr_result,
        attributed,
        speaker_segments,
        diagnostics,
        overlap_enhancement=enhancement,
    )
    if result_consumer is not None:
        result_consumer(result)
    return result


def _run_transcribe(args: argparse.Namespace) -> int:
    overrides = {}
    if args.device:
        overrides["general.device"] = args.device
    config = load_config(args.config, cli_overrides=overrides)
    response = submit_to_service(args, config)
    result = response.get("result")
    if isinstance(result, dict) and isinstance(result.get("output_dir"), str):
        print(result["output_dir"])
    return 0


def _run_service(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    paths = ServicePaths.from_work_dir(config.paths.work_dir)
    client = _service_client(config, paths)
    identifier = uuid.uuid4().hex
    response = (
        client.health(identifier)
        if args.action == "status"
        else client.stop_and_wait(identifier)
    )
    print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    return 0


def _run_models(action: str, args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if action == "download":
        download_models(config.paths.model_dir)
        if config.overlap.enabled:
            acquire_overlap_assets(config.paths.model_dir)
    else:
        result = verify_models(config.paths.model_dir)
        if not result.ok:
            from .errors import ModelError

            raise ModelError("missing models: " + ", ".join(result.missing))
    if config.overlap.enabled:
        worker = JsonOverlapWorkerClient(
            worker_python=config.overlap.worker_python,
            worker_script=Path(__file__).with_name("overlap_worker.py"),
            work_dir=config.paths.work_dir / "model-worker",
            timeout_seconds=config.overlap.timeout_seconds,
        )
        try:
            worker.call(
                action,
                {
                    "model_id": config.overlap.detector_model,
                    "hf_token_env": config.overlap.hf_token_env,
                    "device": config.general.device,
                    "model": "MossFormer2_SS_16K",
                    "cache_dir": str(
                        (config.paths.model_dir / "huggingface").resolve()
                    ),
                },
            )
        except Exception as exc:
            from .errors import ModelError

            raise ModelError(f"overlap models {action} failed: {exc}") from exc
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if not args.skip_cuda:
        require_physical_gpu_zero(config.general.device)
    checks = list(
        run_doctor(
            config,
            check_cuda=not args.skip_cuda,
            smoke_test=None,
        )
    )
    for check in checks:
        print(check)
    return 0


def main(
    argv: Sequence[str] | None = None,
    transcribe_runner: Callable[[argparse.Namespace], int] | None = None,
    model_runner: Callable[[str, argparse.Namespace], int] | None = None,
    doctor_runner: Callable[[argparse.Namespace], int] | None = None,
) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "models":
        parser = _models_parser()
        try:
            args = parser.parse_args(arguments[1:])
        except SystemExit as exc:
            return int(exc.code)
        runner = model_runner or _run_models
        try:
            return runner(args.action, args)
        except TranscriberError as exc:
            print(str(exc), file=sys.stderr)
            return exc.exit_code
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if arguments and arguments[0] == "doctor":
        parser = _doctor_parser()
        try:
            args = parser.parse_args(arguments[1:])
        except SystemExit as exc:
            return int(exc.code)
        runner = doctor_runner or _run_doctor
        try:
            return runner(args)
        except TranscriberError as exc:
            print(str(exc), file=sys.stderr)
            return exc.exit_code
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if arguments and arguments[0] == "service":
        parser = _service_parser()
        try:
            args = parser.parse_args(arguments[1:])
        except SystemExit as exc:
            return int(exc.code)
        try:
            return _run_service(args)
        except TranscriberError as exc:
            print(str(exc), file=sys.stderr)
            return exc.exit_code
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    parser = _parser()
    try:
        args = parser.parse_args(arguments)
    except SystemExit as exc:
        return int(exc.code)
    if args.input_audio is None:
        parser.print_usage(sys.stderr)
        return 2
    runner = transcribe_runner or _run_transcribe
    try:
        return runner(args)
    except TranscriberError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
