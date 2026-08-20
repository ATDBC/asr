"""Command-line entry point for standalone transcription."""

from __future__ import annotations

import argparse
from pathlib import Path
import logging
import shutil
import sys
from typing import Callable, Sequence
import uuid

from .alignment import LocalRedecoder
from .asr import FunAsrEngine
from .audio import FfmpegAudioSlicer, normalize_audio, probe_audio
from .config import load_config
from .diagnostics import export_diagnostics
from .doctor import run_doctor
from .errors import OutputError, TranscriberError
from .exporters import export_transcript, publish_output_directory
from .locking import TaskLock
from .model_registry import (
    build_asr_model,
    download_models,
    verify_models,
)
from .overlap import JsonOverlapWorkerClient, SubprocessOverlapEnhancer
from .pipeline import TranscriptionPipeline
from .speaker_diarization import SpeakerDiarizer


def default_output_dir(input_path: Path) -> Path:
    source = Path(input_path)
    return source.with_name(f"{source.stem}_transcript")


def staging_output_dir(output_dir: Path, request_id: str) -> Path:
    target = Path(output_dir)
    return target.with_name(f".{target.name}.staging-{request_id}")


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
    parser.add_argument("--disable-overlap", action="store_true")
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


def _run_transcribe(args: argparse.Namespace) -> int:
    overrides = {}
    if args.device:
        overrides["general.device"] = args.device
    config = load_config(args.config, cli_overrides=overrides)
    input_path = args.input_audio
    output_dir = args.output_dir or default_output_dir(input_path)
    if not args.overwrite and output_dir.exists():
        raise OutputError(f"output already exists: {output_dir}")

    config.paths.work_dir.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    job_dir = config.paths.work_dir / f"job-{request_id}"
    job_dir.mkdir(parents=True, exist_ok=False)
    staging_dir = staging_output_dir(output_dir, request_id)
    log_path = job_dir / "transcribe.log"
    logger = logging.getLogger(f"funasr_transcriber.job.{job_dir.name}")
    logger.setLevel(getattr(logging, args.log_level))
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    def progress(message: str) -> None:
        print(message, file=sys.stderr)
        logger.info(message)

    try:
        with TaskLock(config.paths.work_dir, input_path):
            progress("[1/7] probing and normalizing audio")
            input_info = probe_audio(
                input_path,
                config.general.max_audio_seconds,
            )
            normalized = job_dir / "normalized.wav"
            normalize_audio(
                input_path,
                normalized,
                max_seconds=config.general.max_audio_seconds,
            )

            progress("[2/7] loading ASR and diarization models")
            asr_engine = FunAsrEngine(
                build_asr_model(
                    config.paths.model_dir,
                    config.general.device,
                )
            )
            diarizer = SpeakerDiarizer.from_config(config)
            local_redecoder = LocalRedecoder(
                asr_engine,
                FfmpegAudioSlicer(),
                job_dir / "redecode",
            )
            overlap_enhancer = None
            if config.overlap.enabled and not args.disable_overlap:
                worker = JsonOverlapWorkerClient(
                    worker_python=config.overlap.worker_python,
                    worker_script=Path(__file__).with_name("overlap_worker.py"),
                    work_dir=job_dir / "overlap-requests",
                    timeout_seconds=config.overlap.timeout_seconds,
                )
                overlap_enhancer = SubprocessOverlapEnhancer(
                    worker=worker,
                    asr_engine=asr_engine,
                    diarizer=diarizer,
                    min_score=config.overlap.speaker_match_min_score,
                    min_margin=config.overlap.speaker_match_min_margin,
                    min_rms=config.overlap.track_min_rms,
                    detector_model=config.overlap.detector_model,
                    hf_token_env=config.overlap.hf_token_env,
                    device=config.general.device,
                    output_dir=job_dir / "overlap-output",
                    cache_dir=config.paths.model_dir / "huggingface",
                )
            pipeline = TranscriptionPipeline(
                config,
                asr_engine,
                diarizer,
                local_redecoder=local_redecoder,
                overlap_enhancer=overlap_enhancer,
            )
            progress("[3/7] transcribing and diarizing")
            result = pipeline.run_normalized(normalized)
            progress("[7/7] exporting transcript")
            export_transcript(
                result.transcript,
                staging_dir,
                overwrite=False,
                input_info=input_info,
            )
            export_diagnostics(
                result.diagnostics,
                staging_dir,
                overwrite=False,
            )
            handler.flush()
            shutil.copy2(log_path, staging_dir / "transcribe.log")
            try:
                publish_output_directory(
                    staging_dir,
                    output_dir,
                    overwrite=args.overwrite,
                )
            except OSError as exc:
                raise OutputError(f"unable to publish output: {exc}") from exc
    finally:
        logger.removeHandler(handler)
        handler.close()
        shutil.rmtree(staging_dir, ignore_errors=True)
        if not args.keep_intermediate:
            shutil.rmtree(job_dir, ignore_errors=True)
    return 0


def _run_models(action: str, args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if action == "download":
        download_models(config.paths.model_dir)
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
    def smoke_test() -> None:
        build_asr_model(config.paths.model_dir, config.general.device)
        SpeakerDiarizer.from_config(config)

    checks = list(
        run_doctor(
            config,
            check_cuda=not args.skip_cuda,
            smoke_test=smoke_test,
        )
    )
    if config.overlap.enabled:
        worker = JsonOverlapWorkerClient(
            worker_python=config.overlap.worker_python,
            worker_script=Path(__file__).with_name("overlap_worker.py"),
            work_dir=config.paths.work_dir / "doctor-worker",
            timeout_seconds=config.overlap.timeout_seconds,
        )
        try:
            worker.call(
                "verify",
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

            raise ModelError(f"overlap model verification failed: {exc}") from exc
        checks.append("overlap_models: verified")
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
