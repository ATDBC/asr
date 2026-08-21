"""Persistent local owners and parallel normal-model execution."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import shutil
import socketserver
import sys
import threading
from typing import Any, Callable

from .audio import normalize_audio, probe_audio
from .config import AppConfig, load_config
from .diagnostics import export_diagnostics
from .errors import OutputError, TranscriberError, TranscriptionError
from .exporters import export_transcript, publish_output_directory
from .gpu import require_concurrent_gpu_capacity
from .pipeline import TranscriptionPipeline, attribute_primary
from .service_protocol import (
    ServiceBusyError,
    ServicePaths,
    ServiceProtocolError,
    ServiceRequest,
    ServiceState,
    ServiceStateStore,
    config_fingerprint,
    encode_message,
    parse_request,
    read_message,
)
from .schemas import AsrResult, SpeakerSegment


def _open_owner(owner: Any) -> Any:
    start = getattr(owner, "start", None)
    if callable(start):
        return start()
    return owner


def _create_and_open(factory: Callable[[], Any]) -> Any:
    return _open_owner(factory())


def _close_owner(owner: Any) -> None:
    close = getattr(owner, "close", None)
    if callable(close):
        close()


class HotModelHost:
    """Own one hot FireRed client and one hot 3D-Speaker diarizer."""

    def __init__(
        self,
        asr_factory: Callable[[], Any],
        diarizer_factory: Callable[[], Any],
        gpu_check: Callable[[], Any],
    ):
        self._asr_factory = asr_factory
        self._diarizer_factory = diarizer_factory
        self._gpu_check = gpu_check
        self.asr_owner: Any | None = None
        self.speaker_owner: Any | None = None
        self.inventories: dict[str, object] = {}

    @property
    def started(self) -> bool:
        return self.asr_owner is not None and self.speaker_owner is not None

    def start(self) -> "HotModelHost":
        if self.started:
            return self
        if self.asr_owner is not None or self.speaker_owner is not None:
            self.close()
        self._gpu_check()
        asr_owner: Any | None = None
        speaker_owner: Any | None = None
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                asr_future = executor.submit(
                    _create_and_open,
                    self._asr_factory,
                )
                speaker_future = executor.submit(
                    _create_and_open,
                    self._diarizer_factory,
                )
                asr_owner = asr_future.result()
                speaker_owner = speaker_future.result()
            status = getattr(asr_owner, "status", None)
            if callable(status):
                self.inventories["firered"] = status()
            inventory = getattr(speaker_owner, "resource_inventory", None)
            if callable(inventory):
                self.inventories["speaker"] = inventory()
        except Exception:
            _close_owner(asr_owner)
            _close_owner(speaker_owner)
            raise
        self.asr_owner = asr_owner
        self.speaker_owner = speaker_owner
        return self

    def close(self) -> None:
        asr_owner, speaker_owner = self.asr_owner, self.speaker_owner
        self.asr_owner = None
        self.speaker_owner = None
        self.inventories = {}
        try:
            _close_owner(asr_owner)
        finally:
            _close_owner(speaker_owner)


@dataclass(frozen=True, slots=True)
class NormalBranchResult:
    asr_result: AsrResult
    speaker_segments: tuple[SpeakerSegment, ...]
    speaker_metrics: dict[str, object]


def _run_speaker(
    speaker_owner: Any,
    normalized: Path,
    profile_enabled: bool,
) -> tuple[tuple[SpeakerSegment, ...], dict[str, object]]:
    profiled = getattr(speaker_owner, "diarize_with_resource_metrics", None)
    if profile_enabled and callable(profiled):
        segments, metrics = profiled(normalized)
        return tuple(segments), dict(metrics)
    return tuple(speaker_owner.diarize(normalized)), {}


def run_normal_branches(
    normalized: Path,
    asr_owner: Any,
    speaker_owner: Any,
    profile_enabled: bool = False,
) -> NormalBranchResult:
    """Start ASR and diarization before waiting for either result."""

    with ThreadPoolExecutor(max_workers=2) as executor:
        asr_future = executor.submit(asr_owner.transcribe, Path(normalized))
        speaker_future = executor.submit(
            _run_speaker,
            speaker_owner,
            Path(normalized),
            profile_enabled,
        )
        asr_result: AsrResult | None = None
        speaker_result: tuple[tuple[SpeakerSegment, ...], dict[str, object]] | None = None
        asr_error: Exception | None = None
        speaker_error: Exception | None = None
        try:
            asr_result = asr_future.result()
        except Exception as exc:
            asr_error = exc
        try:
            speaker_result = speaker_future.result()
        except Exception as exc:
            speaker_error = exc
    if asr_error is not None:
        raise asr_error
    if speaker_error is not None:
        raise speaker_error
    if asr_result is None or speaker_result is None:
        raise RuntimeError("normal model branches returned no result")
    speaker_segments, speaker_metrics = speaker_result
    return NormalBranchResult(
        asr_result=asr_result,
        speaker_segments=speaker_segments,
        speaker_metrics=speaker_metrics,
    )


@dataclass(slots=True)
class _QueuedJob:
    request: ServiceRequest
    future: Future[dict[str, object]]


class DaemonService:
    """Serialize accepted service jobs and drain them on shutdown.

    Network handlers only submit work to this object.  The single consumer is
    deliberately the only caller of ``run_job`` so model ownership remains
    confined to one daemon thread.
    """

    def __init__(
        self,
        queue_limit: int,
        run_job: Callable[[ServiceRequest], dict[str, object]],
    ):
        if queue_limit <= 0:
            raise ValueError("service queue_limit must be positive")
        self._queue: queue.Queue[_QueuedJob] = queue.Queue(maxsize=queue_limit)
        self._run_job = run_job
        self._futures: dict[str, Future[dict[str, object]]] = {}
        self._lock = threading.Lock()
        self._draining = False
        self._started = False
        self._stopped = threading.Event()
        self._consumer: threading.Thread | None = None
        self.completed_ids: list[str] = []

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            if self._draining:
                self._stopped.set()
                return
            self._started = True
            self._consumer = threading.Thread(
                target=self._consume,
                name="funasr-transcriber-service",
                daemon=True,
            )
            self._consumer.start()

    def submit(self, request: ServiceRequest) -> Future[dict[str, object]]:
        with self._lock:
            existing = self._futures.get(request.identifier)
            if existing is not None:
                return existing
            if self._draining:
                raise ServiceBusyError("service is draining and accepts no new jobs")
            if not self._started:
                raise RuntimeError("service has not started")
            future: Future[dict[str, object]] = Future()
            try:
                self._queue.put_nowait(_QueuedJob(request, future))
            except queue.Full as exc:
                raise ServiceBusyError("service queue is full") from exc
            self._futures[request.identifier] = future
            return future

    def request_stop(self) -> None:
        """Reject new jobs, then let the sole consumer finish accepted work."""

        with self._lock:
            self._draining = True
            if not self._started:
                self._stopped.set()

    def wait_stopped(self, timeout: float | None = None) -> bool:
        return self._stopped.wait(timeout)

    def _consume(self) -> None:
        try:
            while True:
                try:
                    queued = self._queue.get(timeout=0.05)
                except queue.Empty:
                    with self._lock:
                        if self._draining:
                            return
                    continue
                try:
                    result = self._run_job(queued.request)
                except BaseException as exc:
                    queued.future.set_exception(exc)
                else:
                    self.completed_ids.append(queued.request.identifier)
                    queued.future.set_result(result)
                finally:
                    self._queue.task_done()
        finally:
            self._stopped.set()


class _ThreadingUnixStreamServer(
    socketserver.ThreadingMixIn,
    socketserver.UnixStreamServer,
):
    daemon_threads = True


class _ServiceRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        host: ServiceSocketServer = self.server.service_host  # type: ignore[attr-defined]
        identifier = ""
        try:
            request = parse_request(read_message(self.rfile))
            identifier = request.identifier
            if request.action == "health":
                payload: dict[str, object] = {
                    "id": identifier,
                    "ok": True,
                    "terminal": True,
                    "status": "ready",
                }
            elif request.action == "submit":
                if request.config_fingerprint != host.config_fingerprint:
                    raise ServiceProtocolError(
                        "service configuration is incompatible; stop it first"
                    )
                result = host.daemon.submit(request).result()
                payload = {
                    "id": identifier,
                    "ok": True,
                    "terminal": True,
                    "result": result,
                }
            else:
                host.daemon.request_stop()
                payload = {
                    "id": identifier,
                    "ok": True,
                    "terminal": True,
                    "status": "draining",
                }
                host.watch_for_stop()
        except Exception as exc:
            payload = {
                "id": identifier,
                "ok": False,
                "terminal": True,
                "error": str(exc),
            }
        self.wfile.write(encode_message(payload))
        self.wfile.flush()


class ServiceSocketServer:
    """Threaded Unix-socket front end for a single ``DaemonService`` owner."""

    def __init__(
        self,
        paths: ServicePaths,
        daemon: DaemonService,
        config_fingerprint: str,
        remove_socket_on_close: bool = True,
    ):
        self.paths = paths
        self.daemon = daemon
        self.config_fingerprint = str(config_fingerprint)
        self._remove_socket_on_close = remove_socket_on_close
        self._server: _ThreadingUnixStreamServer | None = None
        self._thread: threading.Thread | None = None
        self._watcher: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False

    def start(self) -> None:
        with self._lock:
            if self._server is not None:
                return
            if self._closed:
                raise RuntimeError("service socket server is closed")
            self.paths.directory.mkdir(parents=True, exist_ok=True)
            if self.paths.socket.exists():
                raise ServiceProtocolError(
                    f"service socket already exists: {self.paths.socket}"
                )
            self.daemon.start()
            server = _ThreadingUnixStreamServer(
                str(self.paths.socket),
                _ServiceRequestHandler,
            )
            server.service_host = self  # type: ignore[attr-defined]
            self._server = server
            self._thread = threading.Thread(
                target=server.serve_forever,
                name="funasr-transcriber-socket",
                daemon=True,
            )
            self._thread.start()

    def watch_for_stop(self) -> None:
        with self._lock:
            if self._watcher is not None:
                return
            self._watcher = threading.Thread(
                target=self._close_after_drain,
                name="funasr-transcriber-stop",
                daemon=True,
            )
            self._watcher.start()

    def _close_after_drain(self) -> None:
        self.daemon.wait_stopped()
        self.close()

    def close(self) -> None:
        self.daemon.request_stop()
        self.daemon.wait_stopped()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            server = self._server
            thread = self._thread
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        if self._remove_socket_on_close:
            self.paths.socket.unlink(missing_ok=True)


class ServiceJobRunner:
    """Run one queued request against the daemon's resident normal owners."""

    def __init__(
        self,
        config: Any,
        hot_host: HotModelHost,
        probe: Callable[..., Any] = probe_audio,
        normalize: Callable[..., Any] = normalize_audio,
    ):
        self.config = config
        self.hot_host = hot_host
        self._probe = probe
        self._normalize = normalize

    @staticmethod
    def _staging_output_dir(output_dir: Path, request_id: str) -> Path:
        return output_dir.with_name(f".{output_dir.name}.staging-{request_id}")

    def run(self, request: ServiceRequest) -> dict[str, object]:
        if request.action != "submit":
            raise ValueError("service job runner requires a submit request")
        if request.input_audio is None or request.output_dir is None:
            raise ValueError("submit request is missing audio or output path")
        input_audio = Path(request.input_audio).expanduser().resolve()
        output_dir = Path(request.output_dir).expanduser().resolve()
        if output_dir.exists() and not request.overwrite:
            raise OutputError(f"output already exists: {output_dir}")
        work_dir = Path(self.config.paths.work_dir).expanduser().resolve()
        job_dir = work_dir / f"job-{request.identifier}"
        staging_dir = self._staging_output_dir(output_dir, request.identifier)
        if job_dir.exists():
            raise OutputError(f"service job directory already exists: {job_dir}")
        job_dir.mkdir(parents=True, exist_ok=False)
        try:
            input_info = self._probe(
                input_audio,
                self.config.general.max_audio_seconds,
            )
            normalized = job_dir / "normalized.wav"
            self._normalize(
                input_audio,
                normalized,
                self.config.general.max_audio_seconds,
            )
            if not self.hot_host.started:
                self.hot_host.start()
            asr_owner = self.hot_host.asr_owner
            speaker_owner = self.hot_host.speaker_owner
            if asr_owner is None or speaker_owner is None:
                raise TranscriptionError("resident model owners are unavailable")
            pipeline = TranscriptionPipeline(self.config)
            diagnostics = pipeline.new_diagnostics()
            with diagnostics.stage("normal_model_branches"):
                branches = run_normal_branches(
                    normalized,
                    asr_owner,
                    speaker_owner,
                    profile_enabled=request.profile_resources,
                )
            if not branches.speaker_segments:
                raise TranscriptionError("speaker diarization returned no segments")
            with diagnostics.stage("alignment"):
                attributed = attribute_primary(
                    branches.asr_result,
                    branches.speaker_segments,
                    self.config,
                )
            result = pipeline.finish(
                branches.asr_result,
                attributed,
                branches.speaker_segments,
                diagnostics,
            )
            export_transcript(
                result.transcript,
                staging_dir,
                overwrite=False,
                input_info=input_info,
            )
            export_diagnostics(result.diagnostics, staging_dir, overwrite=False)
            try:
                published = publish_output_directory(
                    staging_dir,
                    output_dir,
                    overwrite=request.overwrite,
                )
            except OSError as exc:
                raise OutputError(f"unable to publish output: {exc}") from exc
            return {"output_dir": str(published)}
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
            if not request.keep_intermediate:
                shutil.rmtree(job_dir, ignore_errors=True)


def _require_physical_gpu_zero(config: AppConfig) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    first_visible = visible.split(",", 1)[0].strip() if visible else "0"
    if config.general.device != "cuda:0" or first_visible != "0":
        raise ServiceProtocolError(
            "service is restricted to physical GPU 0; "
            f"device={config.general.device}, CUDA_VISIBLE_DEVICES={visible!r}"
        )


def build_hot_model_host(config: AppConfig, paths: ServicePaths) -> HotModelHost:
    """Construct the only persistent FireRed and 3D-Speaker owners."""

    from .firered_client import FireRedWorkerClient
    from .speaker_diarization import SpeakerDiarizer

    def asr_factory() -> FireRedWorkerClient:
        return FireRedWorkerClient(
            worker_python=config.asr.worker_python,
            model_root=config.paths.model_dir,
            device="cuda:0",
            timeout_seconds=config.asr.timeout_seconds,
            stderr_log_path=paths.directory / "firered-worker.log",
        )

    def gpu_check() -> None:
        require_concurrent_gpu_capacity(
            "cuda:0",
            minimum_total_mib=config.service.minimum_total_mib,
            required_free_mib=config.service.minimum_free_mib,
        )

    return HotModelHost(
        asr_factory=asr_factory,
        diarizer_factory=lambda: SpeakerDiarizer.from_config(config),
        gpu_check=gpu_check,
    )


def run_daemon(config: AppConfig) -> None:
    """Own resident models until the local service has drained and stopped."""

    _require_physical_gpu_zero(config)
    paths = ServicePaths.from_work_dir(config.paths.work_dir)
    fingerprint = config_fingerprint(config)
    state_store = ServiceStateStore(paths)
    existing = state_store.require_compatible(fingerprint)
    if existing is not None:
        raise ServiceProtocolError(
            f"a compatible local service is already running: pid={existing.pid}"
        )
    hot_host: HotModelHost | None = None
    server: ServiceSocketServer | None = None
    wrote_state = False
    try:
        hot_host = build_hot_model_host(config, paths)
        hot_host.start()
        runner = ServiceJobRunner(config, hot_host)
        daemon = DaemonService(config.service.queue_limit, runner.run)
        server = ServiceSocketServer(
            paths,
            daemon,
            fingerprint,
            remove_socket_on_close=False,
        )
        server.start()
        state_store.write(
            ServiceState(
                pid=os.getpid(),
                protocol_version=1,
                config_fingerprint=fingerprint,
            )
        )
        wrote_state = True
        daemon.wait_stopped()
    finally:
        if server is not None:
            server.close()
        if hot_host is not None:
            hot_host.close()
        if wrote_state:
            state = state_store.read()
            if state is not None and state.pid == os.getpid():
                paths.socket.unlink(missing_ok=True)
                paths.state.unlink(missing_ok=True)


def _daemon_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m funasr_transcriber.service")
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _daemon_parser()
    try:
        args = parser.parse_args(argv)
        run_daemon(load_config(args.config))
    except TranscriberError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
