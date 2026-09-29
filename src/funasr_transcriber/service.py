"""Persistent local owners and diarization-first normal-model execution."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import queue
import shutil
import socketserver
import sys
import threading
from time import perf_counter
from typing import Any, Callable

from .alignment import attribute_by_segment
from .audio import normalize_audio, probe_audio
from .config import AppConfig, load_config
from .crop_attribution import attribute_crop_results
from .speaker_crops import CropPlan, build_crop_plan
from .diagnostics import export_diagnostics
from .diagnostic_traces import compact_scheduler_summary, write_deep_traces
from .errors import (
    EnvironmentError as TranscriberEnvironmentError,
    OutputError,
    TranscriberError,
    TranscriptionError,
)
from .exporters import export_transcript, publish_output_directory
from .gpu import (
    build_gpu_stage_record,
    query_gpu,
    require_gpu_capacity,
)
from .overlap import (
    JsonOverlapWorkerClient,
    OverlapEnhancement,
    OverlapPreparation,
    SubprocessOverlapPreparer,
    apply_prepared_overlap,
    extract_overlap_windows_from_segments,
)
from .pipeline import TranscriptionPipeline
from .resource_profile import ResourceSampler
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
from .weighted_aed_calibration import (
    build_expected_runtime_identity,
    load_weight_artifact,
)


LOGGER = logging.getLogger(__name__)


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
    """Own the hot FireRed AED owner and one 3D-Speaker diarizer."""

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
        self._warmed = False
        self.warmup_metrics: dict[str, object] = {
            "performed": False,
            "excluded_from_job_timing": False,
            "reason": "not_performed",
        }

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

    def warmup(self) -> dict[str, object]:
        if not self.started or self.asr_owner is None:
            raise TranscriptionError("resident model owners are unavailable")
        if self._warmed:
            return {
                "performed": False,
                "excluded_from_job_timing": False,
                "reason": "already_performed",
            }
        warmup = getattr(self.asr_owner, "warmup", None)
        if not callable(warmup):
            raise TranscriptionError("resident FireRed owner cannot warm up")
        warmup_result = warmup()
        if not isinstance(warmup_result, dict):
            raise TranscriptionError(
                "resident FireRed owner returned invalid warmup metrics"
            )
        self._warmed = True
        self.warmup_metrics = {
            **warmup_result,
            "performed": True,
            "excluded_from_job_timing": False,
        }
        return dict(self.warmup_metrics)

    def close(self) -> None:
        asr_owner, speaker_owner = self.asr_owner, self.speaker_owner
        self.asr_owner = None
        self.speaker_owner = None
        self.inventories = {}
        self._warmed = False
        self.warmup_metrics = {
            "performed": False,
            "excluded_from_job_timing": False,
            "reason": "not_performed",
        }
        try:
            _close_owner(asr_owner)
        finally:
            _close_owner(speaker_owner)


@dataclass(frozen=True, slots=True)
class NormalBranchResult:
    asr_result: AsrResult
    speaker_segments: tuple[SpeakerSegment, ...]
    speaker_metrics: dict[str, object]
    branch_metrics: dict[str, dict[str, object]]


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
    """Run diarization first, then transcribe each speaker segment as a VAD segment."""

    branch_metrics: dict[str, dict[str, object]] = {}

    def measure(name: str, operation: Callable[..., Any], *args: Any) -> Any:
        started_at = perf_counter()
        try:
            return operation(*args)
        finally:
            ended_at = perf_counter()
            branch_metrics[name] = {
                "started_at_monotonic": started_at,
                "ended_at_monotonic": ended_at,
                "seconds": ended_at - started_at,
            }

    speaker_segments, speaker_metrics = measure(
        "diarization",
        _run_speaker,
        speaker_owner,
        Path(normalized),
        profile_enabled,
    )
    if speaker_metrics:
        branch_metrics["diarization"]["memory"] = speaker_metrics
    asr_result: AsrResult = measure(
        "asr",
        asr_owner.transcribe,
        Path(normalized),
        [[segment.start_ms, segment.end_ms] for segment in speaker_segments],
    )
    asr_response = getattr(asr_owner, "last_response", {})
    if isinstance(asr_response, dict):
        metrics = asr_response.get("metrics")
        if isinstance(metrics, dict):
            if isinstance(metrics.get("scheduler"), dict):
                branch_metrics["asr"]["scheduler"] = metrics["scheduler"]
            if isinstance(metrics.get("parallel"), dict):
                branch_metrics["asr"]["parallel"] = metrics["parallel"]
    return NormalBranchResult(
        asr_result=asr_result,
        speaker_segments=speaker_segments,
        speaker_metrics=speaker_metrics,
        branch_metrics=branch_metrics,
    )


def run_nemotron_branches(
    normalized: Path,
    asr_owner: Any,
    speaker_owner: Any,
    crop_dir: Path,
    profile_enabled: bool = False,
    overlap_preparation: OverlapPreparation | None = None,
    min_score: float = 0.5,
    min_margin: float = 0.05,
    min_rms: float = 0.005,
) -> tuple[NormalBranchResult, tuple[Any, ...], CropPlan]:
    """Keep Nemotron speaker intervals and AED crop paths independent."""
    branch_metrics: dict[str, dict[str, object]] = {}
    speaker_started = perf_counter()
    segments, speaker_metrics = _run_speaker(
        speaker_owner, Path(normalized), profile_enabled
    )
    branch_metrics["diarization"] = {
        "seconds": perf_counter() - speaker_started,
    }
    if speaker_metrics:
        branch_metrics["diarization"]["memory"] = speaker_metrics
    if not segments:
        raise TranscriptionError(
            "Nemotron speaker diarization returned no segments"
        )
    crop_started = perf_counter()
    worker0 = getattr(asr_owner, "worker0", None)
    refine_vad = getattr(worker0, "refine_vad", None)
    crop_plan = build_crop_plan(
        Path(normalized),
        segments,
        crop_dir,
        refine_vad if callable(refine_vad) else None,
        overlap_preparation=overlap_preparation,
        min_score=min_score,
        min_margin=min_margin,
        min_rms=min_rms,
    )
    branch_metrics["crops"] = {
        "seconds": perf_counter() - crop_started,
        "count": len(crop_plan.crops),
    }
    transcribe_crops = getattr(asr_owner, "transcribe_crops", None)
    if not callable(transcribe_crops):
        raise TranscriptionError(
            "resident FireRed owner does not support independent AED crops"
        )
    asr_started = perf_counter()
    crop_results = transcribe_crops(crop_plan)
    asr_result, attributed = attribute_crop_results(
        tuple(crop_results), segments, crop_plan.duration_ms
    )
    branch_metrics["asr"] = {
        "seconds": perf_counter() - asr_started,
    }
    response = getattr(asr_owner, "last_response", {})
    if isinstance(response, dict):
        metrics = response.get("metrics")
        if isinstance(metrics, dict):
            for key in ("scheduler", "parallel"):
                if isinstance(metrics.get(key), dict):
                    branch_metrics["asr"][key] = metrics[key]
    return (
        NormalBranchResult(
            asr_result,
            segments,
            speaker_metrics,
            branch_metrics,
        ),
        attributed,
        crop_plan,
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
        self._pending_count = 0
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
        future, _ = self.accept(request)
        return future

    def accept(self, request: ServiceRequest) -> tuple[Future[dict[str, object]], int]:
        """Accept one FIFO job and return its future plus current work depth."""

        with self._lock:
            existing = self._futures.get(request.identifier)
            if existing is not None:
                return existing, self._pending_count
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
            self._pending_count += 1
            return future, self._pending_count

    def result_for(self, job_id: str) -> Future[dict[str, object]]:
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            raise ServiceProtocolError(f"unknown service job: {job_id}")
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
                    with self._lock:
                        self._pending_count -= 1
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
                _, queue_depth = host.daemon.accept(request)
                payload = {
                    "id": identifier,
                    "ok": True,
                    "terminal": False,
                    "status": "accepted",
                    "job_id": request.identifier,
                    "queue_depth": queue_depth,
                }
            elif request.action == "wait":
                if request.job_id is None:
                    raise ServiceProtocolError("wait job_id must not be empty")
                result = host.daemon.result_for(request.job_id).result()
                payload = {
                    "id": identifier,
                    "ok": True,
                    "terminal": True,
                    "status": "completed",
                    "job_id": request.job_id,
                    "result": result,
                }
            elif request.action == "stop":
                host.daemon.request_stop()
                payload = {
                    "id": identifier,
                    "ok": True,
                    "terminal": True,
                    "status": "draining",
                }
                host.watch_for_stop()
            else:
                raise ServiceProtocolError(
                    f"unsupported service action: {request.action}"
                )
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
            if not self.paths.uses_abstract_socket and self.paths.socket.exists():
                raise ServiceProtocolError(
                    f"service socket already exists: {self.paths.socket}"
                )
            self.daemon.start()
            server = _ThreadingUnixStreamServer(
                self.paths.socket_address,
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
        if self._remove_socket_on_close and not self.paths.uses_abstract_socket:
            self.paths.socket.unlink(missing_ok=True)


class ServiceJobRunner:
    """Run one queued request against the daemon's resident normal owners."""

    def __init__(
        self,
        config: Any,
        hot_host: HotModelHost,
        probe: Callable[..., Any] = probe_audio,
        normalize: Callable[..., Any] = normalize_audio,
        overlap_factory: Callable[[Path], Any] | None = None,
        temporary_diarizer_factory: Callable[[], Any] | None = None,
        temporary_asr_factory: Callable[[Path], Any] | None = None,
        gpu_sampler: Callable[[str], Any] | None = query_gpu,
        sampler_factory: Callable[..., Any] = ResourceSampler,
    ):
        self.config = config
        self.hot_host = hot_host
        self._probe = probe
        self._normalize = normalize
        self._overlap_factory = overlap_factory or self._default_overlap_factory
        self._temporary_diarizer_factory = (
            temporary_diarizer_factory or self._default_temporary_diarizer
        )
        self._temporary_asr_factory = (
            temporary_asr_factory or self._default_temporary_asr
        )
        self._gpu_sampler = gpu_sampler
        self._sampler_factory = sampler_factory
        self._active_overlap_pid: int | None = None
        self._capture_overlap_pid = False

    @staticmethod
    def _staging_output_dir(output_dir: Path, request_id: str) -> Path:
        return output_dir.with_name(f".{output_dir.name}.staging-{request_id}")

    def _default_overlap_factory(self, job_dir: Path) -> SubprocessOverlapPreparer:
        worker = JsonOverlapWorkerClient(
            worker_python=self.config.overlap.worker_python,
            worker_script=Path(__file__).with_name("overlap_worker.py"),
            work_dir=job_dir / "overlap-requests",
            timeout_seconds=self.config.overlap.timeout_seconds,
        )
        return SubprocessOverlapPreparer(
            worker=worker,
            diarizer=None,
            min_score=self.config.overlap.speaker_match_min_score,
            min_margin=self.config.overlap.speaker_match_min_margin,
            min_rms=self.config.overlap.track_min_rms,
            detector_model=self.config.overlap.detector_model,
            hf_token_env=self.config.overlap.hf_token_env,
            device="cuda:0",
            output_dir=job_dir / "overlap-output",
            cache_dir=Path(self.config.paths.model_dir) / "huggingface",
            on_worker_started=(
                self._record_overlap_started
                if self._capture_overlap_pid
                else None
            ),
            on_worker_finished=(
                self._record_overlap_finished
                if self._capture_overlap_pid
                else None
            ),
        )

    def _record_overlap_started(self, pid: int) -> None:
        self._active_overlap_pid = int(pid)

    def _record_overlap_finished(self, pid: int) -> None:
        if self._active_overlap_pid == int(pid):
            self._active_overlap_pid = None

    def _default_temporary_diarizer(self) -> Any:
        from .speaker_diarization import SpeakerDiarizer

        return SpeakerDiarizer.from_config(self.config)

    def _default_temporary_asr(self, job_dir: Path) -> Any:
        from .firered_client import FireRedWorkerClient

        return FireRedWorkerClient(
            worker_python=self.config.asr.worker_python,
            model_root=self.config.paths.model_dir,
            device="cuda:0",
            timeout_seconds=self.config.asr.timeout_seconds,
            stderr_log_path=job_dir / "overlap-firered-worker.log",
            use_half=self.config.asr.use_half,
            warmup_audio=(
                Path(self.config.paths.model_dir)
                / "fireredasr2s"
                / "source"
                / "assets"
                / "hello_zh.wav"
            ),
        )

    def _owned_pid_getters(self) -> dict[str, Callable[[], int | None]]:
        def service_pid() -> int:
            return os.getpid()

        def speaker_pid() -> int:
            owner = self.hot_host.speaker_owner
            pid = getattr(owner, "pid", None)
            return pid if isinstance(pid, int) and pid > 0 else os.getpid()

        def aed_pid(index: int) -> int | None:
            owner = self.hot_host.asr_owner
            pids = getattr(owner, "pids", ())
            if index < len(pids):
                pid = pids[index]
            else:
                pid = getattr(owner, "pid", None) if index == 0 else None
            return pid if isinstance(pid, int) and pid > 0 else None

        def punc_pid() -> int | None:
            owner = self.hot_host.asr_owner
            punc_owner = getattr(owner, "punc_owner", None)
            pid = getattr(punc_owner, "pid", None)
            if pid is None and punc_owner is None:
                pid = getattr(owner, "pid", None)
            return pid if isinstance(pid, int) and pid > 0 else None

        return {
            "service": service_pid,
            "speaker": speaker_pid,
            "aed_worker_0": lambda: aed_pid(0),
            "aed_worker_1": lambda: aed_pid(1),
            "punc_worker": punc_pid,
            "overlap_worker": lambda: self._active_overlap_pid,
        }

    @staticmethod
    def _diagnostics_level(request: ServiceRequest) -> str:
        level = str(getattr(request, "diagnostics_level", "basic")).lower()
        if request.profile_resources and level == "basic":
            return "resources"
        return level

    def _start_resource_sampler(
        self,
        request: ServiceRequest,
        diagnostics_level: str,
    ) -> Any | None:
        if diagnostics_level not in {"resources", "deep"}:
            return None
        sampler = self._sampler_factory(
            "cuda:0",
            self.config.diagnostics.sampling_interval_ms,
            self._owned_pid_getters(),
            trace_enabled=diagnostics_level == "deep",
        )
        sampler.start()
        return sampler

    def _finish_resource_profile(
        self,
        sampler: Any,
        branches: NormalBranchResult,
        diagnostics: Any,
        *,
        include_model_inventory: bool,
        include_memory_trace: bool = False,
    ) -> None:
        snapshot = dict(sampler.stop())
        diagnostics.resource_profile = {
            "enabled": True,
            **snapshot,
            "branches": compact_scheduler_summary(branches.branch_metrics),
            "uncertainty": {
                "driver_memory_unit": "MiB",
                "pid_observed": dict(snapshot.get("pid_observed") or {}),
            },
        }
        if include_memory_trace:
            parallel = branches.branch_metrics.get("asr", {}).get("parallel")
            trace_stage = getattr(sampler, "trace_stage", None)
            if callable(trace_stage) and isinstance(parallel, dict):
                started = parallel.get("aed_started_at_monotonic")
                ended = parallel.get("aed_ended_at_monotonic")
                memory = parallel.get("memory")
                budget = (
                    memory.get("dispatch_transient_budget_mib")
                    if isinstance(memory, dict)
                    else None
                )
                if isinstance(started, (int, float)) and isinstance(
                    ended, (int, float)
                ):
                    diagnostics.resource_profile["aed_memory_trace"] = (
                        trace_stage(
                            float(started),
                            float(ended),
                            effective_budget_mib=(
                                int(budget)
                                if isinstance(budget, (int, float))
                                else None
                            ),
                            safety_margin_mib=(
                                int(self.config.asr.gpu_safety_margin_mib)
                            ),
                        )
                    )
        if include_model_inventory:
            diagnostics.resource_profile["inventories"] = dict(
                getattr(self.hot_host, "inventories", {})
            )

    def _run_nemotron_pipeline(
        self,
        normalized: Path,
        job_dir: Path,
        diagnostics: Any,
        overlap_enabled: bool,
        profile_enabled: bool,
        aed_budget_fraction: float = 1.0,
    ) -> tuple[NormalBranchResult, tuple[Any, ...], CropPlan]:
        speaker_owner = self.hot_host.speaker_owner
        asr_owner = self.hot_host.asr_owner
        branch_metrics: dict[str, dict[str, object]] = {}

        speaker_started = perf_counter()
        segments, speaker_metrics = _run_speaker(
            speaker_owner, Path(normalized), profile_enabled
        )
        branch_metrics["diarization"] = {
            "seconds": perf_counter() - speaker_started,
        }
        if speaker_metrics:
            branch_metrics["diarization"]["memory"] = speaker_metrics
        if not segments:
            raise TranscriptionError(
                "Nemotron speaker diarization returned no segments"
            )

        overlap_preparation: OverlapPreparation | None = None
        if overlap_enabled:
            overlap_windows = extract_overlap_windows_from_segments(segments)
            if overlap_windows:
                overlap_started = perf_counter()
                before_release = (
                    self._gpu_sampler("cuda:0")
                    if self._gpu_sampler is not None
                    else None
                )
                self.hot_host.close()
                try:
                    if before_release is not None and self._gpu_sampler is not None:
                        after_release = self._gpu_sampler("cuda:0")
                        release_record = build_gpu_stage_record(
                            before_release,
                            after_release,
                            peak_gpu_bytes=None,
                            load_seconds=None,
                            inference_seconds=None,
                        )
                        diagnostics.gpu_stages["hot_release_for_overlap"] = release_record
                        if release_record["released"] is not True:
                            raise TranscriberEnvironmentError(
                                "resident models did not release GPU memory for overlap"
                            )
                    preparer = self._overlap_factory(job_dir)
                    separated = preparer.separate_windows(
                        normalized, overlap_windows
                    )
                    temporary_diarizer = _create_and_open(
                        self._temporary_diarizer_factory
                    )
                    try:
                        centroids = temporary_diarizer.speaker_centroids(
                            normalized, segments
                        )
                        overlap_preparation = preparer.match_tracks(
                            separated,
                            centroids,
                            temporary_diarizer,
                        )
                    finally:
                        _close_owner(temporary_diarizer)
                finally:
                    self.hot_host.start()
                    asr_owner = self.hot_host.asr_owner
                    if hasattr(asr_owner, "aed_budget_fraction"):
                        asr_owner.aed_budget_fraction = float(aed_budget_fraction)
                branch_metrics["overlap"] = {
                    "seconds": perf_counter() - overlap_started,
                    "windows": len(overlap_windows),
                }

        crop_started = perf_counter()
        worker0 = getattr(asr_owner, "worker0", None)
        refine_vad = getattr(worker0, "refine_vad", None)
        crop_plan = build_crop_plan(
            Path(normalized),
            segments,
            job_dir / "speaker-crops",
            refine_vad if callable(refine_vad) else None,
            overlap_preparation=overlap_preparation,
            min_score=self.config.overlap.speaker_match_min_score,
            min_margin=self.config.overlap.speaker_match_min_margin,
            min_rms=self.config.overlap.track_min_rms,
        )
        branch_metrics["crops"] = {
            "seconds": perf_counter() - crop_started,
            "count": len(crop_plan.crops),
        }

        transcribe_crops = getattr(asr_owner, "transcribe_crops", None)
        if not callable(transcribe_crops):
            raise TranscriptionError(
                "resident FireRed owner does not support independent AED crops"
            )
        asr_started = perf_counter()
        crop_results = transcribe_crops(crop_plan)
        asr_result, attributed = attribute_crop_results(
            tuple(crop_results), segments, crop_plan.duration_ms
        )
        branch_metrics["asr"] = {
            "seconds": perf_counter() - asr_started,
        }
        response = getattr(asr_owner, "last_response", {})
        if isinstance(response, dict):
            metrics = response.get("metrics")
            if isinstance(metrics, dict):
                for key in ("scheduler", "parallel"):
                    if isinstance(metrics.get(key), dict):
                        branch_metrics["asr"][key] = metrics[key]
        return (
            NormalBranchResult(
                asr_result,
                segments,
                speaker_metrics,
                branch_metrics,
            ),
            attributed,
            crop_plan,
        )

    def _run_overlap_enhancement(
        self,
        normalized: Path,
        attributed: tuple[Any, ...],
        speaker_segments: tuple[SpeakerSegment, ...],
        job_dir: Path,
        diagnostics: Any,
    ) -> OverlapEnhancement:
        is_nemotron = self.config.diarization.backend == "nemotron3"
        nemotron_windows: tuple[dict[str, Any], ...] | None = None
        if is_nemotron:
            nemotron_windows = extract_overlap_windows_from_segments(
                speaker_segments
            )
            if not nemotron_windows:
                return OverlapEnhancement(attributed, (), False)

        speaker_owner = self.hot_host.speaker_owner
        centroid_provider = (
            getattr(speaker_owner, "speaker_centroids", None)
            if speaker_owner is not None
            else None
        )
        centroids = (
            centroid_provider(normalized, speaker_segments)
            if callable(centroid_provider)
            else None
        )
        before_release = (
            self._gpu_sampler("cuda:0") if self._gpu_sampler is not None else None
        )
        self.hot_host.close()
        try:
            if before_release is not None and self._gpu_sampler is not None:
                after_release = self._gpu_sampler("cuda:0")
                release_record = build_gpu_stage_record(
                    before_release,
                    after_release,
                    peak_gpu_bytes=None,
                    load_seconds=None,
                    inference_seconds=None,
                )
                diagnostics.gpu_stages["hot_release_for_overlap"] = release_record
                if release_record["released"] is not True:
                    raise TranscriberEnvironmentError(
                        "resident models did not release GPU memory for overlap"
                    )
            preparer = self._overlap_factory(job_dir)
            if nemotron_windows is not None:
                separated = preparer.separate_windows(
                    normalized, nemotron_windows
                )
            else:
                separated = preparer.detect_and_separate(normalized)
            temporary_diarizer = _create_and_open(
                self._temporary_diarizer_factory
            )
            try:
                if centroids is None:
                    centroids = temporary_diarizer.speaker_centroids(
                        normalized, speaker_segments
                    )
                preparation: OverlapPreparation = preparer.match_tracks(
                    separated,
                    centroids,
                    temporary_diarizer,
                )
            finally:
                _close_owner(temporary_diarizer)
            if not any(window.tracks for window in preparation.windows):
                return apply_prepared_overlap(
                    preparation,
                    attributed,
                    _NoTrackAsr(),
                    self.config.overlap.speaker_match_min_score,
                    self.config.overlap.speaker_match_min_margin,
                )
            temporary_asr = _create_and_open(
                lambda: self._temporary_asr_factory(job_dir)
            )
            try:
                return apply_prepared_overlap(
                    preparation,
                    attributed,
                    temporary_asr,
                    self.config.overlap.speaker_match_min_score,
                    self.config.overlap.speaker_match_min_margin,
                )
            finally:
                _close_owner(temporary_asr)
        finally:
            self.hot_host.start()

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
        sampler: Any | None = None
        profile_finished = False
        cleanup_attempted = False
        previous_budget_fraction = 1.0
        diagnostics_level = self._diagnostics_level(request)
        self._capture_overlap_pid = diagnostics_level in {"resources", "deep"}
        if not self.hot_host.started:
            self.hot_host.start()
        measurement_started = perf_counter()
        sampler = self._start_resource_sampler(request, diagnostics_level)
        try:
            asr_owner = self.hot_host.asr_owner
            speaker_owner = self.hot_host.speaker_owner
            if asr_owner is None or speaker_owner is None:
                raise TranscriptionError("resident model owners are unavailable")
            pipeline = TranscriptionPipeline(self.config)
            diagnostics = pipeline.new_diagnostics()
            warmup = getattr(self.hot_host, "warmup", None)
            warmup_started = perf_counter()
            if callable(warmup):
                warmup_metrics = warmup()
            else:
                warmup_metrics = dict(
                    getattr(self.hot_host, "warmup_metrics", {})
                )
            if not isinstance(warmup_metrics, dict):
                raise TranscriptionError(
                    "resident model warmup returned invalid metrics"
                )
            if bool(warmup_metrics.get("performed")):
                diagnostics.stage_seconds["warmup"] = (
                    perf_counter() - warmup_started
                )
            diagnostics.runtime["warmup"] = dict(warmup_metrics)
            diagnostics.runtime["timing_contract"] = {
                "excluded": ["initial_model_load"],
                "first_job_warmup_included": True,
                "aed_cache_cleanup_included": True,
                "output_publish_included": True,
            }
            diagnostics.runtime["diagnostics_level"] = diagnostics_level
            diagnostics.runtime["speaker_backend"] = (
                self.config.diarization.backend
            )
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
            previous_budget_fraction = getattr(
                asr_owner, "aed_budget_fraction", 1.0
            )
            if hasattr(asr_owner, "aed_budget_fraction"):
                asr_owner.aed_budget_fraction = float(
                    request.aed_budget_fraction
                )
            direct_attributed = None
            crop_plan = None
            with diagnostics.stage("normal_model_branches"):
                if self.config.diarization.backend == "nemotron3":
                    branches, direct_attributed, crop_plan = (
                        self._run_nemotron_pipeline(
                            normalized,
                            job_dir,
                            diagnostics,
                            request.overlap_enabled,
                            diagnostics_level in {"resources", "deep"},
                            aed_budget_fraction=float(request.aed_budget_fraction),
                        )
                    )
                else:
                    branches = run_normal_branches(
                        normalized,
                        asr_owner,
                        speaker_owner,
                        profile_enabled=diagnostics_level
                        in {"resources", "deep"},
                    )
            for branch_name, branch in branches.branch_metrics.items():
                seconds = branch.get("seconds")
                if isinstance(seconds, (int, float)):
                    diagnostics.stage_seconds[f"normal.{branch_name}"] = float(seconds)
                parallel = branch.get("parallel")
                if isinstance(parallel, dict):
                    for stage_name, stage_seconds in parallel.items():
                        if stage_name == "aed_worker_seconds" and isinstance(
                            stage_seconds, list
                        ):
                            for worker_index, worker_seconds in enumerate(
                                stage_seconds
                            ):
                                if isinstance(worker_seconds, (int, float)):
                                    diagnostics.stage_seconds[
                                        f"normal.asr.aed_worker_{worker_index}"
                                    ] = float(worker_seconds)
                            continue
                        if stage_name.endswith("_seconds") and isinstance(
                            stage_seconds, (int, float)
                        ):
                            diagnostics.stage_seconds[
                                f"normal.asr.{stage_name.removesuffix('_seconds')}"
                            ] = float(stage_seconds)
            scheduler = branches.branch_metrics.get("asr", {}).get("scheduler")
            if isinstance(scheduler, dict):
                diagnostics.scheduler = scheduler
            parallel_metrics = branches.branch_metrics.get("asr", {}).get("parallel")
            if isinstance(parallel_metrics, dict):
                diagnostics.scheduler["parallel"] = parallel_metrics
            if crop_plan is not None:
                diagnostics.runtime["nemotron"] = dict(
                    self.hot_host.inventories.get("speaker") or {}
                )
                diagnostics.runtime["nemotron"]["speaker_metrics"] = dict(
                    branches.speaker_metrics
                )
                diagnostics.scheduler["speaker_crops"] = {
                    "count": len(crop_plan.crops),
                    "short_count": crop_plan.short_count,
                    "subpiece_count": crop_plan.subpiece_count,
                    "overlap_duration_ms": crop_plan.overlap_duration_ms,
                    "spliced_overlap_count": crop_plan.spliced_overlap_count,
                }
                for warning in crop_plan.warnings:
                    diagnostics.warn(warning)
            if diagnostics_level == "deep":
                write_deep_traces(
                    staging_dir / "diagnostics",
                    diagnostics.scheduler,
                )
            diagnostics.scheduler = compact_scheduler_summary(diagnostics.scheduler)
            if not branches.speaker_segments:
                raise TranscriptionError("speaker diarization returned no segments")
            if direct_attributed is None:
                with diagnostics.stage("alignment"):
                    attributed = attribute_by_segment(
                        branches.asr_result.tokens,
                        branches.speaker_segments,
                    )
            else:
                attributed = direct_attributed

            enhancement = (
                self._run_overlap_enhancement(
                    normalized,
                    attributed,
                    branches.speaker_segments,
                    job_dir,
                    diagnostics,
                )
                if request.overlap_enabled
                and self.config.diarization.backend != "nemotron3"
                else None
            )
            result = pipeline.finish(
                branches.asr_result,
                attributed,
                branches.speaker_segments,
                diagnostics,
                overlap_enhancement=enhancement,
            )
            cleanup_owner = self.hot_host.asr_owner
            cleanup = getattr(cleanup_owner, "cleanup_cuda_cache", None)
            if not callable(cleanup):
                raise TranscriptionError(
                    "resident FireRed owner cannot clean CUDA cache"
                )
            cleanup_attempted = True
            cleanup_started = perf_counter()
            cleanup_metrics = cleanup()
            cleanup_seconds = perf_counter() - cleanup_started
            if not isinstance(cleanup_metrics, dict):
                raise TranscriptionError(
                    "resident FireRed owner returned invalid cache cleanup metrics"
                )
            diagnostics.stage_seconds["cache_cleanup"] = cleanup_seconds
            diagnostics.runtime["cache_cleanup"] = dict(cleanup_metrics)
            if sampler is not None:
                self._finish_resource_profile(
                    sampler,
                    branches,
                    diagnostics,
                    include_model_inventory=(
                        diagnostics_level == "deep"
                        or self.config.diagnostics.include_model_inventory
                    ),
                    include_memory_trace=diagnostics_level == "deep",
                )
                profile_finished = True
            if diagnostics_level == "off":
                diagnostics.stage_seconds.clear()
                diagnostics.scheduler.clear()
                diagnostics.asr.clear()
                diagnostics.gpu_stages.clear()
            output_started = perf_counter()
            export_transcript(
                result.transcript,
                staging_dir,
                overwrite=False,
                input_info=input_info,
            )
            if diagnostics_level != "off":
                preliminary_seconds = perf_counter() - measurement_started
                diagnostics.stage_seconds["end_to_end_after_load"] = (
                    preliminary_seconds
                )
                diagnostics.stage_seconds["end_to_end_after_warmup"] = (
                    preliminary_seconds
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
            if diagnostics_level != "off":
                output_ended = perf_counter()
                diagnostics.stage_seconds["output"] = (
                    output_ended - output_started
                )
                end_to_end_seconds = output_ended - measurement_started
                diagnostics.stage_seconds["end_to_end_after_load"] = (
                    end_to_end_seconds
                )
                diagnostics.stage_seconds["end_to_end_after_warmup"] = (
                    end_to_end_seconds
                )
                export_diagnostics(
                    result.diagnostics,
                    published,
                    overwrite=True,
                )
            return {"output_dir": str(published)}
        finally:
            if "asr_owner" in locals() and asr_owner is not None and hasattr(
                asr_owner, "aed_budget_fraction"
            ):
                asr_owner.aed_budget_fraction = previous_budget_fraction
            if not cleanup_attempted:
                cleanup_owner = getattr(self.hot_host, "asr_owner", None)
                cleanup = getattr(cleanup_owner, "cleanup_cuda_cache", None)
                if callable(cleanup):
                    try:
                        cleanup()
                    except Exception:
                        LOGGER.exception(
                            "AED CUDA cache cleanup failed while preserving job error"
                        )
            if sampler is not None and not profile_finished:
                sampler.stop()
            self._active_overlap_pid = None
            self._capture_overlap_pid = False
            shutil.rmtree(staging_dir, ignore_errors=True)
            if not request.keep_intermediate:
                shutil.rmtree(job_dir, ignore_errors=True)


class _NoTrackAsr:
    """The overlap applier never calls ASR when no matched track exists."""

    def transcribe(self, path: Path) -> AsrResult:
        raise AssertionError("overlap ASR must not run without matched tracks")


def _require_physical_gpu_zero(config: AppConfig) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    first_visible = visible.split(",", 1)[0].strip() if visible else "0"
    if config.general.device != "cuda:0" or first_visible != "0":
        raise ServiceProtocolError(
            "service is restricted to physical GPU 0; "
            f"device={config.general.device}, CUDA_VISIBLE_DEVICES={visible!r}"
        )


def build_hot_model_host(config: AppConfig, paths: ServicePaths) -> HotModelHost:
    """Construct the persistent AED owner and selected speaker owner."""

    from .firered_client import FireRedWorkerClient
    from .firered_parallel import (
        ParallelFireRedWorkerClient,
        WeightedAedPuncWorkerClient,
    )
    from .speaker_diarization import SpeakerDiarizer
    from .nemotron_client import NemotronWorkerClient
    warmup_audio = (
        Path(config.paths.model_dir)
        / "fireredasr2s"
        / "source"
        / "assets"
        / "hello_zh.wav"
    )

    def client(
        stderr_name: str,
        *,
        enable_vad: bool,
        enable_punc: bool,
        punc_only: bool = False,
    ) -> FireRedWorkerClient:
        return FireRedWorkerClient(
            worker_python=config.asr.worker_python,
            model_root=config.paths.model_dir,
            device="cuda:0",
            timeout_seconds=config.asr.timeout_seconds,
            stderr_log_path=paths.directory / stderr_name,
            use_half=config.asr.use_half,
            enable_vad=enable_vad,
            enable_punc=enable_punc,
            punc_only=punc_only,
            punc_token_budget=config.asr.punc_token_budget,
            punc_max_items=config.asr.punc_max_items,
            warmup_audio=warmup_audio,
        )

    def asr_factory() -> Any:
        worker_count = config.asr.aed_worker_count
        if config.asr.aed_scheduler == "weighted_duration":
            worker0 = client(
                "firered-worker-0.log",
                enable_vad=True,
                enable_punc=False,
            )
            worker1 = (
                client(
                    "firered-worker-1.log",
                    enable_vad=False,
                    enable_punc=False,
                )
                if worker_count == 2
                else None
            )
            punc_owner = client(
                "firered-punc.log",
                enable_vad=False,
                enable_punc=True,
                punc_only=True,
            )
            weight_options: dict[str, object] = {}
            if config.asr.aed_weight_artifact is not None:
                gpu = query_gpu("cuda:0")
                runtime = build_expected_runtime_identity(
                    config.paths.model_dir,
                    gpu_name=gpu.name,
                    driver_version=gpu.driver_version,
                    use_half=config.asr.use_half,
                )
                selection = load_weight_artifact(
                    config.asr.aed_weight_artifact,
                    runtime,
                )
                for warning in selection.warnings:
                    LOGGER.warning("%s", warning)
                weight_options = {
                    "weights": selection.weights,
                    "weight_source": selection.source,
                    "weight_artifact_checksum": selection.artifact_checksum,
                    "weight_warnings": selection.warnings,
                }
            return WeightedAedPuncWorkerClient(
                worker0,
                punc_owner,
                worker1=worker1,
                **weight_options,
                punc_token_budget=config.asr.punc_token_budget,
                punc_max_items=config.asr.punc_max_items,
            )
        return ParallelFireRedWorkerClient(
            client("firered-worker-0.log", enable_vad=True, enable_punc=False),
            client("firered-worker-1.log", enable_vad=False, enable_punc=False),
            punc_owner=client(
                "firered-punc.log",
                enable_vad=False,
                enable_punc=True,
                punc_only=True,
            ),
            punc_token_budget=config.asr.punc_token_budget,
            punc_max_items=config.asr.punc_max_items,
            gpu_safety_margin_mib=config.asr.gpu_safety_margin_mib,
        )

    def gpu_check() -> None:
        require_gpu_capacity(
            "cuda:0",
            minimum_total_mib=config.service.minimum_total_mib,
            minimum_free_mib=(
                config.service.minimum_free_mib
                if config.diarization.backend == "nemotron3"
                or config.asr.aed_scheduler != "weighted_duration"
                else 0
            ),
        )

    def speaker_factory() -> Any:
        if config.diarization.backend == "3d_speaker":
            return SpeakerDiarizer.from_config(config)
        return NemotronWorkerClient(
            worker_python=config.nemotron.worker_python,
            model_dir=config.paths.model_dir,
            timeout_seconds=config.nemotron.timeout_seconds,
            stderr_log_path=paths.directory / "nemotron-worker.log",
        )

    return HotModelHost(
        asr_factory=asr_factory,
        diarizer_factory=speaker_factory,
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
                if not paths.uses_abstract_socket:
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
