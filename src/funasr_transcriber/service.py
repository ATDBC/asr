"""Persistent local owners and parallel normal-model execution."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
from typing import Any, Callable

from .service_protocol import ServiceBusyError, ServiceRequest
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
