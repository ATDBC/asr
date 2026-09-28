"""Local JSON-lines protocol and durable state for the transcription daemon."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, BinaryIO, Mapping

from .config import AppConfig
from .errors import TranscriptionError
from .locking import _pid_is_alive


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1024 * 1024
_MAX_FILESYSTEM_SOCKET_BYTES = 100


class ServiceProtocolError(TranscriptionError):
    """The local service state or protocol exchange is invalid."""


class ServiceBusyError(ServiceProtocolError):
    """The service cannot accept another job."""


@dataclass(frozen=True, slots=True)
class ServicePaths:
    directory: Path
    socket: Path
    state: Path
    lock: Path
    log: Path

    @classmethod
    def from_work_dir(cls, work_dir: Path) -> "ServicePaths":
        directory = Path(work_dir).expanduser().resolve() / "service"
        return cls(
            directory=directory,
            socket=directory / "transcriber.sock",
            state=directory / "state.json",
            lock=directory / "startup.lock",
            log=directory / "service.log",
        )

    @property
    def uses_abstract_socket(self) -> bool:
        """Use Linux's abstract namespace when the filesystem path is too long."""

        return len(os.fsencode(str(self.socket))) > _MAX_FILESYSTEM_SOCKET_BYTES

    @property
    def socket_address(self) -> str:
        """Return the bind/connect address without changing work-directory files."""

        if not self.uses_abstract_socket:
            return str(self.socket)
        digest = hashlib.sha256(os.fsencode(str(self.socket))).hexdigest()[:32]
        return f"\0funasr-transcriber-{os.getuid()}-{digest}"


@dataclass(frozen=True, slots=True)
class ServiceState:
    pid: int
    protocol_version: int
    config_fingerprint: str

    def payload(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "protocol_version": self.protocol_version,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ServiceRequest:
    identifier: str
    action: str
    job_id: str | None = None
    input_audio: Path | None = None
    output_dir: Path | None = None
    config_fingerprint: str | None = None
    overwrite: bool = False
    keep_intermediate: bool = False
    overlap_enabled: bool = False
    profile_resources: bool = False
    diagnostics_level: str = "basic"
    aed_budget_fraction: float = 0.9


def config_fingerprint(config: AppConfig) -> str:
    """Fingerprint the configuration that changes service compatibility."""

    payload = asdict(config)
    encoded = json.dumps(
        payload,
        default=str,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_identifier(payload: Mapping[str, Any]) -> str:
    identifier = str(payload.get("id") or "").strip()
    if not identifier:
        raise ServiceProtocolError("service request id must not be empty")
    return identifier


def _optional_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key, False)
    if not isinstance(value, bool):
        raise ServiceProtocolError(f"{key} must be a boolean")
    return value


def _diagnostics_level(payload: Mapping[str, Any]) -> str:
    value = str(payload.get("diagnostics_level") or "basic").lower()
    if value not in {"off", "basic", "resources", "deep"}:
        raise ServiceProtocolError(
            "diagnostics_level must be off, basic, resources, or deep"
        )
    return value


def _budget_fraction(payload: Mapping[str, Any]) -> float:
    value = float(payload.get("aed_budget_fraction", 0.9))
    if not 0.1 <= value <= 1.0:
        raise ServiceProtocolError(
            "aed_budget_fraction must stay within 0.1..1.0"
        )
    return value


def _absolute_path(payload: Mapping[str, Any], key: str) -> Path:
    value = Path(str(payload.get(key) or ""))
    if not value.is_absolute():
        raise ServiceProtocolError(f"{key} must be an absolute path")
    return value


def parse_request(payload: Mapping[str, Any]) -> ServiceRequest:
    """Validate one external message before it reaches the service queue."""

    identifier = _required_identifier(payload)
    action = str(payload.get("action") or "")
    if action not in {"health", "submit", "wait", "stop"}:
        raise ServiceProtocolError(f"unsupported service action: {action}")
    if action == "wait":
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            raise ServiceProtocolError("wait job_id must not be empty")
        return ServiceRequest(
            identifier=identifier,
            action=action,
            job_id=job_id,
        )
    if action != "submit":
        return ServiceRequest(identifier=identifier, action=action)
    fingerprint = str(payload.get("config_fingerprint") or "").strip()
    if not fingerprint:
        raise ServiceProtocolError("submit config_fingerprint must not be empty")
    return ServiceRequest(
        identifier=identifier,
        action=action,
        input_audio=_absolute_path(payload, "input_audio"),
        output_dir=_absolute_path(payload, "output_dir"),
        config_fingerprint=fingerprint,
        overwrite=_optional_bool(payload, "overwrite"),
        keep_intermediate=_optional_bool(payload, "keep_intermediate"),
        overlap_enabled=_optional_bool(payload, "overlap_enabled"),
        profile_resources=_optional_bool(payload, "profile_resources"),
        diagnostics_level=_diagnostics_level(payload),
        aed_budget_fraction=_budget_fraction(payload),
    )


def encode_message(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def read_message(stream: BinaryIO) -> dict[str, Any]:
    line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not line:
        raise ServiceProtocolError("service connection closed before response")
    if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
        raise ServiceProtocolError("service message is too large or incomplete")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ServiceProtocolError("service returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ServiceProtocolError("service message must be a JSON object")
    return payload


class ServiceStateStore:
    """Own state files only after their recorded process is known dead."""

    def __init__(
        self,
        paths: ServicePaths,
        pid_alive=_pid_is_alive,
    ):
        self.paths = paths
        self._pid_alive = pid_alive

    def read(self) -> ServiceState | None:
        try:
            payload = json.loads(self.paths.state.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ServiceProtocolError(
                f"service state is unreadable: {self.paths.state}"
            ) from exc
        try:
            return ServiceState(
                pid=int(payload["pid"]),
                protocol_version=int(payload["protocol_version"]),
                config_fingerprint=str(payload["config_fingerprint"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceProtocolError("service state is invalid") from exc

    def write(self, state: ServiceState) -> None:
        self.paths.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.paths.state.with_name(
            f".{self.paths.state.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(state.payload(), stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.paths.state)
        finally:
            temporary.unlink(missing_ok=True)

    def require_compatible(self, fingerprint: str) -> ServiceState | None:
        state = self.read()
        if state is None:
            return None
        if not self._pid_alive(state.pid):
            self.remove_stale()
            return None
        if state.protocol_version != PROTOCOL_VERSION:
            raise ServiceProtocolError("running service protocol is incompatible")
        if state.config_fingerprint != fingerprint:
            raise ServiceProtocolError(
                "running service configuration is incompatible; stop it first"
            )
        return state

    def remove_stale(self) -> None:
        """Remove only this service's known state and socket paths."""

        if not self.paths.uses_abstract_socket:
            self.paths.socket.unlink(missing_ok=True)
        self.paths.state.unlink(missing_ok=True)


class ServiceClient:
    """Synchronous client for one local Unix-socket protocol exchange."""

    def __init__(self, paths: ServicePaths, timeout_seconds: float):
        self.paths = paths
        self.timeout_seconds = float(timeout_seconds)

    def _exchange(
        self,
        payload: Mapping[str, Any],
        *,
        wait_for_completion: bool = False,
    ) -> dict[str, Any]:
        if self.timeout_seconds <= 0:
            raise ValueError("service timeout_seconds must be positive")
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(self.timeout_seconds)
        try:
            client.connect(self.paths.socket_address)
            if wait_for_completion:
                client.settimeout(None)
            stream = client.makefile("rwb")
            try:
                stream.write(encode_message(payload))
                stream.flush()
                return read_message(stream)
            finally:
                stream.close()
        except (OSError, TimeoutError) as exc:
            raise ServiceProtocolError(
                f"unable to connect to local service: {exc}"
            ) from exc
        finally:
            client.close()

    @staticmethod
    def _terminal(terminal: dict[str, Any], identifier: str) -> dict[str, Any]:
        if terminal.get("id") != identifier or terminal.get("terminal") is not True:
            raise ServiceProtocolError("service response id or terminal state is invalid")
        if terminal.get("ok") is not True:
            raise ServiceProtocolError(
                str(terminal.get("error") or "service request failed")
            )
        return terminal

    def health(self, identifier: str) -> dict[str, Any]:
        return self._terminal(self._exchange({"id": identifier, "action": "health"}), identifier)

    @staticmethod
    def _submit_payload(request: ServiceRequest) -> dict[str, object]:
        return {
            "id": request.identifier,
            "action": request.action,
            "input_audio": str(request.input_audio),
            "output_dir": str(request.output_dir),
            "config_fingerprint": request.config_fingerprint,
            "overwrite": request.overwrite,
            "keep_intermediate": request.keep_intermediate,
            "overlap_enabled": request.overlap_enabled,
            "profile_resources": request.profile_resources,
            "diagnostics_level": request.diagnostics_level,
            "aed_budget_fraction": request.aed_budget_fraction,
        }

    def submit(self, request: ServiceRequest) -> dict[str, Any]:
        """Submit a job and return its immediate acceptance acknowledgement."""

        if request.action != "submit":
            raise ValueError("submit requires a submit request")
        accepted = self._exchange(self._submit_payload(request))
        if accepted.get("id") != request.identifier:
            raise ServiceProtocolError("service response id is invalid")
        if accepted.get("ok") is not True:
            raise ServiceProtocolError(
                str(accepted.get("error") or "service request failed")
            )
        if accepted.get("terminal") is not False or accepted.get("status") != "accepted":
            raise ServiceProtocolError("service submit acknowledgement is invalid")
        if accepted.get("job_id") != request.identifier:
            raise ServiceProtocolError("service submit job id is invalid")
        return accepted

    def wait(self, job_id: str, identifier: str) -> dict[str, Any]:
        """Wait for one accepted job without re-submitting it."""

        response = self._terminal(
            self._exchange(
                {"id": identifier, "action": "wait", "job_id": str(job_id)},
                wait_for_completion=True,
            ),
            identifier,
        )
        if response.get("job_id") != job_id:
            raise ServiceProtocolError("service wait job id is invalid")
        return response

    def submit_and_wait(self, request: ServiceRequest) -> dict[str, Any]:
        if request.action != "submit":
            raise ValueError("submit_and_wait requires a submit request")
        self.submit(request)
        return self.wait(request.identifier, request.identifier)

    def stop_and_wait(self, identifier: str) -> dict[str, Any]:
        return self._terminal(self._exchange({"id": identifier, "action": "stop"}), identifier)
