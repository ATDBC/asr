"""Client and parser for Alibaba DashScope Qwen-Audio file transcription API.

Integrates qwen-audio-3.1-asr-flash-filetrans with word-level timestamps.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_MODEL = "qwen-audio-3.1-asr-flash-filetrans"
POLL_INTERVAL_SECONDS = 3
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB


class QwenTranscribeError(RuntimeError):
    """Raised when DashScope Qwen transcription fails."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class QwenWordToken:
    """Word-level token with millisecond timestamps and punctuation."""

    text: str
    start_ms: int
    end_ms: int
    punctuation: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")


@dataclass(frozen=True, slots=True)
class QwenSentence:
    """Sentence block returned by Qwen-Audio ASR."""

    sentence_id: int
    speaker_id: int
    start_ms: int
    end_ms: int
    text: str
    words: tuple[QwenWordToken, ...]

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")


@dataclass(frozen=True, slots=True)
class QwenTranscriptionResult:
    """Complete parsed transcription result with sentence and word streams."""

    sentences: tuple[QwenSentence, ...]
    words: tuple[QwenWordToken, ...]
    raw_response: dict[str, Any]

    @property
    def full_text(self) -> str:
        return "".join(w.text + w.punctuation for w in self.words)


def parse_qwen_transcription(data: dict[str, Any]) -> QwenTranscriptionResult:
    """Parse raw DashScope response or cached JSON into structured models.

    Handles root-level transcripts, nested transcription dictionary, or output wrapper.
    """
    payload = data
    if "transcription" in payload and isinstance(payload["transcription"], dict):
        payload = payload["transcription"]
    elif "output" in payload and isinstance(payload["output"], dict):
        output = payload["output"]
        if "results" in output and isinstance(output["results"], list) and output["results"]:
            payload = output["results"][0]

    raw_transcripts = payload.get("transcripts", [])
    if not isinstance(raw_transcripts, list):
        raw_transcripts = []

    sentences: list[QwenSentence] = []
    all_words: list[QwenWordToken] = []

    for transcript in raw_transcripts:
        raw_sentences = transcript.get("sentences", []) if isinstance(transcript, dict) else []
        for s in raw_sentences:
            if not isinstance(s, dict):
                continue
            s_text = str(s.get("text") or "").strip()
            if not s_text:
                continue

            s_id = int(s.get("sentence_id", len(sentences) + 1))
            spk_id = int(s.get("speaker_id", 0))
            s_start = int(s.get("begin_time", 0))
            s_end = int(s.get("end_time", s_start))

            s_words: list[QwenWordToken] = []
            for w in s.get("words", []):
                if not isinstance(w, dict):
                    continue
                w_text = str(w.get("text") or "").strip()
                if not w_text:
                    continue
                w_start = int(w.get("begin_time", s_start))
                w_end = int(w.get("end_time", w_start))
                punct = str(w.get("punctuation") or "").strip()
                conf = float(w.get("confidence", 1.0))

                token = QwenWordToken(
                    text=w_text,
                    start_ms=w_start,
                    end_ms=max(w_start, w_end),
                    punctuation=punct,
                    confidence=conf,
                )
                s_words.append(token)
                all_words.append(token)

            sentences.append(
                QwenSentence(
                    sentence_id=s_id,
                    speaker_id=spk_id,
                    start_ms=s_start,
                    end_ms=max(s_start, s_end),
                    text=s_text,
                    words=tuple(s_words),
                )
            )

    return QwenTranscriptionResult(
        sentences=tuple(sentences),
        words=tuple(all_words),
        raw_response=data,
    )


def prepare_audio_mono16k(audio_path: Path, target_dir: Path) -> Path:
    """Downmix audio to 16kHz mono WAV if necessary."""
    if not shutil.which("ffmpeg"):
        return audio_path

    output_path = target_dir / f"{audio_path.stem}_mono16k.wav"
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-v", "error",
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        LOGGER.warning("FFmpeg downmix failed (%s), using original audio", res.stderr.decode("utf-8", "ignore"))
        return audio_path
    return output_path


class QwenTranscriberClient:
    """Synchronous DashScope REST API client for Qwen-Audio file transcription."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = 1500,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("QWEN_ASR_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or ""
        ).strip()
        if not resolved_key:
            raise QwenTranscribeError(
                "Missing Qwen API key. Please set QWEN_ASR_API_KEY or DASHSCOPE_API_KEY in environment.",
                400,
            )
        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def _get_upload_policy(self, client: httpx.Client) -> dict[str, Any]:
        url = f"{self.base_url}/uploads"
        params = {"action": "getPolicy", "model": self.model}
        resp = client.get(url, headers=self._headers(), params=params)
        if resp.status_code != 200:
            raise QwenTranscribeError(
                f"Failed to get DashScope upload policy: {resp.status_code} {resp.text}"
            )
        data = resp.json()
        policy = data.get("data")
        if not policy or not isinstance(policy, dict):
            raise QwenTranscribeError("Invalid upload policy payload from DashScope")
        return policy

    def _upload_file(self, client: httpx.Client, audio_path: Path) -> str:
        policy = self._get_upload_policy(client)
        upload_host = policy["upload_host"]
        key = f"{policy['upload_dir']}/{uuid4().hex}{audio_path.suffix}"
        form_data = {
            "OSSAccessKeyId": policy["oss_access_key_id"],
            "Signature": policy["signature"],
            "policy": policy["policy"],
            "x-oss-object-acl": policy["x_oss_object_acl"],
            "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
            "key": key,
            "success_action_status": "200",
        }

        with audio_path.open("rb") as f:
            resp = client.post(
                upload_host,
                data=form_data,
                files={"file": (audio_path.name, f)},
                timeout=300.0,
            )

        if resp.status_code not in (200, 204):
            raise QwenTranscribeError(f"Temporary OSS upload failed: {resp.status_code} {resp.text}")

        return f"oss://{key}"

    def _submit_task(self, client: httpx.Client, file_url: str) -> str:
        url = f"{self.base_url}/services/audio/asr/transcription"
        headers = {
            **self._headers(),
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable",
        }
        payload = {
            "model": self.model,
            "input": {"file_urls": [file_url]},
            "parameters": {
                "channel_id": [0],
                "diarization_enabled": True,
                "special_word_filter": json.dumps({"system_reserved_filter": False}),
            },
        }
        resp = client.post(url, headers=headers, json=payload, timeout=30.0)
        if resp.status_code != 200:
            raise QwenTranscribeError(f"Failed to submit transcription task: {resp.status_code} {resp.text}")

        data = resp.json()
        task_id = (data.get("output") or {}).get("task_id")
        if not task_id:
            raise QwenTranscribeError(f"No task_id in DashScope response: {data}")
        return str(task_id)

    def _poll_task(self, client: httpx.Client, task_id: str) -> str:
        url = f"{self.base_url}/tasks/{task_id}"
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            resp = client.get(url, headers=self._headers(), timeout=30.0)
            if resp.status_code != 200:
                raise QwenTranscribeError(f"Failed to query task status: {resp.status_code} {resp.text}")

            data = resp.json()
            output = data.get("output") or {}
            task_status = output.get("task_status")

            if task_status == "SUCCEEDED":
                results = output.get("results") or []
                if not results or results[0].get("subtask_status") != "SUCCEEDED":
                    raise QwenTranscribeError(f"Task succeeded but subtask failed: {output}")
                result_url = results[0].get("transcription_url")
                if not result_url:
                    raise QwenTranscribeError(f"Missing transcription_url in output: {output}")
                return str(result_url)

            if task_status in {"FAILED", "CANCELED"}:
                code = output.get("code") or task_status
                msg = output.get("message") or "Unknown error"
                raise QwenTranscribeError(f"Qwen transcription task failed ({code}): {msg}")

            time.sleep(POLL_INTERVAL_SECONDS)

        raise QwenTranscribeError(f"Qwen transcription timed out after {self.timeout_seconds} seconds")

    def _fetch_result(self, client: httpx.Client, result_url: str) -> dict[str, Any]:
        resp = client.get(result_url, timeout=60.0)
        if resp.status_code != 200:
            raise QwenTranscribeError(f"Failed to download transcription result: {resp.status_code}")
        data = resp.json()
        if not isinstance(data, dict):
            raise QwenTranscribeError("Downloaded result is not a JSON object")
        return data

    def transcribe(self, audio_path: Path) -> QwenTranscriptionResult:
        """Upload audio, submit task, poll until finished, and return structured result."""
        path = audio_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with tempfile.TemporaryDirectory(prefix="qwen_transcribe_") as temp_dir:
            prepared_path = prepare_audio_mono16k(path, Path(temp_dir))
            with httpx.Client(timeout=60.0) as client:
                LOGGER.info("Uploading audio %s to DashScope OSS...", prepared_path.name)
                oss_url = self._upload_file(client, prepared_path)
                LOGGER.info("Submitting ASR task for %s (%s)...", self.model, oss_url)
                task_id = self._submit_task(client, oss_url)
                LOGGER.info("Task submitted: %s. Polling for results...", task_id)
                result_url = self._poll_task(client, task_id)
                LOGGER.info("Downloading result from %s...", result_url)
                raw_data = self._fetch_result(client, result_url)
                return parse_qwen_transcription(raw_data)

    def transcribe_with_cache(
        self,
        audio_path: Path,
        cache_path: Path | None = None,
    ) -> QwenTranscriptionResult:
        """Load from cache if file exists, otherwise transcribe and cache result."""
        if cache_path is not None and cache_path.is_file():
            LOGGER.info("Loading cached Qwen transcription from %s", cache_path)
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return parse_qwen_transcription(data)

        result = self.transcribe(audio_path)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(result.raw_response, f, ensure_ascii=False, indent=2)
            LOGGER.info("Saved Qwen transcription to cache: %s", cache_path)
        return result
