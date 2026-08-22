"""FireRed AED/Punc JSON Lines worker with a hosted VAD helper.

The worker is the AED/Punc functional owner. FireRedVAD is hosted in the same
`.venv-asr` process only to refine long speaker-separation intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, IO, Protocol

from .firered_contract import normalize_official_result
from .firered_scheduler import AdaptiveBatchScheduler


LOGGER = logging.getLogger(__name__)

SCRIPTED_VAD_THRESHOLD_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class AudioInfo:
    sample_rate: int
    channels: int
    subtype: str


@dataclass(frozen=True, slots=True)
class ScheduledSegment:
    """One VAD interval with its position in the caller's chronological list."""

    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def duration_ordered_batches(
    segments: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    batch_size: int,
    frame_budget: int | None = None,
    frame_rate: int = 100,
) -> list[list[ScheduledSegment]]:
    """Group intervals by similar duration while retaining original indices.

    The returned groups are execution order only. Callers must use ``index``
    to restore chronological output order before formatting the response.
    ``frame_budget`` is optional and limits the sum of estimated feature frames
    per batch; an individual interval is allowed to exceed the budget.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if frame_budget is not None and frame_budget <= 0:
        raise ValueError("frame_budget must be positive when provided")
    scheduled = [
        ScheduledSegment(index, float(start_s), float(end_s))
        for index, (start_s, end_s) in enumerate(segments)
    ]
    ordered = sorted(
        scheduled,
        key=lambda item: (item.duration_s, item.index),
    )
    batches: list[list[ScheduledSegment]] = []
    current: list[ScheduledSegment] = []
    current_frames = 0
    for item in ordered:
        item_frames = max(1, int(round(item.duration_s * frame_rate)))
        budget_would_be_exceeded = (
            frame_budget is not None
            and current
            and current_frames + item_frames > frame_budget
        )
        if current and (
            len(current) >= batch_size or budget_would_be_exceeded
        ):
            batches.append(current)
            current = []
            current_frames = 0
        current.append(item)
        current_frames += item_frames
    if current:
        batches.append(current)
    return batches


def _synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _cuda_peak_memory_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    return None


def _cuda_memory_allocated_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    return None


def _cuda_peak_reserved_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_reserved() / (1024 * 1024))
    except ImportError:
        pass
    return None


def _cuda_free_memory_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            return int(free_bytes / (1024 * 1024))
    except ImportError:
        pass
    return None


def _scheduled_process(
    system: FireRedSystem,
    wav_path: str,
    uttid: str,
    vad_segments: list[tuple[float, float]],
    batch_size: int,
    frame_budget: int | None = None,
    memory_limit_mib: int | None = 10_000,
    memory_safety_mib: int = 512,
    free_memory_provider: Callable[[], int | None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Run the pinned modules with duration-aware AED batches.

    This mirrors the small orchestration layer in the pinned FireRed system,
    but keeps the project-owned execution scheduler outside the ignored model
    checkout. All result lists are restored to the caller's VAD order before
    punctuation and response formatting.
    """

    import soundfile as sf

    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    duration_s = wav_np.shape[0] / sample_rate
    config = system.config
    records: list[tuple[int, dict[str, object], object]] = []
    scheduler = AdaptiveBatchScheduler(
        vad_segments,
        max_batch_size=batch_size,
        frame_budget=frame_budget or 8000,
        memory_limit_mib=memory_limit_mib,
        safety_mib=memory_safety_mib,
        free_memory_provider=free_memory_provider or _cuda_free_memory_mib,
    )
    scheduler.plan()
    while (decision := scheduler.next_batch()) is not None:
        batch = list(decision.items)
        batch_uttid: list[str] = []
        batch_wav: list[tuple[int, object]] = []
        for item in batch:
            start = int(item.start_s * sample_rate)
            end = int(item.end_s * sample_rate)
            batch_uttid.append(
                f"{uttid}_s{int(item.start_s * 1000)}_e{int(item.end_s * 1000)}"
            )
            batch_wav.append((sample_rate, wav_np[start:end]))
        _reset_cuda_peak_memory()
        asr_started = clock()
        asr_results = system.asr.transcribe(batch_uttid, batch_wav)
        _synchronize_cuda()
        scheduler.observe(
            decision,
            actual_seconds=clock() - asr_started,
            actual_peak_mib=_cuda_peak_memory_mib(),
        )
        lid_results = (
            system.lid.process(batch_uttid, batch_wav)
            if getattr(config, "enable_lid", False)
            else [None] * len(asr_results)
        )
        item_by_uttid = {
            uttid: item for uttid, item in zip(batch_uttid, batch)
        }
        lid_by_uttid = {
            str(result.get("uttid")): result
            for result in lid_results
            if isinstance(result, dict) and result.get("uttid") is not None
        }
        for asr_result in asr_results:
            result_uttid = str(asr_result.get("uttid") or "")
            item = item_by_uttid.get(result_uttid)
            if item is None:
                raise ValueError(
                    "FireRed AED returned an unknown segment uttid: "
                    f"{result_uttid}"
                )
            lid_result = lid_by_uttid.get(result_uttid)
            text = str(asr_result.get("text", "")).strip()
            if not text or re.search(r"(<blank>)|(<sil>)", text):
                continue
            records.append((item.index, asr_result, lid_result))

    records.sort(key=lambda item: item[0])
    asr_results = [item[1] for item in records]
    lid_results = [item[2] for item in records]
    punc_results: list[dict[str, object]]
    if getattr(config, "enable_punc", False):
        punc_results = []
        punc_batch: list[tuple[dict[str, object], str]] = []
        for index, asr_result in enumerate(asr_results):
            timestamp = asr_result.get("timestamp", [])
            punc_batch.append((asr_result, timestamp))
            if len(punc_batch) < batch_size and index != len(asr_results) - 1:
                continue
            punc_results.extend(
                system.punc.process_with_timestamp(
                    [item[1] for item in punc_batch],
                    [str(item[0]["uttid"]) for item in punc_batch],
                )
            )
            punc_batch = []
    else:
        punc_results = [item for item in asr_results]

    sentences: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    for asr_result, punc_result, lid_result in zip(
        asr_results, punc_results, lid_results
    ):
        start_ms, end_ms = str(asr_result["uttid"]).split("_")[-2:]
        start_ms = int(start_ms[1:])
        end_ms = int(end_ms[1:])
        if getattr(config.asr_config, "return_timestamp", False):
            for index, punc_sentence in enumerate(
                punc_result["punc_sentences"]
            ):
                start = start_ms + int(float(punc_sentence["start_s"]) * 1000)
                end = start_ms + int(float(punc_sentence["end_s"]) * 1000)
                if index == 0:
                    start = start_ms
                if index == len(punc_result["punc_sentences"]) - 1:
                    end = end_ms
                sentence = {
                    "start_ms": start,
                    "end_ms": end,
                    "text": punc_sentence["punc_text"],
                    "asr_confidence": asr_result["confidence"],
                    "lang": None,
                    "lang_confidence": 0,
                }
                if lid_result:
                    sentence["lang"] = lid_result["lang"]
                    sentence["lang_confidence"] = lid_result["confidence"]
                sentences.append(sentence)
        else:
            text = (
                punc_result["punc_text"]
                if getattr(config, "enable_punc", False)
                else asr_result["text"]
            )
            sentences.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "asr_confidence": asr_result["confidence"],
                    "lang": None,
                    "lang_confidence": 0,
                }
            )
        for word, start, end in asr_result.get("timestamp", []):
            words.append(
                {
                    "start_ms": int(float(start) * 1000 + start_ms),
                    "end_ms": int(float(end) * 1000 + start_ms),
                    "text": word,
                }
            )

    text = "".join(str(sentence["text"]) for sentence in sentences)
    text = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text)
    return {
        "uttid": uttid,
        "text": text,
        "sentences": sentences,
        "vad_segments_ms": [
            (int(start * 1000), int(end * 1000))
            for start, end in vad_segments
        ],
        "dur_s": duration_s,
        "words": words,
        "wav_path": wav_path,
        "_scheduler": scheduler.summary(),
    }


def _explicit_aed_process(
    system: FireRedSystem,
    wav_path: str,
    uttid: str,
    batch_plan: list[list[dict[str, object]]],
    vad_segments: list[tuple[float, float]],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Execute a caller-owned immutable batch plan without VAD or Punc."""

    import soundfile as sf

    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    duration_s = wav_np.shape[0] / sample_rate
    records: list[dict[str, object]] = []
    batch_metrics: list[dict[str, object]] = []
    resident_allocated_mib = _cuda_memory_allocated_mib()
    for batch_index, raw_batch in enumerate(batch_plan):
        batch_uttid: list[str] = []
        batch_wav: list[tuple[int, object]] = []
        batch_items: list[dict[str, object]] = []
        for raw_item in raw_batch:
            if not isinstance(raw_item, dict):
                raise ValueError("batch_plan items must be objects")
            index = int(raw_item["index"])
            start_ms = int(raw_item["start_ms"])
            end_ms = int(raw_item["end_ms"])
            if start_ms < 0 or end_ms <= start_ms:
                raise ValueError("batch_plan item bounds are invalid")
            start = int(start_ms * sample_rate / 1000)
            end = int(end_ms * sample_rate / 1000)
            item_uttid = f"{uttid}_s{start_ms}_e{end_ms}"
            batch_uttid.append(item_uttid)
            batch_wav.append((sample_rate, wav_np[start:end]))
            batch_items.append({"index": index, "uttid": item_uttid})
        if not batch_uttid:
            continue
        _reset_cuda_peak_memory()
        started = clock()
        results = system.asr.transcribe(batch_uttid, batch_wav)
        _synchronize_cuda()
        actual_seconds = clock() - started
        batch_metrics.append({
            "batch_index": batch_index,
            "item_count": len(batch_items),
            "indices": [int(item["index"]) for item in batch_items],
            "actual_seconds": actual_seconds,
            "peak_allocated_mib": _cuda_peak_memory_mib(),
            "peak_reserved_mib": _cuda_peak_reserved_mib(),
            "resident_allocated_mib": resident_allocated_mib,
        })
        returned = {str(result.get("uttid")): result for result in results}
        for item in batch_items:
            result = returned.get(str(item["uttid"]))
            if result is None:
                continue
            text = str(result.get("text", "")).strip()
            if not text or re.search(r"(<blank>)|(<sil>)", text):
                continue
            records.append({"index": item["index"], "result": result})
        LOGGER.debug("explicit AED batch completed in %.3fs", actual_seconds)

    records.sort(key=lambda item: int(item["index"]))
    return {
        "uttid": uttid,
        "duration_ms": int(round(duration_s * 1000)),
        "vad_segments_ms": [
            [int(round(start * 1000)), int(round(end * 1000))]
            for start, end in vad_segments
        ],
        "aed_results": records,
        "batch_metrics": batch_metrics,
        "resident_allocated_mib": resident_allocated_mib,
    }


def _punctuate_aed_results(
    system: FireRedSystem,
    request_id: str,
    aed_results: list[dict[str, object]],
    duration_ms: int,
    vad_segments_ms: list[list[int]],
    batch_size: int,
) -> dict[str, object]:
    """Apply one shared Punc model and produce the normal raw result shape."""

    ordered = sorted(aed_results, key=lambda item: int(item["index"]))
    results = [item["result"] for item in ordered]
    sentences: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    for offset in range(0, len(results), max(1, batch_size)):
        group = results[offset:offset + max(1, batch_size)]
        punctuated = system.punc.process_with_timestamp(
            [result.get("timestamp", []) for result in group],
            [str(result["uttid"]) for result in group],
        )
        for asr_result, punc_result in zip(group, punctuated):
            start_ms, end_ms = str(asr_result["uttid"]).split("_")[-2:]
            start_ms = int(start_ms[1:])
            end_ms = int(end_ms[1:])
            for index, sentence in enumerate(punc_result["punc_sentences"]):
                start = start_ms + int(float(sentence["start_s"]) * 1000)
                end = end_ms if index == len(punc_result["punc_sentences"]) - 1 else start_ms + int(float(sentence["end_s"]) * 1000)
                if index == 0:
                    start = start_ms
                sentences.append({
                    "start_ms": start,
                    "end_ms": end,
                    "text": sentence["punc_text"],
                    "asr_confidence": asr_result["confidence"],
                })
            for word, start, end in asr_result.get("timestamp", []):
                words.append({
                    "start_ms": int(float(start) * 1000 + start_ms),
                    "end_ms": int(float(end) * 1000 + start_ms),
                    "text": word,
                })
    text = "".join(str(sentence["text"]) for sentence in sentences)
    text = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text)
    return {
        "uttid": request_id,
        "text": text,
        "sentences": sentences,
        "vad_segments_ms": vad_segments_ms,
        "dur_s": duration_ms / 1000.0,
        "words": words,
    }


def _warmup_process(system: FireRedSystem, wav_path: str) -> None:
    """Exercise the resident AED/Punc CUDA paths before production timing."""

    import soundfile as sf

    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    if getattr(system, "asr", None) is not None:
        uttid = "__firered_warmup__"
        asr_results = system.asr.transcribe(
            [uttid], [(sample_rate, wav_np)]
        )
        punc = getattr(system, "punc", None)
        if punc is not None and asr_results:
            punc.process_with_timestamp(
                [asr_results[0].get("timestamp", [])],
                [uttid],
            )
    _synchronize_cuda()


class FireRedSystem(Protocol):
    def process(self, path: str, uttid: str) -> dict[str, object]:
        """Run the complete VAD, AED, and punctuation path."""


class GuardedVad:
    """Reject unsafe VAD segments before the official system reaches AED."""

    def __init__(self, vad: object):
        self._vad = vad

    def detect(self, path: str):
        result, probabilities = self._vad.detect(path)
        for start_s, end_s in result.get("timestamps", []):
            _check_segment_bounds(float(start_s), float(end_s))
        return result, probabilities

    def __getattr__(self, name: str) -> object:
        return getattr(self._vad, name)


def _check_segment_bounds(start_s: float, end_s: float) -> None:
    if end_s - start_s > 60.0:
        raise ValueError(
            "FireRedVAD segment exceeds the 60 second AED limit: "
            f"{start_s}..{end_s}"
        )


class ScriptedVad:
    """Replay speaker segments, refining long turns from one global VAD pass."""

    def __init__(
        self,
        segments_s: list[tuple[float, float]],
        fallback_vad: object | None = None,
    ):
        self._segments = list(segments_s)
        self._fallback_vad = fallback_vad

    def detect(self, path: str):
        long_segment = any(
            end - start > SCRIPTED_VAD_THRESHOLD_SECONDS
            for start, end in self._segments
        )
        if not long_segment:
            return {"timestamps": list(self._segments)}, None
        neural_segments: list[tuple[float, float]] = []
        if self._fallback_vad is not None:
            neural_result, _ = self._fallback_vad.detect(path)
            neural_segments = [
                (float(start), float(end))
                for start, end in neural_result.get("timestamps", [])
            ]
        output: list[tuple[float, float]] = []
        for start_s, end_s in self._segments:
            duration = end_s - start_s
            if duration <= SCRIPTED_VAD_THRESHOLD_SECONDS:
                output.append((start_s, end_s))
                continue
            subdivided: list[tuple[float, float]] = []
            for neural_start, neural_end in neural_segments:
                intersection_start = max(start_s, neural_start)
                intersection_end = min(end_s, neural_end)
                if intersection_end > intersection_start:
                    subdivided.append((intersection_start, intersection_end))
            if subdivided:
                output.extend(subdivided)
                continue
            cursor = start_s
            while end_s - cursor > SCRIPTED_VAD_THRESHOLD_SECONDS:
                output.append(
                    (cursor, cursor + SCRIPTED_VAD_THRESHOLD_SECONDS)
                )
                cursor += SCRIPTED_VAD_THRESHOLD_SECONDS
            output.append((cursor, end_s))
        for start_s, end_s in output:
            _check_segment_bounds(start_s, end_s)
        return {"timestamps": output}, None


def _parse_scripted_segments(raw: object) -> list[tuple[float, float]]:
    """Validate request-provided VAD segments and convert ms to seconds."""

    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("vad_segments_ms must be a non-empty list")
    segments: list[tuple[float, float]] = []
    previous_start = -1.0
    previous_end = -1.0
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("vad_segments_ms entries must contain two timestamps")
        try:
            start_s = float(item[0]) / 1000.0
            end_s = float(item[1]) / 1000.0
        except (TypeError, ValueError) as exc:
            raise ValueError("vad_segments_ms entries must be numeric") from exc
        if not start_s >= 0.0 or not end_s > start_s:
            raise ValueError("vad_segments_ms bounds are invalid")
        if start_s < previous_start or end_s < previous_end:
            raise ValueError("vad_segments_ms must be monotonic")
        segments.append((start_s, end_s))
        previous_start = start_s
        previous_end = end_s
    return segments


def read_audio_info(path: Path) -> AudioInfo:
    import soundfile as sf

    info = sf.info(str(path))
    return AudioInfo(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=str(info.subtype),
    )


def _validate_audio(
    raw_path: object,
    audio_info: Callable[[Path], AudioInfo],
) -> Path:
    path = Path(str(raw_path or ""))
    if not path.is_absolute():
        raise ValueError("audio_path must be absolute")
    if path.suffix.lower() != ".wav":
        raise ValueError("audio_path must be a WAV file")
    if not path.is_file():
        raise ValueError(f"audio_path does not exist: {path}")
    info = audio_info(path)
    if (
        info.sample_rate != 16000
        or info.channels != 1
        or info.subtype != "PCM_16"
    ):
        raise ValueError("audio must be normalized to 16 kHz mono PCM_16")
    return path


def _write_response(output: IO[str], response: dict[str, object]) -> None:
    output.write(
        json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )
    output.flush()


def _error_response(
    identifier: str | None,
    error: Exception,
) -> dict[str, object]:
    return {
        "id": identifier,
        "ok": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _no_peak_sample() -> int | None:
    return None


def _peak_gpu_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def firered_inventory(system: FireRedSystem) -> dict[str, object]:
    """Return physical inventory facts, including the hosted VAD helper.

    The ``vad`` entry describes a speaker-separation helper even though the
    model is physically loaded by this worker process.
    """

    from .resource_profile import module_inventory

    owners = (
        ("vad", "vad_model"),
        ("asr", "model"),
        ("punc", "model"),
    )
    modules: dict[str, object] = {}
    for owner_name, model_name in owners:
        owner = getattr(system, owner_name, None)
        model = getattr(owner, model_name, None)
        if model is None:
            modules[owner_name] = {"available": False}
        else:
            modules[owner_name] = module_inventory(owner_name, model)
    memory: dict[str, int | None] = {
        "allocated_bytes": None,
        "reserved_bytes": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            memory = {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
            }
    except ImportError:
        pass
    return {"modules": modules, "memory": memory}


def serve(
    input_stream: IO[str],
    protocol_output: IO[str],
    system_factory: Callable[[], FireRedSystem],
    audio_info: Callable[[Path], AudioInfo] = read_audio_info,
    clock: Callable[[], float] = time.perf_counter,
    peak_sampler: Callable[[], int | None] = _no_peak_sample,
    inventory_provider: Callable[[FireRedSystem], dict[str, object]] = (
        firered_inventory
    ),
) -> None:
    """Load FireRed once, then serve requests until shutdown or EOF."""

    load_started = clock()
    system = system_factory()
    load_seconds = clock() - load_started
    inference_seconds = 0.0
    seen_ids: set[str] = set()
    for line in input_stream:
        if not line.strip():
            continue
        identifier: str | None = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            identifier = str(request.get("id") or "")
            if not identifier:
                raise ValueError("request id must not be empty")
            if identifier in seen_ids:
                raise ValueError(f"duplicate request id: {identifier}")
            seen_ids.add(identifier)
            action = str(request.get("action") or "")
            if action == "shutdown":
                _write_response(
                    protocol_output,
                    {"id": identifier, "ok": True, "shutdown": True},
                )
                return
            if action == "warmup":
                path = _validate_audio(request.get("audio_path"), audio_info)
                warmup_started = clock()
                _warmup_process(system, str(path))
                _write_response(
                    protocol_output,
                    {
                        "id": identifier,
                        "ok": True,
                        "warmup_seconds": clock() - warmup_started,
                        "warmup_excluded": True,
                    },
                )
                continue
            if action == "status":
                _write_response(
                    protocol_output,
                    {
                        "id": identifier,
                        "ok": True,
                        "status": inventory_provider(system),
                        "metrics": {
                            "load_seconds": load_seconds,
                            "peak_gpu_bytes": peak_sampler(),
                        },
                    },
                )
                continue
            if action not in {
                "transcribe", "verify", "refine_vad", "aed_plan", "punctuate"
            }:
                raise ValueError(f"unsupported worker action: {action}")
            path = _validate_audio(request.get("audio_path"), audio_info)
            raw_segments = request.get("vad_segments_ms")
            scripted_segments = (
                None
                if raw_segments is None
                else _parse_scripted_segments(raw_segments)
            )
            if action == "refine_vad":
                if scripted_segments is None:
                    raise ValueError("refine_vad requires vad_segments_ms")
                original_vad = getattr(system, "vad", None)
                if original_vad is None:
                    raise ValueError("worker was started without VAD")
                refined, _ = ScriptedVad(scripted_segments, original_vad).detect(str(path))
                _write_response(
                    protocol_output,
                    {
                        "id": identifier,
                        "ok": True,
                        "vad_segments_ms": [
                            [int(round(start * 1000)), int(round(end * 1000))]
                            for start, end in refined["timestamps"]
                        ],
                    },
                )
                continue
            if action == "aed_plan":
                if scripted_segments is None:
                    raise ValueError("aed_plan requires vad_segments_ms")
                raw_plan = request.get("batch_plan")
                if not isinstance(raw_plan, list):
                    raise ValueError("aed_plan requires batch_plan")
                response = _explicit_aed_process(
                    system,
                    str(path),
                    identifier,
                    raw_plan,
                    scripted_segments,
                )
                _write_response(
                    protocol_output,
                    {"id": identifier, "ok": True, **response},
                )
                continue
            if action == "punctuate":
                raw_results = request.get("aed_results")
                if not isinstance(raw_results, list):
                    raise ValueError("punctuate requires aed_results")
                raw_vad_ms = request.get("vad_segments_ms")
                if not isinstance(raw_vad_ms, list):
                    raise ValueError("punctuate requires vad_segments_ms")
                duration_ms = int(request.get("duration_ms") or 0)
                if duration_ms <= 0:
                    raise ValueError("punctuate requires positive duration_ms")
                response = _punctuate_aed_results(
                    system,
                    identifier,
                    raw_results,
                    duration_ms,
                    raw_vad_ms,
                    int(getattr(system.config, "punc_batch_size", 1)),
                )
                _write_response(
                    protocol_output,
                    normalize_official_result(identifier, response),
                )
                continue
            inference_started = clock()
            if scripted_segments is None:
                raw = system.process(str(path), uttid=identifier)
            else:
                original_vad = getattr(system, "vad", None)
                scripted_vad = ScriptedVad(scripted_segments, original_vad)
                system.vad = scripted_vad
                try:
                    if (
                        hasattr(system, "asr")
                        and hasattr(system, "config")
                    ):
                        vad_result, _ = scripted_vad.detect(str(path))
                        raw = _scheduled_process(
                            system,
                            str(path),
                            identifier,
                            list(vad_result["timestamps"]),
                            int(system.config.asr_batch_size),
                            getattr(system.config, "segment_frame_budget", None),
                            getattr(system.config, "segment_memory_limit_mib", 10_000),
                            getattr(system.config, "segment_memory_safety_mib", 512),
                        )
                    else:
                        raw = system.process(str(path), uttid=identifier)
                finally:
                    system.vad = original_vad
            inference_seconds += clock() - inference_started
            response = normalize_official_result(identifier, raw)
            response["metrics"] = {
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "peak_gpu_bytes": peak_sampler(),
            }
            scheduler_metrics = raw.get("_scheduler")
            if isinstance(scheduler_metrics, dict):
                response["metrics"]["scheduler"] = scheduler_metrics
            if action == "verify":
                response["verified"] = True
            _write_response(protocol_output, response)
        except Exception as exc:
            LOGGER.exception("FireRed worker request failed")
            _write_response(
                protocol_output,
                _error_response(identifier or None, exc),
            )


def build_firered_system(
    model_root: Path,
    device: str,
    segment_batch_size: int = 1,
    segment_frame_budget: int = 8000,
    segment_memory_limit_mib: int = 10_000,
    segment_memory_safety_mib: int = 512,
    use_half: bool = False,
    enable_vad: bool = True,
    enable_punc: bool = True,
) -> FireRedSystem:
    """Construct the AED/Punc worker and its hosted speaker-VAD helper."""

    import torch
    from .firered_compat import install_bf16_forced_align_compat
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    from fireredasr2s.fireredvad import FireRedVadConfig

    if not device.startswith("cuda:"):
        raise ValueError("FireRed worker requires an explicit CUDA device")
    try:
        device_index = int(device.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid CUDA device: {device}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the FireRed environment")
    if segment_batch_size <= 0:
        raise ValueError("segment_batch_size must be positive")
    if segment_frame_budget <= 0:
        raise ValueError("segment_frame_budget must be positive")
    if segment_memory_limit_mib <= 0:
        raise ValueError("segment_memory_limit_mib must be positive")
    if segment_memory_safety_mib < 0:
        raise ValueError("segment_memory_safety_mib must not be negative")
    torch.cuda.set_device(device_index)
    install_bf16_forced_align_compat()

    root = Path(model_root).expanduser().resolve() / "fireredasr2s"
    config = FireRedAsr2SystemConfig(
        vad_model_dir=str(root / "FireRedVAD" / "VAD"),
        asr_type="aed",
        asr_model_dir=str(root / "FireRedASR2-AED"),
        punc_model_dir=str(root / "FireRedPunc"),
        vad_config=FireRedVadConfig(
            use_gpu=False,
            smooth_window_size=5,
            speech_threshold=0.4,
            min_speech_frame=20,
            max_speech_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        ),
        asr_config=FireRedAsr2Config(
            use_gpu=True,
            use_half=use_half,
            beam_size=3,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=True,
        ),
        punc_config=FireRedPuncConfig(use_gpu=True),
        asr_batch_size=segment_batch_size,
        punc_batch_size=segment_batch_size,
        enable_vad=enable_vad,
        enable_lid=False,
        enable_punc=enable_punc,
    )
    system = FireRedAsr2System(config)
    system.config.segment_frame_budget = segment_frame_budget
    system.config.segment_memory_limit_mib = segment_memory_limit_mib
    system.config.segment_memory_safety_mib = segment_memory_safety_mib
    if system.vad is not None:
        system.vad = GuardedVad(system.vad)
    return system


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment-batch-size", type=_positive_int, default=1)
    parser.add_argument("--segment-frame-budget", type=_positive_int, default=8000)
    parser.add_argument("--segment-memory-limit-mib", type=_positive_int, default=10_000)
    parser.add_argument("--segment-memory-safety-mib", type=_nonnegative_int, default=512)
    parser.add_argument("--use-half", action="store_true")
    parser.add_argument("--disable-vad", action="store_true")
    parser.add_argument("--disable-punc", action="store_true")
    return parser


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("segment batch size must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return result


def main(argv: list[str] | None = None) -> int:
    protocol_output = sys.stdout
    sys.stdout = sys.stderr
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve(
        sys.stdin,
        protocol_output,
        lambda: build_firered_system(
            args.model_root,
            args.device,
            args.segment_batch_size,
            args.segment_frame_budget,
            args.segment_memory_limit_mib,
            args.segment_memory_safety_mib,
            args.use_half,
            not args.disable_vad,
            not args.disable_punc,
        ),
        peak_sampler=_peak_gpu_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
