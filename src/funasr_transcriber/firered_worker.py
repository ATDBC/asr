"""FireRedASR2S JSON Lines worker, executed only inside `.venv-asr`."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, IO, Protocol

from .firered_contract import normalize_official_result


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioInfo:
    sample_rate: int
    channels: int
    subtype: str


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
            if float(end_s) - float(start_s) > 60.0:
                raise ValueError(
                    "FireRedVAD segment exceeds the 60 second AED limit: "
                    f"{start_s}..{end_s}"
                )
        return result, probabilities

    def __getattr__(self, name: str) -> object:
        return getattr(self._vad, name)


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
    """Return concrete module facts after the official system has loaded."""

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
            if action not in {"transcribe", "verify"}:
                raise ValueError(f"unsupported worker action: {action}")
            path = _validate_audio(request.get("audio_path"), audio_info)
            inference_started = clock()
            raw = system.process(str(path), uttid=identifier)
            inference_seconds += clock() - inference_started
            response = normalize_official_result(identifier, raw)
            response["metrics"] = {
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "peak_gpu_bytes": peak_sampler(),
            }
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
) -> FireRedSystem:
    """Construct the pinned official three-module system in the ASR process."""

    import torch
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
    torch.cuda.set_device(device_index)

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
            use_half=False,
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
        enable_vad=True,
        enable_lid=False,
        enable_punc=True,
    )
    system = FireRedAsr2System(config)
    system.vad = GuardedVad(system.vad)
    return system


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment-batch-size", type=_positive_int, default=1)
    return parser


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("segment batch size must be positive")
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
        ),
        peak_sampler=_peak_gpu_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
