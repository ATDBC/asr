"""Isolated pyannote OSD and ClearVoice worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


TARGET_SAMPLE_RATE = 16000


def _configure_cache(request: dict[str, Any]) -> None:
    if not request.get("cache_dir"):
        return
    cache_dir = Path(request["cache_dir"]).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.chdir(cache_dir)


def _load_detector(request: dict[str, Any]):
    _configure_cache(request)
    import torch
    from pyannote.audio import Pipeline

    token_name = str(request.get("hf_token_env") or "HF_TOKEN")
    token = os.getenv(token_name)
    if not token:
        raise RuntimeError(f"required Hugging Face token is not set: {token_name}")
    model_id = str(
        request.get("model_id") or "pyannote/speaker-diarization-3.1"
    )
    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
    device = str(request.get("device") or "cuda:0")
    if device.startswith("cuda"):
        pipeline.to(torch.device(device))
    return pipeline, model_id


def _detect(request: dict[str, Any]) -> dict[str, Any]:
    pipeline, model_id = _load_detector(request)
    output = pipeline(str(Path(request["audio_path"]).resolve()))
    windows = [
        {
            "id": f"overlap-{index:04d}",
            "start_ms": int(round(segment.start * 1000)),
            "end_ms": int(round(segment.end * 1000)),
        }
        for index, segment in enumerate(
            output.get_overlap().support(),
            start=1,
        )
        if segment.end > segment.start
    ]
    return {"ok": True, "windows": windows, "model": model_id}


def _read_segment(audio_path: Path, start_ms: int, end_ms: int):
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(str(audio_path)) as audio:
        start_frame = max(0, int(round(start_ms * audio.samplerate / 1000)))
        end_frame = min(
            len(audio),
            int(round(end_ms * audio.samplerate / 1000)),
        )
        if end_frame <= start_frame:
            raise ValueError(f"empty overlap window: {start_ms}..{end_ms}")
        audio.seek(start_frame)
        data = audio.read(
            frames=end_frame - start_frame,
            dtype="float32",
            always_2d=True,
        )
    return np.asarray(data.mean(axis=1), dtype="float32")


def _write_wav(path: Path, data) -> None:
    import numpy as np
    import soundfile as sf

    values = np.nan_to_num(
        np.asarray(data, dtype="float32").reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.clip(values, -1.0, 1.0),
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
    )


def _sources(result):
    import numpy as np

    if isinstance(result, dict):
        result = next(iter(result.values()))
    if isinstance(result, np.ndarray):
        if result.ndim >= 2:
            return [result[index].reshape(-1) for index in range(result.shape[0])]
        return [result.reshape(-1)]
    if isinstance(result, (list, tuple)):
        return [np.asarray(item, dtype="float32").reshape(-1) for item in result]
    raise TypeError(f"unsupported ClearVoice result type: {type(result)!r}")


def _rms(data) -> float:
    import numpy as np

    values = np.asarray(data, dtype="float32")
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _separate(request: dict[str, Any]) -> dict[str, Any]:
    _configure_cache(request)
    from clearvoice import ClearVoice

    audio_path = Path(request["audio_path"]).resolve()
    output_dir = Path(request["output_dir"]).resolve()
    model_name = str(request.get("model") or "MossFormer2_SS_16K")
    model = ClearVoice(task="speech_separation", model_names=[model_name])
    response_windows = []
    for window in request.get("windows", []):
        window_id = str(window["id"])
        start_ms = int(window["start_ms"])
        end_ms = int(window["end_ms"])
        mixture = _read_segment(audio_path, start_ms, end_ms)
        mix_path = output_dir / f"{window_id}-mix.wav"
        _write_wav(mix_path, mixture)
        tracks = []
        for index, source in enumerate(
            _sources(model(str(mix_path), online_write=False)),
            start=1,
        ):
            path = output_dir / f"{window_id}-s{index}.wav"
            _write_wav(path, source)
            tracks.append(
                {
                    "index": index,
                    "path": str(path),
                    "rms": _rms(source),
                    "duration_ms": int(
                        round(len(source) * 1000 / TARGET_SAMPLE_RATE)
                    ),
                }
            )
        response_windows.append({**window, "tracks": tracks})
    return {"ok": True, "model": model_name, "windows": response_windows}


def _download(request: dict[str, Any]) -> dict[str, Any]:
    _load_detector(request)
    from clearvoice import ClearVoice

    model_name = str(request.get("model") or "MossFormer2_SS_16K")
    ClearVoice(task="speech_separation", model_names=[model_name])
    return {"ok": True, "model": model_name}


def _verify(request: dict[str, Any]) -> dict[str, Any]:
    previous = {
        name: os.environ.get(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        return _download(request)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run(request: dict[str, Any]) -> dict[str, Any]:
    _configure_cache(request)
    action = request.get("action")
    if action == "detect":
        return _detect(request)
    if action == "separate":
        return _separate(request)
    if action == "download":
        return _download(request)
    if action == "verify":
        return _verify(request)
    raise ValueError(f"unsupported action: {action}")


def main(argv: list[str] | None = None) -> int:
    arguments = argv or sys.argv[1:]
    if len(arguments) != 1:
        print("usage: overlap_worker.py REQUEST_JSON", file=sys.stderr)
        return 2
    request_path = Path(arguments[0]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = Path(request["result_path"]).resolve()
    try:
        response = run(request)
        return_code = 0
    except Exception as exc:
        response = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 2
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
