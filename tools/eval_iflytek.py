"""Evaluate iFLYTEK IFASR transcription on local audio with word/sentence timestamps."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hmac
import json
import os
import random
import string
import subprocess
import time
import urllib.parse
import wave
from pathlib import Path

import httpx

DEFAULT_HOST = "https://office-api-ist-dx.iflyaisol.com"
UPLOAD_PATH = "/v2/upload"
RESULT_PATH = "/v2/getResult"


def random_string(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=length))


def local_datetime_with_tz() -> str:
    now = dt.datetime.now().astimezone()
    return now.strftime("%Y-%m-%dT%H:%M:%S%z")


def sign_params(params: dict[str, object], secret: str) -> str:
    parts = []
    for key, value in sorted(params.items(), key=lambda item: item[0]):
        if value is None or str(value).strip() == "":
            continue
        encoded_key = urllib.parse.quote(str(key), safe="")
        encoded_value = urllib.parse.quote(str(value), safe="")
        parts.append(f"{encoded_key}={encoded_value}")
    base_string = "&".join(parts)
    digest = hmac.new(secret.encode("utf-8"), base_string.encode("utf-8"), digestmod="sha1").digest()
    return base64.b64encode(digest).decode("utf-8")


def build_url(host: str, path: str, params: dict[str, object]) -> str:
    clean = {key: str(value) for key, value in params.items() if value is not None and str(value).strip() != ""}
    query = urllib.parse.urlencode(clean)
    return f"{host}{path}?{query}"


def get_audio_duration_ms(path: Path) -> int:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                return int(round(wf.getnframes() / wf.getframerate() * 1000))
        except Exception:
            pass
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(round(float(res.stdout.strip()) * 1000))


def format_timestamp(ms: int) -> str:
    total_seconds = max(0, ms // 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def read_env(env_path: Path) -> dict[str, str]:
    if not env_path.is_file():
        return {}
    res = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        res[k.strip()] = v.strip().strip("'\"")
    return res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", "-a", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path("/Users/jiajun/HMC/projects/ai_face_demo/Ifasr/.env"))
    parser.add_argument("--language", default="autodialect")
    args = parser.parse_args()

    cfg = read_env(args.env)
    for k in ("IFASR_APP_ID", "IFASR_ACCESS_KEY_ID", "IFASR_ACCESS_KEY_SECRET"):
        if k in os.environ:
            cfg[k] = os.environ[k]
        if not cfg.get(k):
            raise RuntimeError(f"Missing {k} in environment or {args.env}")

    audio_path = args.audio.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    app_id = cfg["IFASR_APP_ID"]
    key_id = cfg["IFASR_ACCESS_KEY_ID"]
    secret = cfg["IFASR_ACCESS_KEY_SECRET"]
    host = cfg.get("IFASR_HOST", DEFAULT_HOST).rstrip("/")

    duration_ms = get_audio_duration_ms(audio_path)
    file_size = audio_path.stat().st_size
    sig_rand = random_string()

    params = {
        "appId": app_id,
        "accessKeyId": key_id,
        "dateTime": local_datetime_with_tz(),
        "signatureRandom": sig_rand,
        "fileSize": str(file_size),
        "fileName": audio_path.name,
        "language": args.language,
        "duration": str(duration_ms),
        "roleType": "1",
        "roleNum": "0",
        "audioMode": "fileStream",
    }
    headers = {
        "Content-Type": "application/octet-stream",
        "signature": sign_params(params, secret),
    }
    upload_url = build_url(host, UPLOAD_PATH, params)

    print(f"Uploading {audio_path.name} ({duration_ms} ms, {file_size} bytes) to iFLYTEK IFASR...")
    with audio_path.open("rb") as f:
        resp = httpx.post(upload_url, headers=headers, content=f.read(), timeout=60)
    data = resp.json()
    if str(data.get("code")) != "000000":
        raise RuntimeError(f"Upload failed: {data}")

    order_id = data["content"]["orderId"]
    print(f"Order created: {order_id}. Polling results...")

    # Poll
    deadline = time.monotonic() + 1800
    while True:
        poll_params = {
            "appId": app_id,
            "accessKeyId": key_id,
            "dateTime": local_datetime_with_tz(),
            "ts": str(int(time.time())),
            "orderId": order_id,
            "signatureRandom": sig_rand,
            "resultType": "transfer",
        }
        poll_headers = {
            "Content-Type": "application/json",
            "signature": sign_params(poll_params, secret),
        }
        res_url = build_url(host, RESULT_PATH, poll_params)
        r = httpx.post(res_url, headers=poll_headers, json={}, timeout=30).json()
        if str(r.get("code")) != "000000":
            raise RuntimeError(f"Poll failed: {r}")

        status = int(r.get("content", {}).get("orderInfo", {}).get("status", 0))
        if status == 4:
            result_data = r
            break
        if status != 3 and status != 0:
            raise RuntimeError(f"Status error: {r}")
        if time.monotonic() > deadline:
            raise TimeoutError("iFLYTEK transcription timeout")
        print("Processing... waiting 5s")
        time.sleep(5)

    # Save raw
    raw_path = out_dir / "ifasr_raw.json"
    raw_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved raw response to {raw_path}")

    # Parse lattice
    raw_order_res = result_data.get("content", {}).get("orderResult", "{}")
    if isinstance(raw_order_res, str):
        order_res = json.loads(raw_order_res)
    else:
        order_res = raw_order_res

    turns = []
    lines_txt = []
    lattice = order_res.get("lattice", [])
    for idx, item in enumerate(lattice, 1):
        spk = item.get("spk", "1")
        # parse json_1best
        one_best = json.loads(item.get("json_1best", "{}"))
        st = one_best.get("st", {})
        bg = int(st.get("bg", 0))
        ed = int(st.get("ed", 0))
        words = []
        for rt in st.get("rt", []):
            for ws in rt.get("ws", []):
                for cw in ws.get("cw", []):
                    w = cw.get("w", "")
                    if w:
                        words.append(w)
        text = "".join(words)
        if text.strip():
            turn_obj = {
                "index": idx,
                "speaker": int(spk) if str(spk).isdigit() else spk,
                "start_ms": bg,
                "end_ms": ed,
                "text": text,
            }
            turns.append(turn_obj)
            lines_txt.append(f"[{format_timestamp(bg)}] 说话人{spk}：{text}")

    txt_path = out_dir / "transcript.txt"
    txt_path.write_text("\n".join(lines_txt) + "\n", encoding="utf-8")

    json_payload = {
        "status": "success",
        "speaker_count": len({t["speaker"] for t in turns}),
        "turns": turns,
        "full_text": "".join(t["text"] for t in turns),
    }
    json_path = out_dir / "transcript.json"
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("=== iFLYTEK Conversational Turns ===")
    print("=" * 70)
    for l in lines_txt:
        print(l)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
