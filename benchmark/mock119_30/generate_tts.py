#!/usr/bin/env python3
"""Generate the fixed mock calls with Gemini 3.8 Flash TTS (REST Interactions API)."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent
MODEL = "gemini-3.8-flash-tts"
URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_body(turns: list[dict]) -> dict:
    content = []
    for turn in turns:
        style = ("calm, clear Japanese emergency dispatcher" if turn["speaker"] == "Dispatcher"
                 else "worried and urgent Japanese caller, speaking naturally")
        content.append({"type": "text", "text": turn["text"], "annotations": [{
            "type": "speech_metadata", "speaker": turn["speaker"], "style": style,
        }]})
    return {
        "model": MODEL,
        "input": [{"type": "user_input", "content": content}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": {"mode": "conversational", "speakers": [
            {"speaker": "Dispatcher", "voice": "Kore"},
            {"speaker": "Caller", "voice": "Puck"},
        ]}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key-file", type=Path, help="File containing API key; alternatively GEMINI_API_KEY")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    key = (args.api_key_file.read_text().strip() if args.api_key_file
           else os.environ.get("GEMINI_API_KEY", "").strip())
    if not key:
        raise SystemExit("Set GEMINI_API_KEY or --api-key-file")
    rows = json.loads((ROOT / "scripts.json").read_text())[:args.limit]
    dest = ROOT / "audio" / "gemini_clean"
    dest.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for row in rows:
        wav = dest / f'{row["id"]}.wav'
        meta = dest / f'{row["id"]}.json'
        body = request_body(row["turns"])
        body_sha = digest(json.dumps(body, ensure_ascii=False, sort_keys=True).encode())
        if wav.exists() and meta.exists():
            previous = json.loads(meta.read_text())
            if previous["request_sha256"] == body_sha and previous["audio_sha256"] == digest(wav.read_bytes()):
                print("existing", row["id"], flush=True)
                continue
            raise RuntimeError(f"Existing audio differs from request: {wav}")
        response = session.post(URL, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                                json=body, timeout=300)
        if not response.ok:
            try:
                error_body = response.json()
                if isinstance(error_body, list):
                    error_body = error_body[0]
                error = error_body.get("error", {})
                detail = error.get("message", "").strip()
            except (ValueError, AttributeError):
                detail = "Unparseable provider response"
            raise RuntimeError(f"Gemini TTS HTTP {response.status_code}: {detail}")
        data = response.json()
        if isinstance(data, list):
            data = data[0]
        blocks = [content for step in data.get("steps", []) if step.get("type") == "model_output"
                  for content in step.get("content", []) if content.get("type") == "audio"]
        if not blocks:
            raise RuntimeError(f"No audio returned for {row['id']}")
        raw = base64.b64decode(blocks[-1]["data"])
        if raw[:4] != b"RIFF":
            raise RuntimeError(f"Expected WAV audio for {row['id']}")
        with sf.SoundFile(io.BytesIO(raw)) as handle:
            sr, frames, channels = handle.samplerate, len(handle), handle.channels
        if frames == 0:
            raise RuntimeError(f"Empty audio for {row['id']}")
        wav.write_bytes(raw)
        meta.write_text(json.dumps({"id": row["id"], "model": MODEL,
                                    "request_sha256": body_sha, "audio_sha256": digest(raw),
                                    "sample_rate": sr, "channels": channels,
                                    "duration_sec": frames / sr,
                                    "created_unix": time.time()}, ensure_ascii=False, indent=2) + "\n")
        print("generated", row["id"], f"{frames / sr:.1f}s", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
