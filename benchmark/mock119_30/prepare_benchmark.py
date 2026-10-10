#!/usr/bin/env python3
"""Turn clean Gemini dialogue into fixed telephone-style benchmark audio and manifest."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent
from scoring import normalize_text

# Mild, representative, and busy profiles based on the cited project's measured
# telephone-band / SNR table. This implementation is independent of its code.
PROFILES = {
    "light": {"highpass_hz": 180, "lowpass_hz": 3700, "noise": "pink", "snr_db": 25,
              "compress_ratio": 1.5, "codec_roundtrip": False, "hum_dbfs": None},
    "median": {"highpass_hz": 220, "lowpass_hz": 3600, "noise": "pink", "snr_db": 21,
               "compress_ratio": 2.0, "codec_roundtrip": True, "hum_dbfs": -42},
    "busy": {"highpass_hz": 280, "lowpass_hz": 3300, "noise": "brown", "snr_db": 16,
             "compress_ratio": 3.0, "codec_roundtrip": True, "hum_dbfs": -34},
}
SAMPLE_RATE = 8000
TARGET_RMS_DBFS = -13.23


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def colored_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freq = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    scale = np.ones_like(freq)
    if kind == "pink":
        scale[1:] = 1 / np.sqrt(freq[1:])
    elif kind == "brown":
        scale[1:] = 1 / freq[1:]
    else:
        raise ValueError(kind)
    scale[0] = 0
    out = np.fft.irfft(spectrum * scale, n=n)
    return out / max(rms(out), 1e-12)


def compress(x: np.ndarray, ratio: float) -> np.ndarray:
    # Feed-forward RMS compressor with 20 ms detector and a -22 dBFS threshold.
    detector = np.sqrt(signal.lfilter(np.full(160, 1 / 160), [1], x * x) + 1e-12)
    threshold = 10 ** (-22 / 20)
    gain = np.minimum(1, (threshold / np.maximum(detector, 1e-8)) ** (1 - 1 / ratio))
    return x * gain


def mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    sf.write(buffer, x, SAMPLE_RATE, format="WAV", subtype="ULAW")
    buffer.seek(0)
    y, _ = sf.read(buffer, dtype="float32")
    return y.astype(np.float64)


def process(src: Path, dest: Path, profile: dict, seed: int) -> dict:
    audio, sr = sf.read(src, always_2d=True, dtype="float64")
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        divisor = gcd(sr, SAMPLE_RATE)
        mono = signal.resample_poly(mono, SAMPLE_RATE // divisor, sr // divisor)
    sos = signal.butter(4, [profile["highpass_hz"], profile["lowpass_hz"]],
                        btype="bandpass", fs=SAMPLE_RATE, output="sos")
    speech = signal.sosfilt(sos, mono)
    speech = compress(speech, profile["compress_ratio"])
    if profile["codec_roundtrip"]:
        speech = mulaw_roundtrip(speech)
    rng = np.random.default_rng(seed)
    noise = colored_noise(len(speech), profile["noise"], rng)
    noise_level = rms(speech) / 10 ** (profile["snr_db"] / 20)
    noise *= noise_level
    hum_dbfs = profile["hum_dbfs"]
    if hum_dbfs is not None:
        t = np.arange(len(speech)) / SAMPLE_RATE
        noise += 10 ** (hum_dbfs / 20) * np.sin(2 * np.pi * 50 * t)
    mixed = speech + noise
    target_rms = 10 ** (TARGET_RMS_DBFS / 20)
    mixed *= target_rms / max(rms(mixed), 1e-12)
    # Smoothly limit peaks while retaining the target RMS. A global peak-scale
    # would make dynamic TTS calls much quieter than the measured call level.
    if np.max(np.abs(mixed)) > 0.95:
        low, high = 0.0, 128.0
        for _ in range(27):
            gain = (low + high) / 2
            limited = 0.95 * np.tanh(gain * mixed / 0.95)
            if rms(limited) < target_rms:
                low = gain
            else:
                high = gain
        mixed = 0.95 * np.tanh(high * mixed / 0.95)
    stereo = np.repeat(mixed.astype("float32")[:, None], 2, axis=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, stereo, SAMPLE_RATE, format="WAV", subtype="ULAW")
    decoded, read_sr = sf.read(dest, always_2d=True, dtype="float32")
    assert read_sr == SAMPLE_RATE and decoded.shape[1] == 2
    return {"duration_sec": len(decoded) / SAMPLE_RATE, "sample_rate": SAMPLE_RATE,
            "channels": 2, "codec": "G.711 mu-law", "rms_dbfs": float(20 * np.log10(rms(decoded))),
            "peak_dbfs": float(20 * np.log10(np.max(np.abs(decoded)) + 1e-12)),
            "seed": seed, "speech_rms_before_mix": rms(speech),
            "noise_rms_before_mix": rms(noise)}


def main() -> None:
    rows = json.loads((ROOT / "scripts.json").read_text())
    output = ROOT / "audio" / "telephone"
    manifest = []
    for i, row in enumerate(rows):
        src = ROOT / "audio" / "gemini_clean" / f'{row["id"]}.wav'
        if not src.exists():
            raise FileNotFoundError(f"Generate TTS first: {src}")
        source_hash = sha_bytes(src.read_bytes())
        dest = output / f'{row["id"]}.wav'
        profile_name = row["noise_profile"]
        seed = 119000 + i
        parameters = {"profile": profile_name, **PROFILES[profile_name], "seed": seed}
        meta_file = output / f'{row["id"]}.json'
        if dest.exists() and meta_file.exists():
            acoustic = json.loads(meta_file.read_text())
            if (acoustic["source_sha256"] != source_hash or acoustic["parameters"] != parameters
                    or acoustic["audio_sha256"] != sha_bytes(dest.read_bytes())):
                raise RuntimeError(f"Existing audio differs from fixed benchmark: {dest}")
        else:
            stats = process(src, dest, PROFILES[profile_name], seed)
            acoustic = {"source_sha256": source_hash, "parameters": parameters,
                        "audio_sha256": sha_bytes(dest.read_bytes()), **stats}
            meta_file.write_text(json.dumps(acoustic, ensure_ascii=False, indent=2) + "\n")
        gold = "".join(turn["text"] for turn in row["turns"])
        manifest.append({"call_id": row["id"], "scenario": row["scenario"],
                         "audio_path": str(dest.relative_to(ROOT)), "audio_sha256": acoustic["audio_sha256"],
                         "clean_audio_path": str(src.relative_to(ROOT)), "clean_audio_sha256": source_hash,
                         "noise_profile": profile_name, "gold_text": gold,
                         "gold_normalized": normalize_text(gold),
                         "duration_sec": acoustic["duration_sec"]})
        print(row["id"], f'{acoustic["duration_sec"]:.1f}s', profile_name, flush=True)
    (ROOT / "manifest.jsonl").write_text("".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in manifest))
    print("wrote", ROOT / "manifest.jsonl", flush=True)


if __name__ == "__main__":
    main()
