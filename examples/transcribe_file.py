#!/usr/bin/env python3
"""Transcribe one audio file with an external frozen 119-call ASR bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from shared_streaming import SharedStreamingASR  # noqa: E402
from postprocess import correct_transcript  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True, help="Complete external frozen bundle")
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="New output directory")
    args = ap.parse_args()
    bundle, audio_path, output = args.bundle.resolve(), args.audio.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    pcm, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    pcm = pcm.mean(axis=1)
    if sr != 16000:
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
    pcm = np.ascontiguousarray(np.clip(pcm, -1, 1), dtype=np.float32)
    engine = SharedStreamingASR(bundle, max_sessions=1, chunk_sec=0.5, beam=3,
                                frame_threshold=75, strict_chunk_cadence=True,
                                cross_sdpa_alignment=True, fp16_weights=True)
    engine.open("input")
    parts = []
    for offset in range(0, len(pcm), 8000):
        parts.append(engine.push("input", pcm[offset:offset + 8000]).get("text", ""))
    parts.append(engine.finish("input").get("text", ""))
    raw_transcript = "".join(parts)
    transcript = correct_transcript(raw_transcript)
    output.mkdir(parents=True)
    (output / "transcript.raw.txt").write_text(raw_transcript + "\n")
    (output / "transcript.txt").write_text(transcript + "\n")
    (output / "run.json").write_text(json.dumps({
        "audio_sha256": sha(audio_path), "bundle_manifest_sha256": sha(bundle / "BUNDLE.json"),
        "model_sha256": json.loads((bundle / "BUNDLE.json").read_text())["files_sha256"]["model/whisper.pt"],
        "input_sample_rate": sr, "duration_sec": len(pcm) / 16000,
        "chunk_sec": 0.5, "beam": 3, "frame_threshold": 75,
        "strict_chunk_cadence": True, "cross_sdpa_alignment": True,
        "fp16_weights": True,
        "postprocess": "runtime/postprocess.py",
    }, ensure_ascii=False, indent=2) + "\n")
    print(output / "transcript.txt")


if __name__ == "__main__":
    main()
