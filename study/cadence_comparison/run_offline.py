"""同じ凍結Whisper重みを標準の全通話オフライン復号で評価する。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from common import MODEL_ROOTS, RUNS, SPLITS, baseline_for, bundle_for, rows_for, save_json, score_call, sha, sum_scores, verify_inputs

PROTOCOL = {"algorithm": "Whisper standard offline", "language": "ja",
            "task": "transcribe", "beam_size": 3,
            "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "condition_on_previous_text": True, "fp16": True,
            "internal_window_sec": 30, "external_vad": False,
            "mel_bins": 128, "tokenizer_languages": 100,
            "seed_rule": "261013 + numeric call ID"}


def pcm_for(row: dict) -> np.ndarray:
    path = Path(row["audio_path"])
    if sha(path) != row["audio_sha256"]:
        raise ValueError(f"音声ハッシュが異なります: {row['call_id']}")
    pcm, sr = sf.read(path, dtype="float32", always_2d=True)
    pcm = pcm.mean(axis=1)
    if sr != 16000:
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
    return np.ascontiguousarray(np.clip(pcm, -1, 1), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_ROOTS, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--max-calls", type=int, help="動作確認用。分割ごとの先頭通話数")
    args = parser.parse_args()
    label = args.model
    bundle = bundle_for(label)
    manifest = json.loads((bundle / "BUNDLE.json").read_text())
    model_sha = manifest["files_sha256"]["model/whisper.pt"]
    for split in args.splits:
        if baseline_for(label, split)["model_sha256"] != model_sha:
            raise ValueError(f"チェックポイント不一致: {label}/{split}")
        verify_inputs(label, split, rows_for(split))
    vendor = bundle / "vendor/SimulStreaming"
    sys.path.insert(0, str(vendor))
    from simulstreaming.whisper.simul_whisper.whisper import load_model  # noqa: E402
    transcribe_module = importlib.import_module("simulstreaming.whisper.simul_whisper.whisper.transcribe")
    decoding_module = importlib.import_module("simulstreaming.whisper.simul_whisper.whisper.decoding")

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(str(bundle / "model/whisper.pt"), device="cpu")
    if model.num_languages != PROTOCOL["tokenizer_languages"]:
        raise ValueError("large-v3の言語トークン数が想定と異なります")
    # 同梱版のtranscribe()はmel数の既定値80を固定している。large-v3の重みは128。
    original_mel = transcribe_module.log_mel_spectrogram
    def model_mel(audio, *, padding=0):
        return original_mel(audio, n_mels=model.dims.n_mels, padding=padding)
    transcribe_module.log_mel_spectrogram = model_mel
    # 同梱版の標準復号経路は言語トークン数99を既定値にしている。
    # large-v3の100に合わせないと、全特殊トークンが1個ずれて英字の誤復号になる。
    from simulstreaming.whisper.simul_whisper.whisper import tokenizer as tokenizer_module
    original_tokenizer = tokenizer_module.get_tokenizer
    def model_tokenizer(multilingual, **kwargs):
        kwargs.setdefault("num_languages", model.num_languages)
        return original_tokenizer(multilingual, **kwargs)
    transcribe_module.get_tokenizer = model_tokenizer
    decoding_module.get_tokenizer = model_tokenizer
    model.half().to("cuda").eval()
    torch.cuda.empty_cache()
    for split in args.splits:
        rows = rows_for(split)
        baseline = baseline_for(label, split)
        directory = RUNS / "results" / "offline_full_call" / label / split
        scored = []
        for index, row in enumerate(rows, 1):
            if args.max_calls is not None and index > args.max_calls:
                break
            cid = row["call_id"]
            output = directory / f"{cid}.json"
            if output.exists():
                item = json.loads(output.read_text())
                if (item["model_sha256"] != model_sha or item["call_id"] != cid or
                        item["audio_sha256"] != row["audio_sha256"] or
                        item.get("protocol") != PROTOCOL or
                        item["primary"]["N"] != row["known_characters"]):
                    raise ValueError(f"再開時の記録が異なります: {output}")
                print("reusing", label, split, index, cid, flush=True)
            else:
                pcm = pcm_for(row)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                torch.manual_seed(261013 + int(cid.rsplit("_", 1)[1]))
                with torch.inference_mode():
                    decoded = transcribe_module.transcribe(model, pcm, language="ja", task="transcribe",
                                         beam_size=3, temperature=tuple(PROTOCOL["temperatures"]), fp16=True,
                                         condition_on_previous_text=True, verbose=None)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                hypothesis = decoded["text"]
                score = score_call(row, hypothesis)
                item = {"call_id": cid, "audio_sha256": row["audio_sha256"],
                        "model_sha256": model_sha, "protocol": PROTOCOL, "primary": score,
                        "transcript_sha256": hashlib.sha256(hypothesis.encode()).hexdigest(),
                        "duration_sec": len(pcm) / 16000, "inference_wall_sec": elapsed,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                save_json(output, item)
                print("completed", label, split, index, cid,
                      f"CER={score['cer']:.6f}", flush=True)
            scored.append(item)
        if len(scored) < len(rows) and args.max_calls is not None:
            print("pilot complete", label, split, len(scored), "calls", flush=True)
            continue
        if [item["call_id"] for item in scored] != [row["call_id"] for row in rows]:
            raise ValueError(f"通話の順序が異なります: {label}/{split}")
        total = sum_scores(scored)
        if total["N"] != baseline["primary"]["N"]:
            raise ValueError(f"評価文字数が0.5秒評価と異なります: {label}/{split}")
        save_json(directory / "COMBINED.json", {"model": label, "mode": "offline_full_call",
                 "calls": len(rows), "manifest_sha256": baseline["manifest_sha256"],
                 "model_sha256": model_sha, "primary": total,
                 "decoder": PROTOCOL,
                 "per_call_private_aggregate": scored,
                 "inference_wall_sec": sum(item["inference_wall_sec"] for item in scored),
                 "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in scored)})
        print("split complete", label, split, f"CER={total['micro_cer']:.6f}", flush=True)


if __name__ == "__main__":
    main()
