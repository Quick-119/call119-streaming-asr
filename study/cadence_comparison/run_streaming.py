"""同じ凍結重み・10通話バッチで1秒/2秒入力のCERを再評価する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from common import MODEL_ROOTS, REPO, RUNS, SPLITS, baseline_for, bundle_for, rows_for, save_json, score_call, sha, sum_scores, verify_inputs
sys.path.insert(0, str(REPO / "runtime"))
from shared_streaming import SharedStreamingASR  # noqa: E402
from variable_decoder_batching import VariableDecoderBatcher  # noqa: E402


def pcm_for(row: dict) -> np.ndarray:
    path = Path(row["audio_path"])
    if sha(path) != row["audio_sha256"]:
        raise ValueError(f"音声ハッシュが異なります: {row['call_id']}")
    pcm, sr = sf.read(path, dtype="float32", always_2d=True)
    pcm = pcm.mean(axis=1)
    if sr != 16000:
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
    return np.ascontiguousarray(np.clip(pcm, -1, 1), dtype=np.float32)


def evaluate_group(engine: SharedStreamingASR, batcher: VariableDecoderBatcher,
                   rows: list[dict], chunk_sec: float) -> dict:
    audio = {row["call_id"]: pcm_for(row) for row in rows}
    ids = list(audio)
    fragments = {cid: [] for cid in ids}
    updates = {cid: 0 for cid in ids}
    for cid in ids:
        engine.open(cid)
    torch.cuda.empty_cache()
    block = round(16000 * chunk_sec)
    begin = time.perf_counter()
    for offset in range(0, max(map(len, audio.values())), block):
        chunks = {cid: pcm[offset:min(len(pcm), offset + block)]
                  for cid, pcm in audio.items() if offset < len(pcm)}
        if chunks:
            produced = batcher.push_many(chunks)
            torch.cuda.synchronize()
            for cid, result in produced.items():
                fragments[cid].append(result.get("text", ""))
                updates[cid] += 1
        finishing = [cid for cid, pcm in audio.items() if offset < len(pcm) <= offset + block]
        if finishing:
            produced = batcher.finish_many(finishing)
            torch.cuda.synchronize()
            for cid in finishing:
                fragments[cid].append(produced[cid].get("text", ""))
                updates[cid] += 1
    if engine.sessions:
        raise RuntimeError("通話状態が残っています")
    elapsed = time.perf_counter() - begin
    scored = []
    for row in rows:
        cid = row["call_id"]
        hypothesis = "".join(fragments[cid])
        score = score_call(row, hypothesis)
        scored.append({"call_id": cid, "primary": score,
                       "transcript_sha256": hashlib.sha256(hypothesis.encode()).hexdigest(),
                       "updates": updates[cid], "duration_sec": len(audio[cid]) / 16000})
    return {"calls": len(rows), "audio_sec": sum(map(len, audio.values())) / 16000,
            "inference_wall_sec": elapsed, "primary": sum_scores(scored),
            "per_call_private_aggregate": scored,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_ROOTS, required=True)
    parser.add_argument("--chunk-sec", type=float, choices=(1.0, 2.0), required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--max-parts", type=int, help="動作確認用。分割ごとの先頭バッチ数")
    args = parser.parse_args()
    label = args.model
    cadence = f"stream_{int(args.chunk_sec)}s"
    bundle = bundle_for(label)
    model_manifest = json.loads((bundle / "BUNDLE.json").read_text())
    model_sha = model_manifest["files_sha256"]["model/whisper.pt"]
    for split in args.splits:
        if baseline_for(label, split)["model_sha256"] != model_sha:
            raise ValueError(f"チェックポイント不一致: {label}/{split}")
        verify_inputs(label, split, rows_for(split))
    engine = SharedStreamingASR(bundle, max_sessions=20, chunk_sec=args.chunk_sec,
                                beam=3, frame_threshold=75, fp16_weights=True,
                                strict_chunk_cadence=True, cross_sdpa_alignment=True)
    batcher = VariableDecoderBatcher(engine, 20, 64)
    if engine.cfg["chunk_sec"] != args.chunk_sec:
        raise ValueError("入力間隔の設定が反映されていません")
    for split in args.splits:
        rows = rows_for(split)
        baseline = baseline_for(label, split)
        directory = RUNS / "results" / cadence / label / split
        parts = []
        for part_number, start in enumerate(range(0, len(rows), 10)):
            if args.max_parts is not None and part_number >= args.max_parts:
                break
            group = rows[start:start + 10]
            original_manifest = MODEL_ROOTS[label] / "evaluation" / label / split / f"manifest_{part_number:02d}.jsonl"
            original_ids = [json.loads(line)["call_id"] for line in original_manifest.read_text().splitlines() if line]
            if original_ids != [row["call_id"] for row in group]:
                raise ValueError(f"既存評価の通話バッチと異なります: {label}/{split}/{part_number}")
            output = directory / f"part_{part_number:02d}.json"
            if output.exists():
                result = json.loads(output.read_text())
                if (result["model_sha256"] != model_sha or result["chunk_sec"] != args.chunk_sec or
                        result["input_manifest_sha256"] != sha(original_manifest) or
                        [item["call_id"] for item in result["per_call_private_aggregate"]] != original_ids):
                    raise ValueError(f"再開時の記録が異なります: {output}")
                print("reusing", cadence, label, split, part_number, flush=True)
            else:
                torch.cuda.reset_peak_memory_stats()
                result = evaluate_group(engine, batcher, group, args.chunk_sec)
                result.update({"model": label, "mode": cadence, "chunk_sec": args.chunk_sec,
                               "model_sha256": model_sha, "input_manifest_sha256": sha(original_manifest),
                               "settings": {"beam": 3, "frame_threshold": 75, "fp16_weights": True,
                                            "strict_chunk_cadence": True, "cross_sdpa_alignment": True,
                                            "decoder_microbatch": 20, "decoder_offset_bucket": 64,
                                            "decoder_batch_finish": True}})
                save_json(output, result)
                print("completed", cadence, label, split, part_number,
                      f"CER={result['primary']['micro_cer']:.6f}", flush=True)
            parts.append(result)
        if len(parts) * 10 < len(rows) and args.max_parts is not None:
            print("pilot complete", cadence, label, split, len(parts), "parts", flush=True)
            continue
        scored = [item for part in parts for item in part["per_call_private_aggregate"]]
        if [item["call_id"] for item in scored] != [row["call_id"] for row in rows]:
            raise ValueError(f"通話の順序が異なります: {label}/{split}")
        total = sum_scores(scored)
        if total["N"] != baseline["primary"]["N"]:
            raise ValueError(f"評価文字数が0.5秒評価と異なります: {label}/{split}")
        save_json(directory / "COMBINED.json", {"model": label, "mode": cadence, "calls": len(rows),
                 "manifest_sha256": baseline["manifest_sha256"], "model_sha256": model_sha,
                 "primary": total, "per_call_private_aggregate": scored,
                 "inference_wall_sec": sum(part["inference_wall_sec"] for part in parts),
                 "peak_reserved_bytes": max(part["peak_reserved_bytes"] for part in parts)})
        print("split complete", cadence, label, split, f"CER={total['micro_cer']:.6f}", flush=True)


if __name__ == "__main__":
    main()
