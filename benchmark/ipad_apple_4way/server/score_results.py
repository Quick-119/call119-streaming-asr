#!/usr/bin/env python3
"""iPadのファイル入力・0.5秒入力の4方式をGold v3.2で採点する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RESEARCH = Path(os.environ["CALL119_ASR_RESEARCH_ROOT"])
DATA = RESEARCH / "outputs/261014_ipad_apple_speech_eval"
sys.path.insert(0, str(RESEARCH / "outputs/261013_three_model_cadence_cer"))
from common import HELD_OUT, baseline_for, rows_for, score_call  # noqa: E402

METHODS = ("sf_file_fixed_ondevice", "st_file_ondevice",
           "sf_stream_05_ondevice", "st_stream_05_ondevice")
STREAM_METHODS = frozenset(METHODS[2:])
SF_METHODS = frozenset((METHODS[0], METHODS[2]))
MODEL = "quality_full_then_human_replay"


def paired_ci(new: list[dict], old: list[dict], draws: int = 10000) -> list[float]:
    rng = random.Random(26101405)
    deltas = []
    for _ in range(draws):
        indexes = [rng.randrange(len(new)) for _ in new]
        denominator = sum(new[i]["N"] for i in indexes)
        deltas.append(100 * sum(new[i]["errors"] - old[i]["errors"] for i in indexes) / denominator)
    deltas.sort()
    return [deltas[int(.025 * draws)], deltas[int(.975 * draws)]]


def baseline_calls() -> dict[str, dict]:
    result = {}
    for split in HELD_OUT:
        data = baseline_for(MODEL, split)
        for item in data["per_call_private_aggregate"]:
            result[item["call_id"]] = item["primary"]
    return result


def totals(items: list[dict]) -> dict:
    result = {key: sum(item[key] for item in items) for key in ("S", "D", "I", "N", "errors")}
    result["CER_percent"] = 100 * result["errors"] / result["N"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", choices=("pilot1", "full126"), required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="採点結果の保存先。省略時はprivate/<bundle>/scored")
    args = parser.parse_args()
    local = DATA / "private" / args.bundle
    manifest_path = local / "ipad_files/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mapping = json.loads((local / "server_mapping_private.json").read_text())
    received = json.loads(args.results.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (received.get("schema_version") != 3 or received.get("app_revision") != 1
            or received.get("dataset_id") != manifest["dataset_id"]
            or received.get("manifest_sha256") != expected_hash):
        raise ValueError("iPad結果の方式・評価集合・manifestが一致しません")
    cases = {case["id"]: case for case in manifest["cases"]}
    records = {}
    for item in received.get("records", []):
        key = (item["method"], item["id"])
        if key in records or item["method"] not in METHODS or item["id"] not in cases:
            raise ValueError(f"重複または未知の結果: {key}")
        case = cases[item["id"]]
        if item["audio_sha256"] != case["sha256"] or abs(item["duration_sec"] - case["duration_sec"]) > .02:
            raise ValueError(f"iPad入力音声が一致しません: {key}")
        if item["status"] == "ok" and abs(item["fed_audio_sec"] - case["duration_sec"]) > .02:
            raise ValueError(f"成功とされた通話の音声投入量が不足: {key}")
        if item["status"] == "ok" and item["method"] in SF_METHODS:
            diagnostics = item.get("sf_segments")
            if not isinstance(diagnostics, list) or len(diagnostics) != len(case["segments"]):
                raise ValueError(f"SFSpeech区間診断が不足: {key}")
            for part, diagnostic in zip(case["segments"], diagnostics):
                expected_audio = (part["end_sample"] - part["start_sample"]) / 16000
                input_audio = float(diagnostic["input_audio_sec"])
                processed = float(diagnostic["processed_audio_sec"])
                if (abs(input_audio - expected_audio) > .02 or processed <= 0
                        or processed + 1.5 < input_audio):
                    raise ValueError(f"SFSpeechで区間全体の処理を確認できません: {key}")
        if item["status"] == "ok" and item["method"] in STREAM_METHODS:
            feeds = item.get("feed_events")
            if not isinstance(feeds, list) or not feeds:
                raise ValueError(f"0.5秒入力時刻の記録がありません: {key}")
            last_audio = 0.0
            last_wall = 0.0
            for event in feeds:
                audio = float(event["audio_end_sec"])
                wall = float(event["sent_at_sec"])
                if (not last_audio < audio <= case["duration_sec"] + .02 or
                        audio - last_audio > .501 or wall < last_wall or
                        wall < audio - .1 or wall > float(item["wall_sec"]) + .1):
                    raise ValueError(f"入力時刻または0.5秒間隔が不正: {key}")
                last_audio, last_wall = audio, wall
            if abs(last_audio - case["duration_sec"]) > .02:
                raise ValueError(f"通話終端まで投入されていません: {key}")
        records[key] = item
    source_rows = {row["call_id"]: row for split in HELD_OUT for row in rows_for(split)}
    baseline = baseline_calls()
    selected = [mapping[case["id"]]["call_id"] for case in manifest["cases"]]
    whisper = totals([baseline[call_id] for call_id in selected])
    output = {
        "dataset_id": manifest["dataset_id"], "calls": len(cases),
        "manifest_sha256": expected_hash,
        "ipad_results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "protocol": "four_way_file_and_real_time_0.5_sec_input",
        "existing_whisper_stream_0.5s": whisper, "apple": {},
    }
    private = {}
    for method in METHODS:
        per_call = []
        scores = []
        successful = []
        event_lags = []
        feed_lags = []
        feed_chunks = 0
        misses = 0
        wall = []
        for case in manifest["cases"]:
            identity = case["id"]
            row = source_rows[mapping[identity]["call_id"]]
            item = records.get((method, identity))
            ok = item is not None and item.get("status") == "ok" and isinstance(item.get("text"), str)
            hypothesis = item["text"] if ok else ""
            score = score_call(row, hypothesis)
            if score["N"] != mapping[identity]["known_characters"]:
                raise ValueError("Gold分母が既存値と一致しません")
            score = {key: score[key] for key in ("S", "D", "I", "N", "errors", "cer")}
            scores.append(score)
            if ok:
                successful.append(score)
                wall.append(float(item["wall_sec"]))
                misses += int(item["deadline_miss_count"])
                event_lags.extend(float(event["arrival_sec"]) - float(event["audio_end_sec"])
                                  for event in item.get("events", []))
                feed_lags.extend(float(event["sent_at_sec"]) - float(event["audio_end_sec"])
                                 for event in item["feed_events"])
                feed_chunks += len(item["feed_events"])
            per_call.append({
                "id": identity, "call_id": row["call_id"], "split": row["split"],
                "status": item.get("status") if item else "missing", "score": score,
                "text_sha256": hashlib.sha256(hypothesis.encode()).hexdigest(),
                "error": item.get("error") if item and not ok else None,
                "events": len(item.get("events", [])) if item else 0,
            })
        all_totals = totals(scores)
        success_cer = (totals(successful)["CER_percent"] if successful else None)
        summary = {
            "success_calls": len(successful), "failed_or_missing_calls": len(cases) - len(successful),
            "coverage_percent": 100 * len(successful) / len(cases),
            "complete_set_CER_percent_missing_as_empty": all_totals["CER_percent"],
            "success_only_CER_percent": success_cer,
            **{key: all_totals[key] for key in ("S", "D", "I", "N", "errors")},
            "wall_sec_total_successes": sum(wall),
            "wall_sec_median_successes": statistics.median(wall) if wall else None,
            "arrival_minus_audio_end_sec_median": statistics.median(event_lags) if event_lags else None,
            "feed_chunk_count": feed_chunks,
            "feed_wall_minus_audio_end_sec_median": statistics.median(feed_lags) if feed_lags else None,
            "chunk_deadline_miss_count": misses,
            "chunk_deadline_miss_percent": 100 * misses / feed_chunks if feed_chunks else None,
            "delta_pp_vs_whisper_0.5s": (all_totals["CER_percent"] - whisper["CER_percent"]
                                         if method in STREAM_METHODS and len(successful) == len(cases)
                                         else None),
        }
        if len(successful) == len(cases) and method in STREAM_METHODS:
            summary["paired_bootstrap_95pct_pp_vs_whisper_0.5s"] = paired_ci(
                scores, [baseline[call_id] for call_id in selected])
        output["apple"][method] = summary
        private[method] = per_call
    destination = args.output_dir or HERE / "private" / args.bundle / "scored"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "SUMMARY.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    (destination / "PER_CALL_PRIVATE.json").write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n")
    lines = ["# iPad Apple音声認識・4方式再評価", "",
             f"集合：`{manifest['dataset_id']}`、{len(cases)}通話。Gold v3.2 masked micro CER。", "",
             "| 方式 | 成功 | 欠損を空文とした便宜値 | 成功件のみCER |",
             "|---|---:|---:|---:|",
             f"| Whisper既存0.5秒入力 | {len(cases)}/{len(cases)} | {whisper['CER_percent']:.3f}% | {whisper['CER_percent']:.3f}% |"]
    for method, stats in output["apple"].items():
        success = f"{stats['success_only_CER_percent']:.3f}%" if stats["success_only_CER_percent"] is not None else "—"
        lines.append(f"| {method} | {stats['success_calls']}/{len(cases)} | {stats['complete_set_CER_percent_missing_as_empty']:.3f}% | {success} |")
    lines.extend(["", "成功が0件の方式にはCERがない。欠損を空文とした便宜値は認識精度ではない。成功件のみのCERも全通話完了前は暫定値。",
                  "0.5秒方式は実時間ペーシングで投入。Whisper既存方式は0.5秒塊を順次渡すが実時間待機はない。",
                  "SFSpeechは45～54秒区間を個別認識して連結し、区間ごとの処理済み音声長を検証。SpeechTranscriberはファイル入力または全通話の連続ストリーム。",
                  "iPad入力は16 kHz PCM16、Whisper既存値は元の8 kHz μ-law音声のため再符号化差がある。",
                  "CERは通話終了後の組立文。各結果本文・通話別採点は非公開private内。",
                  "0.5秒の投入予定から50 ms超遅れた塊は`chunk_deadline_miss_percent`で確認する。SFSpeechには区間確定待ちが別途入る。", ""])
    (destination / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(destination / "REPORT.md"), "summary": output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
