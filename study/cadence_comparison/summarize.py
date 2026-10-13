"""同じ174通話で0.5秒・1秒・2秒・全通話オフラインのCERを集計する。"""
from __future__ import annotations

import json
from pathlib import Path

from common import HELD_OUT, MODEL_ROOTS, RUNS, SPLITS, baseline_for, rows_for, save_json

MODES = ("stream_0.5s", "stream_1s", "stream_2s", "offline_full_call")
GROUPS = {"development": SPLITS[:2], "held_out": HELD_OUT, "all": SPLITS}


def result_for(label: str, mode: str, split: str) -> dict:
    if mode == "stream_0.5s":
        return baseline_for(label, split)
    return json.loads((RUNS / "results" / mode / label / split / "COMBINED.json").read_text())


def main() -> None:
    matrix = {}
    for label in MODEL_ROOTS:
        by_mode = {}
        baseline_hash = baseline_for(label, SPLITS[0])["model_sha256"]
        for mode in MODES:
            split_data = {}
            for split in SPLITS:
                row = result_for(label, mode, split)
                baseline = baseline_for(label, split)
                expected_ids = [x["call_id"] for x in rows_for(split)]
                actual_ids = [x["call_id"] for x in row["per_call_private_aggregate"]]
                if (actual_ids != expected_ids or row["calls"] != len(expected_ids) or
                        row["manifest_sha256"] != baseline["manifest_sha256"] or
                        row["model_sha256"] != baseline_hash or
                        row["primary"]["N"] != baseline["primary"]["N"]):
                    raise ValueError(f"比較条件が異なります: {label}/{mode}/{split}")
                split_data[split] = {"calls": row["calls"], "N": row["primary"]["N"],
                                     "errors": row["primary"]["errors"],
                                     "cer": row["primary"]["micro_cer"]}
            groups = {}
            for group_name, splits in GROUPS.items():
                calls = sum(split_data[split]["calls"] for split in splits)
                n = sum(split_data[split]["N"] for split in splits)
                errors = sum(split_data[split]["errors"] for split in splits)
                groups[group_name] = {"calls": calls, "N": n, "errors": errors, "cer": errors / n}
            by_mode[mode] = {"splits": split_data, "groups": groups}
        for mode in MODES[1:]:
            for group_name in GROUPS:
                by_mode[mode]["groups"][group_name]["delta_pp_vs_0.5s"] = 100 * (
                    by_mode[mode]["groups"][group_name]["cer"] -
                    by_mode["stream_0.5s"]["groups"][group_name]["cer"])
        matrix[label] = {"model_sha256": baseline_hash, "modes": by_mode}
    output = {"source_kind": "real_119_recording", "metric": "human_gold_v3.2_masked_micro_cer",
              "models": matrix, "privacy": "集計値のみ。音声、文字起こし、通話IDは含まない。"}
    save_json(RUNS / "COMPARISON_AGGREGATE.json", output)
    names = {"whisper_large_v3_zero_shot": "通話学習なし", "pseudo_quality_full": "自動教師段階",
             "quality_full_then_human_replay": "人手教師追加後"}
    mode_names = {"stream_0.5s": "0.5秒入力", "stream_1s": "1秒入力",
                  "stream_2s": "2秒入力", "offline_full_call": "全通話オフライン"}
    lines = ["# 3モデルの入力間隔別CER（実通報）", "",
             "主要値は学習から除外した126通話のGold v3.2 masked micro CER。低いほど良い。", "",
             "| 推論方式 | 通話学習なし | 自動教師段階 | 人手教師追加後 |", "|---|---:|---:|---:|"]
    for mode in MODES:
        values = [matrix[label]["modes"][mode]["groups"]["held_out"]["cer"] for label in MODEL_ROOTS]
        lines.append("| " + mode_names[mode] + " | " + " | ".join(f"{100*v:.3f}%" for v in values) + " |")
    lines += ["", "## 評価条件", "", "- 6分割174通話をすべて評価し、そのうち学習から除外した126通話を主要な比較に使用。",
              "- 音声・正解文・採点基準・チェックポイントは0.5秒評価と同一。推論結果本文は保存せず、通話ID・採点値・文字出力のSHA-256のみを非公開記録に保存。",
              "- 1秒・2秒入力はSimulStreamingの更新間隔のみを変更し、beam 3、frame threshold 75、AlignAtt/VAC、FP16、10通話バッチを維持。",
              "- 全通話オフラインは同じ重みを標準Whisper復号に読み込み、通話全体を一度に渡す。内部では最大30秒の窓に分割。日本語転写、beam 3、標準の温度再試行、前文脈あり、外部VADなし。再試行の乱数は通話IDから固定。",
              "- 同梱のオフライン復号コードはメル数80・言語トークン数99を既定とするため、large-v3の128メル・100言語トークンへ評価時だけ補正。重み・バンドルは変更していない。",
              "- オフライン推論はストリーミングと復号器・VADが異なるため、CER差は入力間隔だけの効果ではない。",
              "- いずれの方式にも追加学習は不要。", ""]
    (RUNS / "RESULTS.md").write_text("\n".join(lines))
    print("wrote", RUNS / "COMPARISON_AGGREGATE.json", "and", RUNS / "RESULTS.md")


if __name__ == "__main__":
    main()
