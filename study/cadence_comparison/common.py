"""実通報の評価データを非公開のまま読み込み、既存のGold v3.2で採点する。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "runs" / "cadence_comparison"
SPLITS = ("dev_old", "dev_new", "test_old", "test_new", "test_fresh", "test_fresh2")
HELD_OUT = SPLITS[2:]

# 設定とその参照先はGit管理対象外に置く。実通報の情報をコードに埋め込まない。
CONFIG_PATH = Path(os.environ.get("CALL119_CADENCE_CONFIG", REPO / "private" / "cadence_config.json"))
if not CONFIG_PATH.is_file():
    raise SystemExit(f"非公開の評価設定がありません: {CONFIG_PATH}")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
MODEL_ROOTS = {name: Path(path) for name, path in CONFIG["model_roots"].items()}
MANIFEST_ROOT = Path(CONFIG["manifest_root"])
SCORER_DIR = Path(CONFIG["gold_v32_scorer_dir"])
sys.path.insert(0, str(SCORER_DIR))
from evaluate_human_gold_v32 import masked_cer, normalize_text  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_for(split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(split)
    return [json.loads(line) for line in (MANIFEST_ROOT / f"{split}_calls.jsonl").read_text().splitlines() if line]


def bundle_for(label: str) -> Path:
    return MODEL_ROOTS[label] / "bundles" / label


def baseline_for(label: str, split: str) -> dict:
    return json.loads((MODEL_ROOTS[label] / "evaluation" / label / split / "COMBINED.json").read_text())


def verify_inputs(label: str, split: str, rows: list[dict]) -> dict:
    base = baseline_for(label, split)
    if sha(MANIFEST_ROOT / f"{split}_calls.jsonl") != base["manifest_sha256"]:
        raise ValueError(f"入力マニフェストが0.5秒評価と異なります: {label}/{split}")
    known = {item["call_id"]: item["primary"]["N"] for item in base["per_call_private_aggregate"]}
    if len(rows) != base["calls"] or {row["call_id"]: row["known_characters"] for row in rows} != known:
        raise ValueError(f"通話IDまたは正解文字数が異なります: {label}/{split}")
    return base


def score_call(row: dict, hypothesis: str) -> dict:
    gold = json.loads(Path(row["gold_scoring_path"]).read_text(encoding="utf-8"))
    if (gold["audio_sha256"] != row["audio_sha256"] or
            gold["source_sha256"] != row["source_sha256"] or
            gold["known_characters"] != row["known_characters"]):
        raise ValueError(f"正解文と入力が一致しません: {row['call_id']}")
    return masked_cer(gold["segments"], normalize_text(hypothesis),
                      int(gold["unknown_policy"]["primary_cap_chars"]))


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def sum_scores(items: list[dict]) -> dict:
    keys = ("S", "D", "I", "N", "errors")
    total = {key: sum(item["primary"][key] for item in items) for key in keys}
    total["micro_cer"] = total["errors"] / total["N"]
    return total
