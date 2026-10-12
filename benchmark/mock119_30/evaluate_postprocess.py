#!/usr/bin/env python3
"""保存済みの模擬119通話30件を、推論を繰り返さずに後処理評価する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "runtime"))
from postprocess import correct_transcript  # noqa: E402
from scoring import masked_cer, normalize_text  # noqa: E402


def evaluate(path: Path) -> dict:
    result = json.loads(path.read_text())
    manifest_hash = hashlib.sha256((HERE / "manifest.jsonl").read_bytes()).hexdigest()
    if result.get("manifest_sha256") != manifest_hash:
        raise ValueError("模擬通報の評価マニフェストが異なります")
    calls = result.get("per_call", [])
    manifest = [json.loads(line) for line in (HERE / "manifest.jsonl").read_text().splitlines() if line]
    if (len(calls) != 30
            or [call.get("call_id") for call in calls] != [row["call_id"] for row in manifest]
            or any(call.get("reference_normalized") != row["gold_normalized"]
                   for call, row in zip(calls, manifest))):
        raise ValueError("模擬通報30件の結果を指定してください")
    details = []
    for call in calls:
        raw = call["hypothesis"]
        fixed = correct_transcript(raw)
        segments = [{"type": "known", "text": call["reference_normalized"]}]
        before = masked_cer(segments, normalize_text(raw))
        after = masked_cer(segments, normalize_text(fixed))
        details.append({"call_id": call["call_id"], "changed": fixed != raw,
                        "before_errors": before["errors"], "after_errors": after["errors"],
                        "error_delta": after["errors"] - before["errors"], "N": before["N"]})
    n = sum(row["N"] for row in details)
    before = sum(row["before_errors"] for row in details)
    after = sum(row["after_errors"] for row in details)
    return {"source": path.name, "model": result["label"], "calls": len(details),
            "manifest_sha256": manifest_hash, "postprocess": "runtime/postprocess.py",
            "before_errors": before, "after_errors": after, "N": n,
            "before_cer": before / n, "after_cer": after / n,
            "changed_calls": sum(row["changed"] for row in details),
            "improved_calls": sum(row["error_delta"] < 0 for row in details),
            "worsened_calls": sum(row["error_delta"] > 0 for row in details),
            "per_call": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = [evaluate(path) for path in args.input]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n")
    for row in results:
        print(row["model"], row["before_errors"], "→", row["after_errors"],
              "悪化通話", row["worsened_calls"])


if __name__ == "__main__":
    main()
