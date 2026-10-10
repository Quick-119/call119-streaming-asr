#!/usr/bin/env python3
"""Fail closed when a tracked file is outside the public release allowlist."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED = {".gitignore", "README.md", "scripts/audit_public_tree.py"}
TEXT_SUFFIXES = {".md", ".py", ".json", ".jsonl", ".csv", ".txt"}
BLOCKED_TEXT = (
    re.compile(b"/home/" + b"okazaki/"),
    re.compile(b"call119_asr_" + b"dataset"),
    re.compile(rb"AQ\.[A-Za-z0-9_-]{25,}"),
    re.compile(rb"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}"),
)


def permitted(path: str) -> bool:
    p = Path(path)
    if path in FIXED:
        return True
    if p.parts[:1] == ("docs",) and len(p.parts) == 2 and p.suffix == ".md":
        return True
    if p.parts[:1] == ("runtime",) and len(p.parts) == 2 and p.suffix == ".py":
        return True
    if p.parts[:1] == ("examples",) and len(p.parts) == 2 and p.suffix == ".py":
        return True
    if p.parts[:1] == ("models",) and len(p.parts) == 3 and p.parts[1] in {
        "quality_full_then_human_replay", "pseudo_quality_full"
    } and p.name in {"README.md", "STREAMING_CONFIG.json", "requirements-lock.txt"}:
        return True
    if p.parts[:2] == ("study", "pseudo_vs_human") and len(p.parts) == 3 and p.name in {
        "README.md", "summary.json"
    }:
        return True
    if p.parts[:2] == ("benchmark", "mock119_30"):
        if len(p.parts) == 3 and p.name in {
            "README.md", "FINAL_REPORT.md", "AUDIO_INDEX.md", "scripts.json", "manifest.jsonl",
            "generate_tts.py", "prepare_benchmark.py", "evaluate_bundle.py",
            "compare_results.py", "scoring.py",
        }:
            return True
        if len(p.parts) == 4 and p.parts[2] == "results" and p.name in {
            "pseudo_quality_full.json", "quality_full_then_human_replay.json",
            "comparison.json", "comparison.csv"
        }:
            return True
        if (len(p.parts) == 5 and p.parts[2] == "audio"
                and p.parts[3] in {"gemini_clean", "telephone"}
                and p.name.startswith("mock_") and p.suffix in {".wav", ".json"}):
            return True
        if (len(p.parts) == 4 and p.parts[2] == "transcripts"
                and p.name.startswith("mock_") and p.suffix == ".txt"):
            return True
    return False


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw = subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=ROOT)
    paths = [p.decode() for p in raw.split(b"\0") if p]
    problems = []
    for name in paths:
        path = ROOT / name
        if not permitted(name):
            problems.append(f"Not allowlisted: {name}")
            continue
        if not path.is_file():
            problems.append(f"Missing regular file: {name}")
            continue
        if path.stat().st_size > 2_000_000:
            problems.append(f"File exceeds 2 MB cap: {name}")
        if path.suffix in TEXT_SUFFIXES:
            data = path.read_bytes()
            for pattern in BLOCKED_TEXT:
                if pattern.search(data):
                    problems.append(f"Forbidden content in: {name}")
                    break
    manifest = ROOT / "benchmark/mock119_30/manifest.jsonl"
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        if len(rows) != 30 or len({r["call_id"] for r in rows}) != 30:
            problems.append("Benchmark manifest must contain 30 unique synthetic calls")
        expected = set()
        expected_sidecars = set()
        for row in rows:
            for prefix, hash_key in (("audio_path", "audio_sha256"),
                                     ("clean_audio_path", "clean_audio_sha256")):
                relative = Path(row[prefix])
                if relative.is_absolute() or ".." in relative.parts or not relative.name.startswith("mock_"):
                    problems.append(f"Unsafe benchmark audio path: {relative}")
                    continue
                target = manifest.parent / relative
                expected.add(target.relative_to(ROOT).as_posix())
                if not target.is_file() or sha(target) != row[hash_key]:
                    problems.append(f"Audio hash mismatch: {relative}")
                sidecar = target.with_suffix(".json")
                expected_sidecars.add(sidecar.relative_to(ROOT).as_posix())
                if not sidecar.is_file():
                    problems.append(f"Missing synthetic audio metadata: {sidecar}")
                else:
                    metadata = json.loads(sidecar.read_text())
                    if metadata.get("audio_sha256") != row[hash_key]:
                        problems.append(f"Audio metadata hash mismatch: {sidecar}")
        tracked_audio = {p for p in paths if p.endswith(".wav")}
        if tracked_audio != expected:
            problems.append("Tracked WAV files differ from synthetic manifest")
        tracked_sidecars = {p for p in paths if p.startswith("benchmark/mock119_30/audio/")
                            and p.endswith(".json")}
        if tracked_sidecars != expected_sidecars:
            problems.append("Tracked audio metadata differ from synthetic manifest")
        scripts = json.loads((manifest.parent / "scripts.json").read_text())
        if len(scripts) != 30 or [s["id"] for s in scripts] != [r["call_id"] for r in rows]:
            problems.append("Synthetic scripts differ from benchmark manifest")
        expected_transcripts = set()
        for script in scripts:
            name = f"benchmark/mock119_30/transcripts/{script['id']}.txt"
            expected_transcripts.add(name)
            lines = [("指令員" if turn["speaker"] == "Dispatcher" else "通報者") + "：" + turn["text"]
                     for turn in script["turns"]]
            transcript = "\n".join(lines) + "\n"
            if not (ROOT / name).is_file() or (ROOT / name).read_text() != transcript:
                problems.append(f"Synthetic transcript mismatch: {name}")
        tracked_transcripts = {p for p in paths if p.startswith("benchmark/mock119_30/transcripts/")}
        if tracked_transcripts != expected_transcripts:
            problems.append("Tracked transcripts differ from synthetic scripts")
    elif any(p.endswith(".wav") for p in paths):
        problems.append("WAV files require a synthetic manifest")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Public-tree audit passed: {len(paths)} tracked files")


if __name__ == "__main__":
    main()
