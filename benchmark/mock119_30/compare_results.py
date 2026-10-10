#!/usr/bin/env python3
"""Compare two evaluations against the exact same frozen mock-call manifest."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=REPO / 'runs' / 'mock119_30' / 'comparison.json')
    args = ap.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    if baseline['manifest_sha256'] != candidate['manifest_sha256']:
        raise ValueError('The evaluation manifests differ')
    if baseline['settings'] != candidate['settings']:
        raise ValueError('The inference settings differ')
    a = {row['call_id']: row for row in baseline['per_call']}
    b = {row['call_id']: row for row in candidate['per_call']}
    if set(a) != set(b):
        raise ValueError('The evaluated call IDs differ')
    calls = []
    for cid in a:
        old = a[cid]
        new = b[cid]
        calls.append({'call_id': cid, 'scenario': old['scenario'], 'noise_profile': old['noise_profile'],
                      'baseline_cer': old['score']['cer'], 'candidate_cer': new['score']['cer'],
                      'cer_change_pp': 100 * (new['score']['cer'] - old['score']['cer']),
                      'reference_chars': old['score']['N']})
    old_cer = baseline['total']['micro_cer']
    new_cer = candidate['total']['micro_cer']
    rng = random.Random(119)
    paired = [(a[cid]['score']['errors'], b[cid]['score']['errors'], a[cid]['score']['N'])
              for cid in a]
    changes = []
    for _ in range(10000):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        changes.append(100 * sum(row[1] - row[0] for row in sample) / sum(row[2] for row in sample))
    changes.sort()
    result = {'baseline': baseline['label'], 'candidate': candidate['label'],
              'manifest_sha256': baseline['manifest_sha256'], 'calls': len(calls),
              'baseline_micro_cer': old_cer, 'candidate_micro_cer': new_cer,
              'candidate_minus_baseline_pp': 100 * (new_cer - old_cer),
              'relative_error_reduction': (old_cer - new_cer) / old_cer if old_cer else None,
              'paired_bootstrap_95pp': [changes[250], changes[9750]],
              'improved_calls': sum(row['cer_change_pp'] < 0 for row in calls),
              'tied_calls': sum(row['cer_change_pp'] == 0 for row in calls),
              'worse_calls': sum(row['cer_change_pp'] > 0 for row in calls),
              'per_call': calls}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calls[0]))
        writer.writeheader()
        writer.writerows(calls)
    print(json.dumps({k: result[k] for k in result if k != 'per_call'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
