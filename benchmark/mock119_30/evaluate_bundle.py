#!/usr/bin/env python3
"""Reusable fixed-manifest 0.5-second streaming ASR evaluation for a frozen bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO / 'runtime'), str(ROOT)]
from shared_streaming import SharedStreamingASR  # noqa: E402
from variable_decoder_batching import VariableDecoderBatcher  # noqa: E402
from scoring import masked_cer, normalize_text  # noqa: E402

BLOCK = 8000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--manifest', type=Path, default=ROOT / 'manifest.jsonl')
    ap.add_argument('--output-root', type=Path, default=REPO / 'runs' / 'mock119_30')
    ap.add_argument('--batch-calls', type=int, default=10)
    args = ap.parse_args()
    if not 1 <= args.batch_calls <= 20:
        raise ValueError('batch-calls must be 1..20')
    if not args.label or '/' in args.label or args.label.startswith('.'):
        raise ValueError('Label must be one path component')
    bundle = args.bundle.resolve()
    manifest = args.manifest.resolve()
    output = args.output_root.resolve() / args.label
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise FileExistsError(output / 'result.json')
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 30 or len({row['call_id'] for row in rows}) != len(rows):
        raise ValueError('Expected the 30 fixed, unique mock calls')
    audio = {}
    for row in rows:
        path = Path(row['audio_path'])
        if not path.is_absolute():
            path = manifest.parent / path
        if sha(path) != row['audio_sha256']:
            raise ValueError(f'Audio hash mismatch for {row["call_id"]}')
        pcm, sr = sf.read(path, dtype='float32', always_2d=True)
        pcm = pcm.mean(axis=1)
        if sr != 16000:
            pcm = librosa.resample(pcm, orig_sr=sr, target_sr=16000, res_type='soxr_hq')
        audio[row['call_id']] = np.ascontiguousarray(np.clip(pcm, -1, 1), dtype=np.float32)
        if normalize_text(row['gold_text']) != row['gold_normalized']:
            raise ValueError(f'Gold normalization mismatch for {row["call_id"]}')
    t0 = time.perf_counter()
    engine = SharedStreamingASR(bundle, max_sessions=20, chunk_sec=0.5, beam=3,
                                frame_threshold=75, strict_chunk_cadence=True,
                                cross_sdpa_alignment=True, fp16_weights=True)
    torch.cuda.empty_cache()
    batcher = VariableDecoderBatcher(engine, 20, 64)
    records = {row['call_id']: {'parts': [], 'events': []} for row in rows}
    started = time.perf_counter()
    ids = list(audio)
    for group_start in range(0, len(ids), args.batch_calls):
        group = {cid: audio[cid] for cid in ids[group_start:group_start + args.batch_calls]}
        for call_id in group:
            engine.open(call_id)
        for offset in range(0, max(map(len, group.values())), BLOCK):
            chunks = {cid: pcm[offset:min(len(pcm), offset + BLOCK)]
                      for cid, pcm in group.items() if offset < len(pcm)}
            if chunks:
                results = batcher.push_many(chunks)
                torch.cuda.synchronize()
                for cid, result in results.items():
                    records[cid]['parts'].append(result.get('text', ''))
                    records[cid]['events'].append({'audio_end_sec': min(len(audio[cid]), offset + BLOCK) / 16000,
                                                   'result': result, 'final': False})
            finishing = [cid for cid, pcm in group.items() if offset < len(pcm) <= offset + BLOCK]
            if finishing:
                finished = batcher.finish_many(finishing)
                torch.cuda.synchronize()
                for cid in finishing:
                    records[cid]['parts'].append(finished[cid].get('text', ''))
                    records[cid]['events'].append({'audio_end_sec': len(audio[cid]) / 16000,
                                                   'result': finished[cid], 'final': True})
                    print('completed', cid, flush=True)
        assert not engine.sessions
        print('group', group_start // args.batch_calls + 1, 'complete', flush=True)
    inference_wall = time.perf_counter() - started
    scored = []
    for row in rows:
        cid = row['call_id']
        raw = ''.join(records[cid]['parts'])
        norm = normalize_text(raw)
        score = masked_cer([{'type': 'known', 'text': row['gold_normalized']}], norm, 0)
        scored.append({'call_id': cid, 'scenario': row['scenario'], 'duration_sec': len(audio[cid]) / 16000,
                       'noise_profile': row['noise_profile'], 'reference': row['gold_text'],
                       'reference_normalized': row['gold_normalized'],
                       'hypothesis': raw, 'hypothesis_normalized': norm, 'score': score})
        (output / f'{cid}.events.json').write_text(json.dumps(records[cid]['events'], ensure_ascii=False,
                                                      indent=2) + '\n')
    totals = {key: sum(r['score'][key] for r in scored) for key in ('S', 'D', 'I', 'N', 'errors')}
    result = {
        'label': args.label, 'bundle': bundle.name, 'bundle_sha256': sha(bundle / 'BUNDLE.json'),
        'model_sha256': sha(bundle / 'model/whisper.pt'),
        'manifest': str(manifest.relative_to(REPO) if manifest.is_relative_to(REPO) else manifest),
        'manifest_sha256': sha(manifest),
        'settings': {'chunk_sec': 0.5, 'batch_calls': args.batch_calls,
                     'beam': 3, 'frame_threshold': 75,
                     'strict_chunk_cadence': True, 'cross_sdpa_alignment': True,
                     'decoder_microbatch': 20, 'decoder_variable_offsets': True,
                     'decoder_offset_bucket': 64, 'decoder_batch_finish': True,
                     'fp16_weights': True,
                     'normalizer': 'benchmark/mock119_30/scoring.py',
                     'scorer': 'benchmark/mock119_30/scoring.py'},
        'calls': len(rows), 'audio_hours': sum(r['duration_sec'] for r in scored) / 3600,
        'load_wall_sec': started - t0, 'inference_wall_sec': inference_wall,
        'total': {**totals, 'micro_cer': totals['errors'] / totals['N'],
                  'macro_cer': statistics.fmean(r['score']['cer'] for r in scored)},
        'per_call': scored,
    }
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'label': args.label, 'calls': len(rows), 'total': result['total']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
