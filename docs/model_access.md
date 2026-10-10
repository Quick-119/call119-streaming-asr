# Getting and running a frozen model bundle

This repository provides configuration, integration code, and a synthetic benchmark. **It cannot run a model with GitHub files alone.** Obtain one of these complete frozen bundles from the project maintainers through a separate, access-controlled channel:

| Bundle directory | Purpose | Required `model/whisper.pt` SHA-256 |
|---|---|---|
| `quality_full_then_human_replay` | Selected human-follow-up model | `8a8a65377c50c339aba765756b174e0069441c0649f6207f100f2f59aec0a466` |
| `pseudo_quality_full` | AI-teacher-stage comparison model | `6a6a1e65351d05384ed849dbd50258c95dfe10f2896598d6ab99e1359b03f685` |

Each complete bundle must contain at least `BUNDLE.json`, `STREAMING_CONFIG.json`, `model/whisper.pt`, `vendor/SimulStreaming/`, `vendor/silero_vad.jit`, `scripts/streaming_alignment_heads.py`, and the other files listed in `BUNDLE.json`. The VAD file has SHA-256 `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`. The bundle's own manifest verifies all files before loading. **A bare checkpoint is insufficient** for this exact streaming pipeline.

Put the extracted bundle at `local_bundles/<bundle name>/`. `local_bundles/` is ignored by Git. Do not commit the bundle or input audio from real emergency calls.

## Environment

The recorded environment was Linux, Python 3.14.4, PyTorch 2.13.0+cu130, a CUDA-compatible GPU, and the package versions in [`requirements-lock.txt`](../models/quality_full_then_human_replay/requirements-lock.txt). The multi-call batching helper additionally needs `greenlet==3.5.4`. Use a CUDA/PyTorch build compatible with your driver. The original workstation required a local NVIDIA library workaround because its loaded kernel driver and user-space libraries differed; that workaround is machine-specific and is not a model requirement.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r models/quality_full_then_human_replay/requirements-lock.txt greenlet==3.5.4
.venv/bin/python examples/transcribe_file.py \
  --bundle local_bundles/quality_full_then_human_replay \
  --audio benchmark/mock119_30/audio/telephone/mock_01_chest_pain.wav \
  --output runs/example_01
```

The installed bundle's `STREAMING_CONFIG.json` is authoritative. The copies in [`models/`](../models/) document the exact configuration used for the published results. Keep the bundle directory intact so its own hash checks pass.

## Runtime behavior

Audio is decoded, averaged to mono, resampled to 16 kHz, and sent to the stream in contiguous 8,000-sample blocks. The runner uses Japanese transcription, beam 3, frame threshold 75, strict 0.5-second chunk cadence, cross-attention SDPA alignment, and FP16 resident weights. A call's successive `result["text"]` values are newly committed text; concatenate each once, including the final flush. The runtime maintains separate state per call.

The 30-call benchmark uses the same model and settings in groups of ten calls with decoder microbatching. File replay does not measure the wall-clock latency of a production telephone service.
