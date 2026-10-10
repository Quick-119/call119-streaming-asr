# 119-call streaming ASR

Japanese emergency-call ASR research with a Whisper large-v3 model and 0.5-second SimulStreaming input updates. This repository contains the runtime integration, model configuration, aggregate study results, and a **fully synthetic** 30-call benchmark. It contains no real 119-call recordings or transcripts and no model checkpoints.

## What is here

| Directory | Contents |
|---|---|
| [`models/`](models/) | Configuration and exact model SHA-256 for the AI-teacher stage and the human-follow-up stage |
| [`runtime/`](runtime/) | The shared multi-call inference runner and batching helpers used by the comparisons |
| [`examples/`](examples/) | Single-file inference example using an externally supplied frozen bundle |
| [`study/pseudo_vs_human/`](study/pseudo_vs_human/) | Aggregate comparison on real-call evaluation sets; no call-level data |
| [`benchmark/mock119_30/`](benchmark/mock119_30/) | Gemini TTS scripts, synthetic audio, frozen references, and 30-call CER results |

The selected 2026-10-08 model is `quality_full_then_human_replay`: a stage trained on 12,232 AI-transcribed calls followed by 1,186 human-transcribed calls. The AI-stage checkpoint is `pseudo_quality_full`. The source corpus had 14,514 calls; 14,514 is **not** the number used in this AI stage. Both stages started from an earlier `c37` adapter that already had human-data exposure, so this comparison measures the **additional human stage**, not a never-human-trained model.

## To use the model

You must obtain the **complete frozen bundle** separately from the project maintainers. Its `model/whisper.pt` is approximately 3.1 GB and is not published here. The bundle also needs `BUNDLE.json`, `vendor/SimulStreaming/`, `vendor/silero_vad.jit`, and its `scripts/` files. The two bundles have different model hashes; see the [model access guide](docs/model_access.md) for the complete list and setup commands. A base Whisper download or LoRA adapter is unnecessary when the complete frozen bundle is available.

After placing the bundle in the ignored `local_bundles/` directory, run the single-file example on a WAV file:

```bash
python examples/transcribe_file.py \
  --bundle local_bundles/quality_full_then_human_replay \
  --audio benchmark/mock119_30/audio/telephone/mock_01_chest_pain.wav \
  --output runs/example_01
```

The input is fed in 8,000-sample chunks at 16 kHz. The **0.5 seconds refers to input update cadence**, not a guarantee of end-to-end display latency. The [benchmark guide](benchmark/mock119_30/README.md) shows how to evaluate another frozen bundle on the same 30 synthetic calls. Running inference or the benchmark does **not** require the real-call dataset.

## Reported CER

| Evaluation | AI-teacher stage | Human-follow-up stage | Improvement |
|---|---:|---:|---:|
| 126 held-out real calls | 28.87% | 20.27% | 8.60 percentage points |
| 30 synthetic Gemini TTS calls | 6.02% | 4.88% | 1.14 percentage points |

See the [real-call aggregate study](study/pseudo_vs_human/README.md) and [synthetic benchmark report](benchmark/mock119_30/FINAL_REPORT.md) for protocols and limitations. Synthetic performance does not represent real-call performance.

## Privacy and release checks

Only explicitly selected files are included. The [release policy](docs/privacy.md), `.gitignore`, and `scripts/audit_public_tree.py` prevent accidental inclusion of private inputs, transcripts, generated per-call real-call outputs, credentials, and checkpoints. Run `python scripts/audit_public_tree.py` before pushing.
