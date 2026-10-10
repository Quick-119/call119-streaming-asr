# `quality_full_then_human_replay`

Selected 2026-10-08 model: Whisper large-v3 native FP16 weights, 0.5-second SimulStreaming with AlignAtt and VAC. The AI-teacher stage used 12,232 calls and the follow-up human-teacher stage used 1,186 calls (6,258 windows, 45.92 hours). This checkpoint is the human-follow-up candidate used in the published comparison.

The large `model/whisper.pt` is deliberately absent. Required SHA-256: `8a8a65377c50c339aba765756b174e0069441c0649f6207f100f2f59aec0a466`. Obtain the **complete frozen bundle** separately; see the [model access guide](../../docs/model_access.md). The configuration here is a public copy for inspection, not a runnable bundle by itself.
