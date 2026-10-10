# `pseudo_quality_full`

AI-teacher-stage comparison checkpoint: Whisper large-v3 native FP16 weights, 0.5-second SimulStreaming with AlignAtt and VAC. This stage used 12,232 AI-transcribed calls (100,801 windows, 498.23 hours) from a 14,514-call source corpus. Its initial `c37` adapter had earlier human-data exposure.

The large `model/whisper.pt` is deliberately absent. Required SHA-256: `6a6a1e65351d05384ed849dbd50258c95dfe10f2896598d6ab99e1359b03f685`. Obtain the **complete frozen bundle** separately; see the [model access guide](../../docs/model_access.md). The configuration here is a public copy for inspection, not a runnable bundle by itself.
