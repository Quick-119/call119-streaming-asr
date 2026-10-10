# AI-teacher stage and human-follow-up stage

This is the **aggregate-only** release of the comparison between `pseudo_quality_full` and `quality_full_then_human_replay`. No real-call audio, reference text, hypotheses, call identifiers, or call-level scores are included. Machine-readable split totals are in [`summary.json`](summary.json).

## Training stages

| Stage | Calls | Training windows | Audio hours |
|---|---:|---:|---:|
| AI teacher, one epoch | 12,232 | 100,801 | 498.23 |
| Human follow-up, one epoch | 1,186 | 6,258 | 45.92 |

The source corpus had 14,514 calls. Quality filtering reduced the AI-teacher stage to 12,232 calls. The follow-up consisted of 797 newly annotated calls and 389 replayed older human-annotated calls. Both stages started from an earlier `c37` adapter with prior human-data exposure. Therefore this comparison estimates the increment from the **final human follow-up stage**. It does not compare a strictly never-human-trained model with a human-trained model, nor isolate the new annotations from the older replayed annotations.

## CER

Both frozen FP16 models used the same SimulStreaming configuration: 0.5-second strict input chunks, beam 3, frame threshold 75, AlignAtt and VAC. CER is pooled character errors divided by pooled known reference characters after NFKC, katakana-to-hiragana conversion, and removal of whitespace and punctuation. Numeral or kanji spelling variants remain errors.

| Evaluation group | Calls | AI stage | Human follow-up | Change |
|---|---:|---:|---:|---:|
| Development | 48 | 22.95% | 16.97% | −5.98 points |
| Held-out evaluation | 126 | 28.87% | 20.27% | **−8.60 points** |
| All reported calls | 174 | 27.18% | 19.33% | −7.85 points |

The paired-call bootstrap 95% interval for the held-out difference was **−12.09 to −5.82 points** (5,000 resamples). These evaluation calls were excluded from this training, but some held-out sets had been viewed in earlier research; they are not a completely untouched blind test. The 0.5 seconds measures input update cadence, not end-to-end production latency.

`summary.json` preserves the six split-level aggregate counts and model SHA-256 fingerprints while omitting all call-level data. The synthetic benchmark in [`benchmark/mock119_30`](../../benchmark/mock119_30/) is a separate evaluation distribution and should not be compared as if it represented real-call CER.
