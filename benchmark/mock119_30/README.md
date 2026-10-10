# 模擬119通報30件の固定CERベンチマーク

`scripts.json`の創作台本30件を[Gemini 3.8 Flash TTS](https://ai.google.dev/gemini-api/docs/speech-generation)の複数話者モードで音声化した。消防側はKore、通報者側はPuck。**実際の119通報音声やその文字起こしは一切含まない。** 元音声と電話風の雑音付き音声は [`AUDIO_INDEX.md`](AUDIO_INDEX.md) から試聴できる。

電話風音声は[119-call-noise-generator](https://github.com/Quick-119/119-call-noise-generator)の実通報に関する集計・候補表（参照コミット`06020f6dbd10d32c117a04915255e69ccf629284`）に基づく独立実装で作った。電話帯域、圧縮、合成したピンク雑音・ブラウン雑音、場合によって50 Hzのハム音、G.711 μ-law形式の8 kHz・2チャンネルWAVを使う。設定SNRは軽度（`light`）25 dB、中程度（`median`）21 dB、騒音多め（`busy`）16 dB。雑音の乱数シードと音響条件は各音声のJSONに記録した。

## 固定された評価入力と結果

- `manifest.jsonl`: 30音声の**相対パス**、SHA-256、台本全文、正規化した正解文。計586.0秒・正解3,075文字。
- `audio/gemini_clean/`: Gemini生成直後のWAV。
- `audio/telephone/`: 3モデルのCER評価に使用した雑音付きWAV。
- `transcripts/`: 各台本の読みやすい表示。すべて`scripts.json`の創作文から生成したもの。
- `results/`: 通話学習なし・AI教師段階・人手追加後の3モデルの文字出力、通話別CER、モデル間比較のJSONとCSV。詳細は[`FINAL_REPORT.md`](FINAL_REPORT.md)。

正解はTTS入力文を連結したもの。TTS音声が逐語的に完全一致しているかは人手聴取で監査していない。正規化とS/D/Iの採点は元研究のhuman Gold v3.2と同じであることを、3モデルの合計90出力で照合した。NFKC、カタカナ→ひらがな、空白・句読点除去後にCERを計算する。漢数字と数字、漢字の異表記は誤りとして数える。

| モデル | この研究の通話データによる追加学習 | 30件の合算CER |
|---|---|---:|
| `whisper_large_v3_zero_shot` | なし | 4.68% |
| `pseudo_quality_full` | 自動文字起こし12,232通話 | 6.02% |
| `quality_full_then_human_replay` | 上記に加えて人手文字起こし1,186通話 | 4.88% |

通話学習なしのモデルはWhisper自体の事前学習済み重みです。この30件は創作音声であり、実通報の性能を表すものではありません。

## 別モデルを同じ30件で評価する

GitHubにチェックポイントはない。先に[完全なfrozen bundleの入手・配置方法](../../docs/model_access.md)に従い、`local_bundles/<名前>/`に置く。音声と正解文はGitHubから取得できるので、Gemini APIキーや実通報データは評価に不要。通話学習なしの比較モデルは[`whisper_large_v3_zero_shot`](../../models/whisper_large_v3_zero_shot/README.md)で説明している。

```bash
.venv/bin/python benchmark/mock119_30/evaluate_bundle.py \
  --bundle local_bundles/quality_full_then_human_replay \
  --label my_model
```

結果はGitから除外した`runs/mock119_30/my_model/result.json`と各通話の0.5秒イベントに保存される。既存ラベルは上書きしない。比較するには:

```bash
.venv/bin/python benchmark/mock119_30/compare_results.py \
  --baseline benchmark/mock119_30/results/pseudo_quality_full.json \
  --candidate runs/mock119_30/my_model/result.json
```

比較スクリプトはマニフェストと推論条件の一致を検証する。評価は10通話ずつ3組で、厳密な0.5秒刻みの入力、beam 3、frame threshold 75、FP16重み、cross SDPA alignment、decoder microbatch 20を使用する。**0.5秒は入力更新間隔で、実時間の表示遅延保証ではない。**

## 再生成について

`generate_tts.py`を再実行する場合だけGemini APIキーが必要。キーは環境変数`GEMINI_API_KEY`または`--api-key-file`で渡し、リポジトリへ保存しない。既存音声はリクエストと音声のSHA-256が一致する場合だけ再利用する。`prepare_benchmark.py`は雑音を固定した乱数シードで再生成し、マニフェストを検証・更新する。TTSを新規生成すると波形が変わる可能性があるため、公開済み30件と別のベンチマークとして保存・評価すること。

この30件は短い創作通報で、実通報の分布を代表しない。実通報でのCERは[別の集計評価](../../study/pseudo_vs_human/README.md)を参照。
