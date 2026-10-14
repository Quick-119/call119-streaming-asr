# 119番通報向けストリーミング音声認識

Whisper large-v3とSimulStreamingを使った、119番通報向け音声認識の研究成果を公開するリポジトリです。推論時は0.5秒ごとに音声を入力します。モデルの設定・推論コード・比較の集計結果と、**実通報を使わずに作成した模擬通報30件**の評価データを収録しています。実際の119番通報音声やその文字起こし、学習済みチェックポイントは公開していません。

## 収録内容

| ディレクトリ | 内容 |
|---|---|
| [`models/`](models/) | 通話学習なし、自動文字起こし教師段階、人手教師追加後の3モデルの設定とチェックポイントのSHA-256 |
| [`runtime/`](runtime/) | 比較実験で使った複数通話のストリーミング推論コードとバッチ処理補助コード |
| [`examples/`](examples/) | 別途入手した完全なモデル一式を使い、音声ファイル1件を推論する例 |
| [`study/pseudo_vs_human/`](study/pseudo_vs_human/) | 実通報の評価における集計比較。通話ごとのデータは含まない |
| [`benchmark/mock119_30/`](benchmark/mock119_30/) | Gemini TTSによる創作台本・模擬音声・正解文・30件のCER評価結果 |
| [`benchmark/ipad_apple_4way/`](benchmark/ipad_apple_4way/) | iPadでSFSpeechRecognizer・SpeechTranscriberのファイル入力／0.5秒入力を比較するアプリと非公開Gold用採点器 |

2026年10月8日時点で選んだモデルは`quality_full_then_human_replay`です。自動文字起こしした12,232通話で学習した後、人手で文字起こしした1,186通話を使って追加学習しています。自動文字起こし教師段階のモデルは`pseudo_quality_full`、この研究の通話データで追加学習していない比較モデルは`whisper_large_v3_zero_shot`です。元の候補コーパスは14,514通話ですが、自動文字起こし教師段階で使用したのは品質選別後の12,232通話です。後半2モデルの出発点である`c37`アダプターには以前の人手データによる学習が含まれます。したがって、後半2モデルの差は**最後の人手教師追加段階の効果**を示す一方、自動文字起こし教師段階そのものを純粋なAIのみの学習と解釈することはできません。

## モデルを使うために必要なもの

管理者から**完全なfrozen bundle（モデルと実行に必要なファイル一式）**を別途入手してください。約3.1 GBの`model/whisper.pt`はGitHubにありません。`BUNDLE.json`、`vendor/SimulStreaming/`、`vendor/silero_vad.jit`、`scripts/`内の必要なファイルも必要です。モデルごとにチェックポイントのハッシュ値が異なります。入手物の一覧、配置方法、動作環境は[モデル利用ガイド](docs/model_access.md)に記載しています。完全なbundleがあれば、WhisperのベースモデルやLoRAアダプターを別途ダウンロードする必要はありません。

bundleをGit管理対象外の`local_bundles/`に配置した後、次の例でWAVファイルを推論できます。

```bash
python examples/transcribe_file.py \
  --bundle local_bundles/quality_full_then_human_replay \
  --audio benchmark/mock119_30/audio/telephone/mock_01_chest_pain.wav \
  --output runs/example_01
```

16 kHzの音声を8,000サンプルずつ入力します。**0.5秒は音声の入力更新間隔であり、認識結果の表示遅延を保証する値ではありません。** 別のbundleを同じ模擬通報30件で評価する方法は[ベンチマーク利用ガイド](benchmark/mock119_30/README.md)を参照してください。推論と模擬通報の評価に実通報データは不要です。

## CERの集計結果

| 評価データ | 通話学習なし | 自動文字起こし教師段階 | 人手教師追加後 |
|---|---:|---:|---:|
| 学習から除外した実通報126件 | 39.23% | 28.87% | 20.27% |
| Gemini TTSによる模擬通報30件 | 4.68% | 6.02% | 4.88% |

評価方法と限界は[実通報の集計比較](study/pseudo_vs_human/README.md)と[模擬通報の評価報告](benchmark/mock119_30/FINAL_REPORT.md)に記載しています。模擬通報での性能を実通報での性能と同一視することはできません。

## 機密データの扱いと公開前の確認

公開対象として選んだファイルだけを収録しています。[公開方針](docs/privacy.md)を確認してください。`.gitignore`は元データ、実通報の文字起こし、通話ごとの評価出力、チェックポイントなどの混入を防ぐための設定です。`scripts/audit_public_tree.py`は追跡中のファイルを許可リストと照合し、認証情報らしい文字列と模擬音声のハッシュ値も検証します。変更を公開する前に`python scripts/audit_public_tree.py`を実行してください。
