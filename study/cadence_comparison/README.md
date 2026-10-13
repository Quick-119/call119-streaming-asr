# 3モデルの入力間隔別CER（実通報）

主要値は学習から除外した126通話のGold v3.2 masked micro CER。低いほど良い。

| 推論方式 | 通話学習なし | 自動教師段階 | 人手教師追加後 |
|---|---:|---:|---:|
| 0.5秒入力 | 39.233% | 28.869% | 20.268% |
| 1秒入力 | 31.597% | 26.008% | 19.663% |
| 2秒入力 | 29.674% | 24.619% | 20.072% |
| 全通話オフライン | 31.283% | 26.636% | 20.659% |

## 評価条件

- 6分割174通話をすべて評価し、そのうち学習から除外した126通話を主要な比較に使用。
- 音声・正解文・採点基準・チェックポイントは0.5秒評価と同一。推論結果本文は保存せず、通話ID・採点値・文字出力のSHA-256のみを非公開記録に保存。
- 1秒・2秒入力はSimulStreamingの更新間隔のみを変更し、beam 3、frame threshold 75、AlignAtt/VAC、FP16、10通話バッチを維持。
- 全通話オフラインは同じ重みを標準Whisper復号に読み込み、通話全体を一度に渡す。内部では最大30秒の窓に分割。日本語転写、beam 3、標準の温度再試行、前文脈あり、外部VADなし。再試行の乱数は通話IDから固定。
- 同梱のオフライン復号コードはメル数80・言語トークン数99を既定とするため、large-v3の128メル・100言語トークンへ評価時だけ補正。重み・バンドルは変更していない。
- オフライン推論はストリーミングと復号器・VADが異なるため、CER差は入力間隔だけの効果ではない。
- いずれの方式にも追加学習は不要。

## 公開ファイルと非公開ファイル

[`COMPARISON_AGGREGATE.json`](COMPARISON_AGGREGATE.json)にはモデルのチェックポイントSHA-256、分割・評価群ごとの通話数、正解文字数、誤り数、CERだけを収録しました。音声、正解文、推論文、通話ID、通話別スコア、入力マニフェストは含みません。個別の評価出力は`runs/cadence_comparison/`に保存され、Git管理対象外です。

`run_streaming.py`は公開済みの[`runtime/`](../../runtime/)を使用します。`run_offline.py`は各bundleに同梱されたWhisper実装を使用します。`summarize.py`は、全方式・全分割の入力順、通話数、正解文字数、モデルSHA-256が一致することを確認してから集計します。

## 再現に必要な非公開入力

データ管理者の許可を得た評価環境で、次を別途用意してください。これらをリポジトリ内へ追加しないでください。

1. 3モデルの完全なfrozen bundle。配置と依存環境は[モデル利用ガイド](../../docs/model_access.md)を参照。
2. 実通報174件の6分割マニフェスト。各行には`call_id`、`audio_path`、`audio_sha256`、`source_sha256`、`gold_scoring_path`、`known_characters`が必要。
3. Gold v3.2の非公開採点JSONと、`evaluate_human_gold_v32.py`を含む採点コード。句読点・空白の正規化や正解不明区間の扱いを同一にするために使用。
4. 3モデルそれぞれの0.5秒評価`COMBINED.json`と、各分割の`manifest_00.jsonl`以降。再評価の入力とチェックポイントが同一か検証するために使用。

`private/cadence_config.json`を次の形で作成します。パスは手元の絶対パスに置き換えてください。`private/`はGit管理対象外です。

```json
{
  "manifest_root": "/非公開/評価マニフェストのディレクトリ",
  "gold_v32_scorer_dir": "/非公開/採点コードのディレクトリ",
  "model_roots": {
    "whisper_large_v3_zero_shot": "/非公開/通話学習なしの評価ディレクトリ",
    "pseudo_quality_full": "/非公開/自動教師段階の評価ディレクトリ",
    "quality_full_then_human_replay": "/非公開/人手教師追加後の評価ディレクトリ"
  }
}
```

各モデルの評価ディレクトリには`bundles/<モデル名>/`と`evaluation/<モデル名>/<分割名>/`が必要です。設定ファイルを別の場所に置く場合は、環境変数`CALL119_CADENCE_CONFIG`で指定できます。

## 実行

対応するCUDA環境で、GPUメモリに余裕がある状態で1モデルずつ実行します。`python`はWhisper、PyTorch、librosa、soundfile、SimulStreamingの依存パッケージを導入した環境のものを使用してください。

```bash
python study/cadence_comparison/run_streaming.py --model whisper_large_v3_zero_shot --chunk-sec 1
python study/cadence_comparison/run_streaming.py --model pseudo_quality_full --chunk-sec 1
python study/cadence_comparison/run_streaming.py --model quality_full_then_human_replay --chunk-sec 1
python study/cadence_comparison/run_streaming.py --model whisper_large_v3_zero_shot --chunk-sec 2
python study/cadence_comparison/run_streaming.py --model pseudo_quality_full --chunk-sec 2
python study/cadence_comparison/run_streaming.py --model quality_full_then_human_replay --chunk-sec 2
python study/cadence_comparison/run_offline.py --model whisper_large_v3_zero_shot
python study/cadence_comparison/run_offline.py --model pseudo_quality_full
python study/cadence_comparison/run_offline.py --model quality_full_then_human_replay
python study/cadence_comparison/summarize.py
```

途中で中断した場合、既存の通話別結果を照合して再利用します。集計は`runs/cadence_comparison/COMPARISON_AGGREGATE.json`と`RESULTS.md`へ出力されます。公開用の集計ファイルへ転記する際は、[公開前監査](../../scripts/audit_public_tree.py)を実行してください。
