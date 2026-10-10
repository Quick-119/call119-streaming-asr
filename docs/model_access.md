# 学習済みモデル一式の入手と実行

このリポジトリには設定、推論コード、模擬通報ベンチマークを収録しています。**GitHub上のファイルだけではモデルを実行できません。** プロジェクトの管理者から、次のいずれかの完全なfrozen bundleをアクセス管理された別経路で入手してください。

| bundleのディレクトリ名 | 用途 | 必要な`model/whisper.pt`のSHA-256 |
|---|---|---|
| `quality_full_then_human_replay` | 人手教師追加後の採用モデル | `8a8a65377c50c339aba765756b174e0069441c0649f6207f100f2f59aec0a466` |
| `pseudo_quality_full` | 自動文字起こし教師段階の比較モデル | `6a6a1e65351d05384ed849dbd50258c95dfe10f2896598d6ab99e1359b03f685` |

完全なbundleには、少なくとも`BUNDLE.json`、`STREAMING_CONFIG.json`、`model/whisper.pt`、`vendor/SimulStreaming/`、`vendor/silero_vad.jit`、`scripts/streaming_alignment_heads.py`と、`BUNDLE.json`に列挙されたその他のファイルが必要です。VADファイルのSHA-256は`e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`です。読み込み前にbundle自身のマニフェストでファイルの整合性を確認します。**チェックポイント単体では、このストリーミング推論を同じ条件で実行できません。**

入手したbundleを`local_bundles/<bundle名>/`に配置してください。`local_bundles/`はGitの追跡対象から除外しています。bundleや実通報の入力音声をコミットしないでください。

## 動作環境

実験時の環境はLinux、Python 3.14.4、PyTorch 2.13.0+cu130、CUDA対応GPU、および[`requirements-lock.txt`](../models/quality_full_then_human_replay/requirements-lock.txt)に記録したパッケージです。複数通話のバッチ処理には追加で`greenlet==3.5.4`が必要です。GPUドライバーに対応するCUDA/PyTorchビルドを使用してください。元の実験機では読み込まれたNVIDIAカーネルドライバーとユーザー空間ライブラリの版が異なったため、ローカルでライブラリを差し替えて実行しました。この対処はその実験機固有であり、モデルの一般的な要件ではありません。

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r models/quality_full_then_human_replay/requirements-lock.txt greenlet==3.5.4
.venv/bin/python examples/transcribe_file.py \
  --bundle local_bundles/quality_full_then_human_replay \
  --audio benchmark/mock119_30/audio/telephone/mock_01_chest_pain.wav \
  --output runs/example_01
```

実際に配置したbundle内の`STREAMING_CONFIG.json`を設定の基準としてください。[`models/`](../models/)内のコピーは公開した評価結果の設定を確認するためのものです。bundleを構成するファイルを移動・削除すると、bundle自身のハッシュ検証に失敗する可能性があります。

## 推論時の処理

音声を読み込み、複数チャンネルの場合は平均してモノラルに変換し、16 kHzにリサンプリングしてから、連続する8,000サンプルずつストリームへ入力します。日本語の文字起こし、beam 3、frame threshold 75、厳密な0.5秒刻みの入力、cross-attention SDPA alignment、FP16重みを使用します。各時点の`result["text"]`は新しく確定した文字列なので、最後のflushを含めて各文字列を一度ずつ連結してください。通話ごとの推論状態は独立して保持します。

模擬通報30件のベンチマークは同じモデル・設定を使い、10通話ずつ3組に分けてdecoder microbatchingを行います。音声ファイルの再生による評価は、実運用の電話システムにおける表示遅延を測るものではありません。
