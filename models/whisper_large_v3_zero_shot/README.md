# `whisper_large_v3_zero_shot`

この研究の119番通報データで追加学習していない、[OpenAI Whisper large-v3](https://huggingface.co/openai/whisper-large-v3)の事前学習済み重みです。「未学習」は**この研究の通話データによる追加学習が0件**という意味で、Whisperの一般的な事前学習をしていないという意味ではありません。元のHugging Face形式の`model.safetensors`のSHA-256は`a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`です。

既存の2モデルと同じWhisper large-v3構造、物理FP16重み、SimulStreamingの0.5秒入力・AlignAtt・VAC・beam 3・frame threshold 75で評価しました。比較用に変換したnative `model/whisper.pt`のSHA-256は`68c16d3d1a31760d28ffbba2bff68267cb3875cfe5a6d82306bc413bce0cd13b`です。`STREAMING_CONFIG.json`は、実験時の有効な推論設定を既存モデルと同一にしたコピーです。そこに残る`run_id`は共通設定の識別子で、重みの学習来歴を示しません。AlignAttのヘッド指定を含む推論設定は既存モデルに合わせており、この未追加学習モデル専用に最適化した設定ではありません。

チェックポイントと完全なfrozen bundleはこのリポジトリには含めていません。ベンチマークと同じ実行経路で利用するには、[モデル利用ガイド](../../docs/model_access.md)に従って、チェックポイントを含む完全なbundleを用意してください。
