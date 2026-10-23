# iPadでのApple音声認識・4方式評価

実通報126通話のGold v3.2評価集合を、iPad上のSFSpeechRecognizerとSpeechTranscriberで再評価するアプリと採点器です。実音声・文字起こし・評価用Gold・結果JSONはこのリポジトリに含めません。必要な非公開データと準備手順は[モデル利用案内](../../docs/model_access.md)も参照してください。

## 方式

| 方法 | 音声入力 | 結果JSONのmethod |
|---|---|---|
| SFSpeechRecognizer | 約50秒区間をファイル入力 | `sf_file_fixed_ondevice` |
| SpeechTranscriber | 通話全体をファイル入力 | `st_file_ondevice` |
| SFSpeechRecognizer | 約50秒区間に0.5秒ずつ実時間入力 | `sf_stream_05_ondevice` |
| SpeechTranscriber | 通話全体に0.5秒ずつ実時間入力 | `st_stream_05_ondevice` |

アプリは4方式を同じ評価用音声で実行し、`results_4way_<dataset_id>.json`へ通話ごとに原子的に保存します。方式を選択でき、成功済みの通話を飛ばして再開できます。旧アプリとBundle ID・結果ファイルを分けています。

## 修正の背景

前回のファイル入力実験ではSFSpeechの126通話だけ成功し、SpeechTranscriberの126通話は全件「日本語モデル未導入」で失敗しました。`126/252`は処理停止位置ではなく、**成功件数**でした。言語の「対応」はモデル導入済みを意味しません。Appleの[SpeechTranscriber](https://developer.apple.com/documentation/speech/speechtranscriber)と[AssetInventory](https://developer.apple.com/documentation/speech/assetinventory)の状態を別に確認するよう修正しました。

旧SFSpeech実装は最初の`result.isFinal`で結果を確定していました。修正版は[認識タスクdelegate](https://developer.apple.com/documentation/speech/sfspeechrecognitiontaskdelegate)で発話を蓄積し、タスク全体の完了を待ちます。各区間で処理済み音声秒数を確認できない場合や、投入音声より短い場合は成功扱いせず、最初の失敗で一括実行を止めます。

## 実行

1. Xcode 26以降を備えたMacで`mac/Call119AppleEval.xcodeproj`を開き、`Signing & Capabilities`のTeamを設定してiPadOS 26以降のiPadへインストールします。iPad実機へのビルドはPro6000のLinux環境ではできません。
2. 非公開のPro6000環境にある`outputs/261014_ipad_apple_speech_eval/private/full126/full126_to_mac.zip`をMacへ取り寄せて展開し、`manifest.json`と全WAVをFinderのiPad「ファイル」→**119音声4方式評価**の書類直下へ転送します。音声と結果をGitHubや共有クラウドへ置かないでください。
3. アプリでmanifestを再読込し、ネット接続中に「日本語モデルを準備」を押します。**「モデル導入済み」**と表示されてから機内モードにします。SpeechTranscriberが未導入なら実行前に止まります。
4. 4方式を選び、まず「最初の1通話」で最終文とSFSpeechの区間診断を確認します。問題があれば結果JSONを回収して原因を調べます。問題がなければ同じ音声・同じアプリで「全通話」を実行します。iPadは給電し、アプリを前面表示します。
5. Finderから`results_4way_gold_v32_heldout_full126.json`をMacに回収し、Pro6000の非公開領域へ戻します。採点例：

```bash
export CALL119_ASR_RESEARCH_ROOT=/path/to/call119_asr_research
PY="$CALL119_ASR_RESEARCH_ROOT/outputs/260806_output/.venv/bin/python"
APP=/path/to/call119-streaming-asr/benchmark/ipad_apple_4way
"$PY" "$APP/server/score_results.py" --bundle full126 \
  --results "$APP/private/full126/returned_results.json"
```

コードの場所に合わせて`APP`を変更してください。採点には非公開の評価集合manifest・ID対応表・Gold v3.2評価コードが`CALL119_ASR_RESEARCH_ROOT`以下に必要です。採点結果は`private/full126/scored/`に出力されます。通話別の正解・認識結果は`private/`内に限定してください。

Appleには16 kHz PCM16、既存Whisperには8 kHz μ-law元音声を投入しています。またAppleの0.5秒方式は実時間待機し、既存Whisperは待機しません。この違いを含む参考比較としてCERを読んでください。
