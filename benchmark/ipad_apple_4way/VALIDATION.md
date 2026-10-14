# 修正版の確認記録

- 旧結果JSONの126/252を再確認：SFSpeech 126/126成功、SpeechTranscriber 0/126成功。SpeechTranscriberの全126件のエラーは日本語モデル未導入。
- Apple公式のSpeechTranscriber・AssetInventory仕様を確認：対応言語と導入済みモデルは別。新アプリは両構成の`AssetInventory.status(forModules:)`を表示・検証する。
- SFSpeechの旧コードは最初の`result.isFinal`で終了していた。新アプリは`SFSpeechRecognitionTaskDelegate`で発話を蓄積し、`didFinishSuccessfully`を待つ。区間の処理済み音声秒数が不明・不足なら失敗にする。
- Python採点器の構文検査と、4方式各1件の**合成結果**による採点を実施：4方式すべて1件成功として読めた。SFSpeechの処理済み音声長を意図的に不足させた結果は採点前に拒否された。
- `Info.plist`のXML読込と、配布ZIPの整合性を確認。

Pro6000はLinuxでXcode・Apple Speechフレームワークを利用できない。SwiftのiPad実機ビルド、モデルダウンロード、1通話・126通話の認識動作は未実施。まずMacでビルドし、1通話を実行して結果JSONを確認する。
