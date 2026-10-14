import SwiftUI

struct ContentView: View {
    @StateObject private var model = BenchmarkViewModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("評価集合") {
                    Text(model.dataset?.datasetId ?? "未読込")
                    Text(model.status).font(.footnote)
                    Button("manifest.jsonを再読込") { model.load() }
                        .disabled(model.busy)
                }
                Section("端末対応") {
                    LabeledContent("SFSpeech", value: model.legacySupport)
                    LabeledContent("SpeechTranscriber", value: model.modernSupport)
                    Button("対応状況を確認") { model.checkSupport() }
                    Button("日本語モデルを準備") { model.installModern() }
                        .disabled(model.busy)
                }
                Section("実行方式") {
                    Toggle("SFSpeech・ファイル入力", isOn: $model.runSFFile)
                    Toggle("SpeechTranscriber・ファイル入力", isOn: $model.runSTFile)
                    Toggle("SFSpeech・0.5秒入力", isOn: $model.runSFStream)
                    Toggle("SpeechTranscriber・0.5秒入力", isOn: $model.runSTStream)
                }
                .disabled(model.busy)
                Section("実通報CER評価") {
                    Text("成功 \(model.completed)・失敗 \(model.failed)・対象 \(model.total)")
                    Button("最初の1通話") { model.run(limit: 1) }
                        .disabled(model.busy)
                    Button("最初の10通話") { model.run(limit: 10) }
                        .disabled(model.busy)
                    Button("全通話") { model.run(limit: Int.max) }
                        .disabled(model.busy)
                    if model.busy {
                        Button("停止して保存") { model.cancel() }
                            .tint(.red)
                    }
                }
                .disabled(model.dataset == nil)
                Section("現在の文字起こし") {
                    ScrollView {
                        Text(model.liveTranscript.isEmpty ? "音声投入後に表示します" : model.liveTranscript)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(minHeight: 200, maxHeight: 380)
                }
            }
            .navigationTitle("119音声・4方式評価")
            .onAppear {
                model.load()
                model.checkSupport()
            }
        }
    }
}
