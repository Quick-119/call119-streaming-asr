import CryptoKit
import Foundation
import Speech
import SwiftUI
import UIKit

@MainActor
final class BenchmarkViewModel: ObservableObject {
    @Published var status = "manifest.jsonをアプリの書類に入れてください"
    @Published var liveTranscript = ""
    @Published var busy = false
    @Published var dataset: DatasetManifest?
    @Published var completed = 0
    @Published var failed = 0
    @Published var total = 0
    @Published var modernSupport = "未確認"
    @Published var legacySupport = "未確認"
    @Published var runSFFile = true
    @Published var runSTFile = true
    @Published var runSFStream = true
    @Published var runSTStream = true

    private var resultDocument: ResultDocument?
    private var runTask: Task<Void, Never>?
    private var documentDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private func fileHash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1024 * 1024) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func resultURL(for manifest: DatasetManifest) -> URL {
        documentDirectory.appendingPathComponent("results_4way_\(manifest.datasetId).json")
    }

    private func excludeFromBackup(_ input: URL) throws {
        var url = input
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try url.setResourceValues(values)
    }

    private func updateCounts(_ results: ResultDocument, calls: Int) {
        completed = results.records.filter { $0.status == "ok" }.count
        failed = results.records.filter { $0.status == "error" }.count
        total = calls * RecognitionMethod.allCases.count
    }

    func load() {
        guard !busy else { return }
        do {
            let url = documentDirectory.appendingPathComponent("manifest.json")
            let data = try Data(contentsOf: url)
            let parsed = try makeDecoder().decode(DatasetManifest.self, from: data)
            guard parsed.schemaVersion == 1, !parsed.cases.isEmpty else {
                throw BenchmarkError.invalid("manifest.jsonの形式が違います")
            }
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            try excludeFromBackup(url)
            for item in parsed.cases {
                let full = documentDirectory.appendingPathComponent(item.file)
                if FileManager.default.fileExists(atPath: full.path) { try excludeFromBackup(full) }
                for part in item.segments {
                    let segment = documentDirectory.appendingPathComponent(part.file)
                    if FileManager.default.fileExists(atPath: segment.path) { try excludeFromBackup(segment) }
                }
            }
            var results = ResultDocument(schemaVersion: 3, datasetId: parsed.datasetId,
                                         manifestSha256: hash, appRevision: 1, records: [])
            let output = resultURL(for: parsed)
            if FileManager.default.fileExists(atPath: output.path) {
                try excludeFromBackup(output)
                results = try makeDecoder().decode(ResultDocument.self, from: Data(contentsOf: output))
                guard results.schemaVersion == 3, results.appRevision == 1,
                      results.datasetId == parsed.datasetId, results.manifestSha256 == hash else {
                    throw BenchmarkError.invalid("既存結果とmanifestまたはアプリ版が異なります")
                }
            }
            dataset = parsed
            resultDocument = results
            updateCounts(results, calls: parsed.cases.count)
            status = "\(parsed.cases.count)通話。成功\(completed)、失敗\(failed)、対象\(total)"
        } catch {
            dataset = nil
            resultDocument = nil
            status = "読込失敗：\(error.localizedDescription)"
        }
    }

    func checkSupport() {
        Task {
            do {
                _ = try StreamEngines.legacyRecognizer()
                legacySupport = "端末内日本語に対応"
            } catch { legacySupport = error.localizedDescription }
            do {
                let locale = try await StreamEngines.modernLocale()
                let asset = try await StreamEngines.modernAssetStatus()
                switch asset {
                case .installed: modernSupport = "\(locale.identifier)：モデル導入済み"
                case .downloading: modernSupport = "\(locale.identifier)：ダウンロード中"
                case .supported: modernSupport = "\(locale.identifier)：対応、モデル未導入"
                case .unsupported: modernSupport = "\(locale.identifier)：モデル非対応"
                @unknown default: modernSupport = "\(locale.identifier)：状態不明"
                }
            } catch { modernSupport = error.localizedDescription }
        }
    }

    func installModern() {
        guard !busy else { return }
        busy = true
        status = "SpeechTranscriber日本語モデルを準備中。ネット接続を維持してください"
        Task {
            do {
                try await StreamEngines.installModernAsset()
                status = "モデル導入を確認しました。4方式の実験を開始できます"
            } catch { status = "モデル準備失敗：\(error.localizedDescription)" }
            busy = false
            checkSupport()
        }
    }

    private func save(_ results: ResultDocument, for manifest: DatasetManifest) throws {
        let url = resultURL(for: manifest)
        try makeEncoder().encode(results).write(to: url, options: .atomic)
        try excludeFromBackup(url)
        resultDocument = results
        updateCounts(results, calls: manifest.cases.count)
    }

    func cancel() {
        runTask?.cancel()
        status = "停止処理中。現在の結果を保存します"
    }

    func run(limit: Int) {
        guard !busy, let dataset, var results = resultDocument else { return }
        let methods = RecognitionMethod.allCases.filter { method in
            switch method {
            case .sfFile: return runSFFile
            case .stFile: return runSTFile
            case .sfStream: return runSFStream
            case .stStream: return runSTStream
            }
        }
        guard !methods.isEmpty else { status = "実行方式を選択してください"; return }
        busy = true
        liveTranscript = ""
        runTask = Task {
            do {
                if methods.contains(.sfFile) || methods.contains(.sfStream) {
                    guard await StreamEngines.requestLegacyAuthorization() else {
                        throw BenchmarkError.invalid("SFSpeechの音声認識権限を許可してください")
                    }
                    _ = try StreamEngines.legacyRecognizer()
                }
                if methods.contains(.stFile) || methods.contains(.stStream) {
                    try await StreamEngines.requireModernInstalled()
                }
            } catch {
                status = "実行前確認で停止：\(error.localizedDescription)"
                busy = false
                runTask = nil
                checkSupport()
                return
            }

            let selected = Array(dataset.cases.prefix(limit))
            var stoppedByError = false
            methodLoop: for method in methods {
                for item in selected {
                    if Task.isCancelled { break methodLoop }
                    if results.records.contains(where: {
                        $0.id == item.id && $0.method == method.rawValue && $0.status == "ok"
                    }) { continue }
                    status = "\(method.title)：\(item.id)（成功\(completed)、失敗\(failed)）"
                    liveTranscript = ""
                    let started = ProcessInfo.processInfo.systemUptime
                    var outcome: StreamOutcome?
                    var errorText: String?
                    do {
                        let fullURL = documentDirectory.appendingPathComponent(item.file)
                        guard try fileHash(fullURL) == item.sha256 else {
                            throw BenchmarkError.invalid("音声SHA-256不一致：\(item.file)")
                        }
                        let update: @Sendable (String) -> Void = { [weak self] snapshot in
                            Task { @MainActor in self?.liveTranscript = snapshot }
                        }
                        switch method {
                        case .sfFile, .sfStream:
                            var segments: [(URL, Double)] = []
                            for part in item.segments {
                                let url = documentDirectory.appendingPathComponent(part.file)
                                guard try fileHash(url) == part.sha256 else {
                                    throw BenchmarkError.invalid("区間SHA-256不一致：\(part.file)")
                                }
                                segments.append((url, Double(part.startSample) / 16000.0))
                            }
                            if method == .sfFile {
                                outcome = try await StreamEngines.legacyFileCall(segments: segments,
                                                                                  onUpdate: update)
                            } else {
                                outcome = try await StreamEngines.legacyCall(segments: segments,
                                                                              onUpdate: update)
                            }
                        case .stFile:
                            outcome = try await StreamEngines.modernFileCall(url: fullURL,
                                                                              onUpdate: update)
                        case .stStream:
                            outcome = try await StreamEngines.modernCall(url: fullURL,
                                                                          onUpdate: update)
                        }
                        guard let outcome, abs(outcome.fedAudioSec - item.durationSec) < 0.02 else {
                            throw BenchmarkError.invalid("音声全体を解析できていません")
                        }
                    } catch {
                        errorText = error.localizedDescription
                    }
                    let record = ResultRecord(id: item.id, method: method.rawValue,
                                              status: Task.isCancelled ? "cancelled" :
                                                (errorText == nil ? "ok" : "error"),
                                              text: outcome?.text ?? "", audioSha256: item.sha256,
                                              durationSec: item.durationSec,
                                              wallSec: ProcessInfo.processInfo.systemUptime - started,
                                              error: errorText, deviceModel: UIDevice.current.model,
                                              systemVersion: UIDevice.current.systemVersion,
                                              fedAudioSec: outcome?.fedAudioSec ?? 0,
                                              deadlineMissCount: outcome?.deadlineMissCount ?? 0,
                                              events: outcome?.events ?? [],
                                              feedEvents: outcome?.feedEvents ?? [],
                                              sfSegments: outcome?.sfSegments ?? [])
                    results.records.removeAll { $0.id == item.id && $0.method == method.rawValue }
                    results.records.append(record)
                    do { try save(results, for: dataset) }
                    catch {
                        status = "結果保存失敗：\(error.localizedDescription)"
                        busy = false
                        runTask = nil
                        return
                    }
                    if let errorText, !Task.isCancelled {
                        status = "\(method.title)の\(item.id)で停止：\(errorText)"
                        stoppedByError = true
                        break methodLoop
                    }
                }
            }
            if !stoppedByError {
                status = Task.isCancelled
                    ? "停止しました。成功\(completed)、失敗\(failed)。再開できます"
                    : "終了：成功\(completed)、失敗\(failed)。results_4way_*.jsonをMacへ戻してください"
            }
            busy = false
            runTask = nil
        }
    }
}
