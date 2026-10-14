import AVFoundation
import CoreMedia
import CryptoKit
import Foundation
import Speech

struct StreamEvent: Codable {
    let arrivalSec: Double
    let audioEndSec: Double
    let isFinal: Bool
    let textSha256: String
    let textCharacters: Int
}

struct FeedEvent: Codable {
    let audioEndSec: Double
    let sentAtSec: Double
}

struct SFSegmentDiagnostic: Codable {
    let audioOffsetSec: Double
    let inputAudioSec: Double
    let processedAudioSec: Double
    let finishedReadingAudio: Bool
    let finalUtterances: Int
    let partialUpdates: Int
    let textCharacters: Int
}

struct StreamOutcome {
    let text: String
    let events: [StreamEvent]
    let wallSec: Double
    let fedAudioSec: Double
    let deadlineMissCount: Int
    let feedEvents: [FeedEvent]
    var sfSegments: [SFSegmentDiagnostic] = []
}

private func textHash(_ value: String) -> String {
    SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
}

private final class SFCollector: NSObject, SFSpeechRecognitionTaskDelegate {
    private let lock = NSLock()
    private let callStarted: Double
    private let audioOffset: Double
    private let prefix: String
    private let onUpdate: @Sendable (String) -> Void
    private var continuation: CheckedContinuation<String, Error>?
    private var outcome: Result<String, Error>?
    private var storedEvents: [StreamEvent] = []
    private var finalParts: [String] = []
    private var latestPartial = ""
    private var processedAudioSec = 0.0
    private var finishedReading = false
    private var partialCount = 0

    init(callStarted: Double, audioOffset: Double, prefix: String,
         onUpdate: @escaping @Sendable (String) -> Void) {
        self.callStarted = callStarted
        self.audioOffset = audioOffset
        self.prefix = prefix
        self.onUpdate = onUpdate
        super.init()
    }

    func speechRecognitionTask(_ task: SFSpeechRecognitionTask,
                               didHypothesizeTranscription transcription: SFTranscription) {
        record(transcription, isFinal: false)
    }

    func speechRecognitionTask(_ task: SFSpeechRecognitionTask,
                               didFinishRecognition result: SFSpeechRecognitionResult) {
        record(result.bestTranscription, isFinal: true)
    }

    private func record(_ transcription: SFTranscription, isFinal: Bool) {
        let current = transcription.formattedString
        let end = transcription.segments.last.map { $0.timestamp + $0.duration } ?? 0
        var preview: String?
        lock.lock()
        if outcome == nil {
            storedEvents.append(StreamEvent(
                arrivalSec: ProcessInfo.processInfo.systemUptime - callStarted,
                audioEndSec: audioOffset + end,
                isFinal: isFinal,
                textSha256: textHash(current), textCharacters: current.count))
            if isFinal {
                if !current.isEmpty { finalParts.append(current) }
                latestPartial = ""
            } else {
                latestPartial = current
                partialCount += 1
            }
            let parts = finalParts + (latestPartial.isEmpty ? [] : [latestPartial])
            preview = prefix + parts.joined(separator: "\n")
        }
        lock.unlock()
        if let preview { onUpdate(preview) }
    }

    func speechRecognitionTask(_ task: SFSpeechRecognitionTask,
                               didProcessAudioDuration duration: TimeInterval) {
        lock.lock()
        processedAudioSec = max(processedAudioSec, duration)
        lock.unlock()
    }

    func speechRecognitionTaskFinishedReadingAudio(_ task: SFSpeechRecognitionTask) {
        lock.lock()
        finishedReading = true
        lock.unlock()
    }

    func speechRecognitionTask(_ task: SFSpeechRecognitionTask,
                               didFinishSuccessfully successfully: Bool) {
        var completion: CheckedContinuation<String, Error>?
        var answer: Result<String, Error>?
        lock.lock()
        if outcome == nil {
            if successfully, finalParts.isEmpty, !latestPartial.isEmpty {
                outcome = .failure(BenchmarkError.invalid("SFSpeechが最終結果を返さず、途中候補のみで終了しました"))
            } else if successfully {
                outcome = .success(finalParts.joined(separator: "\n"))
            } else if let error = task.error, (error as NSError).code == 1110 {
                // Appleの「認識可能な発話なし」は無音区間として空文を確定する。
                outcome = .success("")
            } else {
                outcome = .failure(task.error ?? BenchmarkError.invalid("SFSpeechの処理が失敗しました"))
            }
            completion = continuation
            continuation = nil
            answer = outcome
        }
        lock.unlock()
        if let completion, let answer { completion.resume(with: answer) }
    }

    func speechRecognitionTaskWasCancelled(_ task: SFSpeechRecognitionTask) {
        fail(task.error ?? BenchmarkError.invalid("SFSpeechの処理が中断されました"))
    }

    func fail(_ error: Error) {
        var completion: CheckedContinuation<String, Error>?
        lock.lock()
        if outcome == nil {
            outcome = .failure(error)
            completion = continuation
            continuation = nil
        }
        let answer = outcome
        lock.unlock()
        if let completion, let answer { completion.resume(with: answer) }
    }

    func value() async throws -> String {
        try await withCheckedThrowingContinuation { next in
            lock.lock()
            if let outcome {
                lock.unlock()
                next.resume(with: outcome)
            } else {
                continuation = next
                lock.unlock()
            }
        }
    }

    var events: [StreamEvent] {
        lock.lock()
        defer { lock.unlock() }
        return storedEvents
    }

    var processedDuration: Double {
        lock.lock()
        defer { lock.unlock() }
        return processedAudioSec
    }

    func diagnostic(inputAudioSec: Double) -> SFSegmentDiagnostic {
        lock.lock()
        defer { lock.unlock() }
        return SFSegmentDiagnostic(audioOffsetSec: audioOffset, inputAudioSec: inputAudioSec,
                                   processedAudioSec: processedAudioSec,
                                   finishedReadingAudio: finishedReading,
                                   finalUtterances: finalParts.count,
                                   partialUpdates: partialCount,
                                   textCharacters: (finalParts.isEmpty ? latestPartial :
                                                    finalParts.joined(separator: "\n")).count)
    }
}

private struct Phrase {
    let begin: Double
    let end: Double
    let text: String
}

private struct ProgressiveCollector {
    private var phrases: [Phrase] = []
    private(set) var events: [StreamEvent] = []

    mutating func receive(_ result: SpeechTranscriber.Result, callStarted: Double) -> String {
        let begin = max(0, CMTimeGetSeconds(result.range.start))
        let end = max(begin, CMTimeGetSeconds(result.range.end))
        let current = String(result.text.characters)
        // 揮発結果は同じ音声範囲を修正するため、旧候補を除いて最新候補に置く。
        let boundaryTolerance = 0.01
        phrases.removeAll { old in
            old.begin < end - boundaryTolerance && begin < old.end - boundaryTolerance
        }
        if !current.isEmpty { phrases.append(Phrase(begin: begin, end: end, text: current)) }
        phrases.sort { $0.begin < $1.begin }
        let snapshot = phrases.map(\.text).joined()
        events.append(StreamEvent(
            arrivalSec: ProcessInfo.processInfo.systemUptime - callStarted,
            audioEndSec: end, isFinal: result.isFinal,
            textSha256: textHash(snapshot), textCharacters: snapshot.count))
        return snapshot
    }

    var text: String { phrases.map(\.text).joined() }
}

private final class AudioInputBridge {
    private let outputFormat: AVAudioFormat
    private let converter: AVAudioConverter?
    private let ratio: Double

    init(source: AVAudioFormat, output: AVAudioFormat) throws {
        outputFormat = output
        ratio = output.sampleRate / source.sampleRate
        if source.isEqual(output) {
            converter = nil
        } else {
            guard let converter = AVAudioConverter(from: source, to: output) else {
                throw BenchmarkError.invalid("SpeechAnalyzer用の音声形式へ変換できません")
            }
            self.converter = converter
        }
    }

    func convert(_ input: AVAudioPCMBuffer) throws -> [AnalyzerInput] {
        guard let converter else { return [AnalyzerInput(buffer: input)] }
        let capacity = AVAudioFrameCount(max(4096, Int(ceil(Double(input.frameLength) * ratio)) + 1024))
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
            throw BenchmarkError.invalid("変換音声バッファを確保できません")
        }
        var supplied = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
            if supplied {
                inputStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            inputStatus.pointee = .haveData
            return input
        }
        if status == .error {
            if let conversionError { throw conversionError }
            throw BenchmarkError.invalid("音声形式変換に失敗")
        }
        return output.frameLength > 0 ? [AnalyzerInput(buffer: output)] : []
    }

    func flush() throws -> [AnalyzerInput] {
        guard let converter else { return [] }
        var converted: [AnalyzerInput] = []
        for _ in 0..<32 {
            guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 4096) else {
                throw BenchmarkError.invalid("変換音声バッファを確保できません")
            }
            var conversionError: NSError?
            let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
                inputStatus.pointee = .endOfStream
                return nil
            }
            if status == .error {
                if let conversionError { throw conversionError }
                throw BenchmarkError.invalid("変換終端で失敗")
            }
            if output.frameLength > 0 { converted.append(AnalyzerInput(buffer: output)) }
            if status == .endOfStream || output.frameLength == 0 { break }
        }
        return converted
    }
}

enum StreamEngines {
    static let japanese = Locale(identifier: "ja-JP")

    static func legacyRecognizer() throws -> SFSpeechRecognizer {
        guard let recognizer = SFSpeechRecognizer(locale: japanese), recognizer.isAvailable,
              recognizer.supportsOnDeviceRecognition else {
            throw BenchmarkError.invalid("このiPadの日本語SFSpeechは端末内認識に非対応です")
        }
        return recognizer
    }

    static func requestLegacyAuthorization() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }

    static func modernLocale() async throws -> Locale {
        guard SpeechTranscriber.isAvailable,
              let locale = await SpeechTranscriber.supportedLocale(equivalentTo: japanese) else {
            throw BenchmarkError.invalid("このiPadではSpeechTranscriber日本語が使用できません")
        }
        return locale
    }

    private static func modernModules() async throws -> [SpeechTranscriber] {
        let locale = try await modernLocale()
        return [SpeechTranscriber(locale: locale, preset: .transcription),
                SpeechTranscriber(locale: locale, transcriptionOptions: [],
                                  reportingOptions: [.volatileResults],
                                  attributeOptions: [.audioTimeRange])]
    }

    static func modernAssetStatus() async throws -> AssetInventory.Status {
        let modules = try await modernModules()
        return await AssetInventory.status(forModules: modules)
    }

    static func requireModernInstalled() async throws {
        let status = try await modernAssetStatus()
        if case .installed = status { return }
        throw BenchmarkError.invalid("SpeechTranscriber日本語モデルが未導入です。ネット接続中に「日本語モデルを準備」を完了してください")
    }

    static func installModernAsset() async throws {
        let modules = try await modernModules()
        if let request = try await AssetInventory.assetInstallationRequest(supporting: modules) {
            try await request.downloadAndInstall()
        }
        // downloadAndInstall()の戻りだけでは導入完了を意味しない場合がある。
        for _ in 0..<300 {
            let status = await AssetInventory.status(forModules: modules)
            if case .installed = status { return }
            try await Task.sleep(nanoseconds: 2_000_000_000)
        }
        throw BenchmarkError.invalid("モデル導入が完了しませんでした。ネット接続・空き容量を確認し、再度準備してください")
    }

    private static func pace(until target: Double) async throws -> Bool {
        try Task.checkCancellation()
        let remaining = target - ProcessInfo.processInfo.systemUptime
        if remaining > 0 {
            try await Task.sleep(nanoseconds: UInt64(remaining * 1_000_000_000))
            return false
        }
        return remaining < -0.05
    }

    private static func legacySegment(
        url: URL, callStarted: Double, audioOffset: Double, prefix: String,
        onUpdate: @escaping @Sendable (String) -> Void
    ) async throws -> StreamOutcome {
        let recognizer = try legacyRecognizer()
        let file = try AVAudioFile(forReading: url)
        let rate = file.processingFormat.sampleRate
        let frames = AVAudioFrameCount(rate / 2)
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true
        request.addsPunctuation = false
        request.taskHint = .dictation
        let collector = SFCollector(callStarted: callStarted, audioOffset: audioOffset,
                                    prefix: prefix, onUpdate: onUpdate)
        let task = recognizer.recognitionTask(with: request, delegate: collector)
        let timeout = DispatchWorkItem {
            collector.fail(BenchmarkError.invalid("SFSpeechの区間処理が120秒で確定しませんでした"))
            task.cancel()
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + 120, execute: timeout)
        defer { timeout.cancel() }
        let started = ProcessInfo.processInfo.systemUptime
        var sent: Int64 = 0
        var missed = 0
        var feedEvents: [FeedEvent] = []
        do {
            while file.framePosition < file.length {
                try Task.checkCancellation()
                guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frames) else {
                    throw BenchmarkError.invalid("音声バッファを確保できません")
                }
                try file.read(into: buffer, frameCount: frames)
                if buffer.frameLength == 0 { break }
                sent += Int64(buffer.frameLength)
                if try await pace(until: started + Double(sent) / rate) { missed += 1 }
                request.append(buffer)
                feedEvents.append(FeedEvent(audioEndSec: audioOffset + Double(sent) / rate,
                                            sentAtSec: ProcessInfo.processInfo.systemUptime - callStarted))
            }
            request.endAudio()
            let text = try await collector.value()
            let processed = collector.processedDuration
            if processed <= 0 {
                throw BenchmarkError.invalid("SFSpeechの処理済み音声長を確認できません。区間\(String(format: "%.1f", audioOffset))秒")
            }
            if processed + 1.5 < Double(sent) / rate {
                throw BenchmarkError.invalid(
                    "SFSpeechが投入音声の途中で終了しました（処理\(String(format: "%.1f", processed))秒／投入\(String(format: "%.1f", Double(sent) / rate))秒）")
            }
            return StreamOutcome(text: text, events: collector.events,
                                 wallSec: ProcessInfo.processInfo.systemUptime - started,
                                 fedAudioSec: Double(sent) / rate, deadlineMissCount: missed,
                                 feedEvents: feedEvents,
                                 sfSegments: [collector.diagnostic(inputAudioSec: Double(sent) / rate)])
        } catch {
            task.cancel()
            collector.fail(error)
            throw error
        }
    }

    private static func legacyFileSegment(
        url: URL, callStarted: Double, audioOffset: Double, prefix: String,
        onUpdate: @escaping @Sendable (String) -> Void
    ) async throws -> StreamOutcome {
        let recognizer = try legacyRecognizer()
        let file = try AVAudioFile(forReading: url)
        let audioSec = Double(file.length) / file.processingFormat.sampleRate
        let request = SFSpeechURLRecognitionRequest(url: url)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true
        request.addsPunctuation = false
        request.taskHint = .dictation
        let collector = SFCollector(callStarted: callStarted, audioOffset: audioOffset,
                                    prefix: prefix, onUpdate: onUpdate)
        let started = ProcessInfo.processInfo.systemUptime
        let task = recognizer.recognitionTask(with: request, delegate: collector)
        let timeout = DispatchWorkItem {
            collector.fail(BenchmarkError.invalid("SFSpeechのファイル区間が180秒で完了しませんでした"))
            task.cancel()
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + 180, execute: timeout)
        defer { timeout.cancel() }
        do {
            let text = try await collector.value()
            let processed = collector.processedDuration
            if processed <= 0 {
                throw BenchmarkError.invalid("SFSpeechの処理済み音声長を確認できません。区間\(String(format: "%.1f", audioOffset))秒")
            }
            if processed + 1.5 < audioSec {
                throw BenchmarkError.invalid(
                    "SFSpeechがファイルの途中で終了しました（処理\(String(format: "%.1f", processed))秒／区間\(String(format: "%.1f", audioSec))秒）")
            }
            return StreamOutcome(text: text, events: collector.events,
                                 wallSec: ProcessInfo.processInfo.systemUptime - started,
                                 fedAudioSec: audioSec, deadlineMissCount: 0, feedEvents: [],
                                 sfSegments: [collector.diagnostic(inputAudioSec: audioSec)])
        } catch {
            task.cancel()
            collector.fail(error)
            throw error
        }
    }

    static func legacyFileCall(
        segments: [(URL, Double)], onUpdate: @escaping @Sendable (String) -> Void
    ) async throws -> StreamOutcome {
        let callStarted = ProcessInfo.processInfo.systemUptime
        var parts: [String] = []
        var events: [StreamEvent] = []
        var diagnostics: [SFSegmentDiagnostic] = []
        var fed = 0.0
        for (url, offset) in segments {
            try Task.checkCancellation()
            let prefix = parts.isEmpty ? "" : parts.joined(separator: "\n") + "\n"
            let result = try await legacyFileSegment(url: url, callStarted: callStarted,
                                                     audioOffset: offset, prefix: prefix,
                                                     onUpdate: onUpdate)
            parts.append(result.text)
            events.append(contentsOf: result.events)
            diagnostics.append(contentsOf: result.sfSegments)
            fed += result.fedAudioSec
            onUpdate(parts.joined(separator: "\n"))
        }
        return StreamOutcome(text: parts.joined(separator: "\n"), events: events,
                             wallSec: ProcessInfo.processInfo.systemUptime - callStarted,
                             fedAudioSec: fed, deadlineMissCount: 0, feedEvents: [],
                             sfSegments: diagnostics)
    }

    static func modernFileCall(
        url: URL, onUpdate: @escaping @Sendable (String) -> Void
    ) async throws -> StreamOutcome {
        try await requireModernInstalled()
        let transcriber = SpeechTranscriber(locale: try await modernLocale(), preset: .transcription)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let file = try AVAudioFile(forReading: url)
        let audioSec = Double(file.length) / file.processingFormat.sampleRate
        let started = ProcessInfo.processInfo.systemUptime
        let consumer = Task { () throws -> String in
            var finalized = ""
            var partial = ""
            for try await result in transcriber.results {
                let current = String(result.text.characters)
                if result.isFinal {
                    finalized += current
                    partial = ""
                } else {
                    partial = current
                }
                onUpdate(finalized + partial)
            }
            return finalized + partial
        }
        do {
            guard let lastSample = try await analyzer.analyzeSequence(from: file) else {
                throw BenchmarkError.invalid("SpeechTranscriberが音声を読み取れませんでした")
            }
            let analyzedSec = CMTimeGetSeconds(lastSample)
            if analyzedSec.isFinite, analyzedSec + 1.5 < audioSec {
                throw BenchmarkError.invalid("SpeechTranscriberの解析音声長が不足しています")
            }
            try await analyzer.finalizeAndFinish(through: lastSample)
            let text = try await consumer.value
            return StreamOutcome(text: text, events: [],
                                 wallSec: ProcessInfo.processInfo.systemUptime - started,
                                 fedAudioSec: audioSec, deadlineMissCount: 0, feedEvents: [])
        } catch {
            consumer.cancel()
            await analyzer.cancelAndFinishNow()
            throw error
        }
    }

    static func legacyCall(
        segments: [(URL, Double)], onUpdate: @escaping @Sendable (String) -> Void
    ) async throws -> StreamOutcome {
        let callStarted = ProcessInfo.processInfo.systemUptime
        var parts: [String] = []
        var events: [StreamEvent] = []
        var fed = 0.0
        var misses = 0
        var feedEvents: [FeedEvent] = []
        var diagnostics: [SFSegmentDiagnostic] = []
        for (url, offset) in segments {
            let prefix = parts.isEmpty ? "" : parts.joined(separator: "\n") + "\n"
            let result = try await legacySegment(url: url, callStarted: callStarted,
                                                 audioOffset: offset, prefix: prefix,
                                                 onUpdate: onUpdate)
            parts.append(result.text)
            events.append(contentsOf: result.events)
            fed += result.fedAudioSec
            misses += result.deadlineMissCount
            feedEvents.append(contentsOf: result.feedEvents)
            diagnostics.append(contentsOf: result.sfSegments)
            onUpdate(parts.joined(separator: "\n"))
        }
        return StreamOutcome(text: parts.joined(separator: "\n"), events: events,
                             wallSec: ProcessInfo.processInfo.systemUptime - callStarted,
                             fedAudioSec: fed, deadlineMissCount: misses,
                             feedEvents: feedEvents, sfSegments: diagnostics)
    }

    static func modernCall(
        url: URL, onUpdate: @escaping @Sendable (String) -> Void
    ) async throws -> StreamOutcome {
        let locale = try await modernLocale()
        let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [],
                                            reportingOptions: [.volatileResults],
                                            attributeOptions: [.audioTimeRange])
        try await requireModernInstalled()
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let file = try AVAudioFile(forReading: url)
        guard let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
            throw BenchmarkError.invalid("SpeechAnalyzer用の音声形式がありません")
        }
        let bridge = try AudioInputBridge(source: file.processingFormat, output: analyzerFormat)
        let rate = file.processingFormat.sampleRate
        let frames = AVAudioFrameCount(rate / 2)
        let (inputs, builder) = AsyncStream.makeStream(of: AnalyzerInput.self)
        let started = ProcessInfo.processInfo.systemUptime
        let consumer = Task { () throws -> ProgressiveCollector in
            var collector = ProgressiveCollector()
            for try await result in transcriber.results {
                onUpdate(collector.receive(result, callStarted: started))
            }
            return collector
        }
        let analysis = Task { () throws -> Void in
            if let last = try await analyzer.analyzeSequence(inputs) {
                try await analyzer.finalizeAndFinish(through: last)
            } else {
                await analyzer.cancelAndFinishNow()
            }
        }
        var sent: Int64 = 0
        var missed = 0
        var feedEvents: [FeedEvent] = []
        do {
            while file.framePosition < file.length {
                try Task.checkCancellation()
                guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frames) else {
                    throw BenchmarkError.invalid("音声バッファを確保できません")
                }
                try file.read(into: buffer, frameCount: frames)
                if buffer.frameLength == 0 { break }
                sent += Int64(buffer.frameLength)
                if try await pace(until: started + Double(sent) / rate) { missed += 1 }
                for input in try bridge.convert(buffer) { builder.yield(input) }
                feedEvents.append(FeedEvent(audioEndSec: Double(sent) / rate,
                                            sentAtSec: ProcessInfo.processInfo.systemUptime - started))
            }
            for input in try bridge.flush() { builder.yield(input) }
            builder.finish()
            try await analysis.value
            let collected = try await consumer.value
            return StreamOutcome(text: collected.text, events: collected.events,
                                 wallSec: ProcessInfo.processInfo.systemUptime - started,
                                 fedAudioSec: Double(sent) / rate, deadlineMissCount: missed,
                                 feedEvents: feedEvents)
        } catch {
            builder.finish()
            analysis.cancel()
            consumer.cancel()
            await analyzer.cancelAndFinishNow()
            throw error
        }
    }
}
