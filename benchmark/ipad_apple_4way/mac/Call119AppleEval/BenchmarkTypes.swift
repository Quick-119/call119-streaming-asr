import Foundation

struct SegmentItem: Codable {
    let file: String
    let sha256: String
    let startSample: Int
    let endSample: Int
}

struct CaseItem: Codable, Identifiable {
    let id: String
    let file: String
    let sha256: String
    let durationSec: Double
    let segments: [SegmentItem]
}

struct DatasetManifest: Codable {
    let schemaVersion: Int
    let datasetId: String
    let source: String
    let audioFormat: String
    let segmentation: String
    let cases: [CaseItem]
}

struct ResultRecord: Codable {
    let id: String
    let method: String
    let status: String
    let text: String
    let audioSha256: String
    let durationSec: Double
    let wallSec: Double
    let error: String?
    let deviceModel: String
    let systemVersion: String
    let fedAudioSec: Double
    let deadlineMissCount: Int
    let events: [StreamEvent]
    let feedEvents: [FeedEvent]
    let sfSegments: [SFSegmentDiagnostic]
}

struct ResultDocument: Codable {
    let schemaVersion: Int
    let datasetId: String
    let manifestSha256: String
    let appRevision: Int
    var records: [ResultRecord]
}

enum BenchmarkError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message): return message
        }
    }
}

enum RecognitionMethod: String, CaseIterable, Identifiable {
    case sfFile = "sf_file_fixed_ondevice"
    case stFile = "st_file_ondevice"
    case sfStream = "sf_stream_05_ondevice"
    case stStream = "st_stream_05_ondevice"

    var id: String { rawValue }
    var title: String {
        switch self {
        case .sfFile: return "SFSpeech（ファイル入力・修正版）"
        case .stFile: return "SpeechTranscriber（ファイル入力）"
        case .sfStream: return "SFSpeechRecognizer（0.5秒入力・約50秒区間）"
        case .stStream: return "SpeechTranscriber（0.5秒入力・全通話）"
        }
    }
}

func makeDecoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
}

func makeEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return encoder
}
