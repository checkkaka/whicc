import Foundation

/// One line in the JSONL stream produced by `translate_stream.py` / `whicc.py`.
///
/// Field names and JSON keys are kept identical to the wire format
/// so the Python side needs no change.
struct TranslationEvent: Decodable, Equatable, Sendable {
    let eventType: String
    let sourceKey: String?
    let revision: Int?
    let eventMonoNs: UInt64?
    let speechStartMonoNs: UInt64?
    let speechEndMonoNs: UInt64?
    let finalReusedPartial: Bool?
    let sourceUpdateMode: String?
    let sourceText: String?
    let deltaSourceText: String?
    let translatedDeltaText: String?
    let translatedFullText: String?
    let translateMs: Double?
    let sharedPrefixLen: Int?
    let glossaryHits: [String]?
    let retried: Bool?
    let fallbackReason: String?
    let error: String?
    // whicc transcription fields
    let text: String?
    let status: String?
    let statusColor: String?
    // 流式翻译字段。短 SSE 通常只写完整结果；长 SSE 可写累计全文，且同一
    // source_key 的跨 revision 展示统一限为最多约 1 次/秒。历史事件仍兼容。
    let isStreamingToken: Bool?
    let streamingPiece: String?

    enum CodingKeys: String, CodingKey {
        case eventType = "event_type"
        case sourceKey = "source_key"
        case revision
        case eventMonoNs = "event_mono_ns"
        case speechStartMonoNs = "speech_start_mono_ns"
        case speechEndMonoNs = "speech_end_mono_ns"
        case finalReusedPartial = "final_reused_partial"
        case sourceUpdateMode = "source_update_mode"
        case sourceText = "source_text"
        case deltaSourceText = "delta_source_text"
        case translatedDeltaText = "translated_delta_text"
        case translatedFullText = "translated_full_text"
        case translateMs = "translate_ms"
        case sharedPrefixLen = "shared_prefix_len"
        case glossaryHits = "glossary_hits"
        case retried
        case fallbackReason = "fallback_reason"
        case error
        case text
        case status
        case statusColor = "status_color"
        case isStreamingToken = "is_streaming_token"
        case streamingPiece = "streaming_piece"
    }
}

// MARK: - Event kind helpers

extension TranslationEvent {
    var isTranslationFinal: Bool { eventType == "translation_final" || eventType == "translation_reset" }
    var isTranslationPartial: Bool { eventType == "translation_partial" }
    var isTranslationError: Bool { eventType == "translation_error" }
    var isPartial: Bool { eventType == "partial" }
    var isFinal: Bool { eventType == "final" }
    var isStatus: Bool { eventType == "status" }
}
