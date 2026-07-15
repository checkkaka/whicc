import XCTest
@testable import whicc_macui

@MainActor
final class OverlayStateTests: XCTestCase {
    private func event(_ json: String) throws -> TranslationEvent {
        try JSONDecoder().decode(TranslationEvent.self, from: Data(json.utf8))
    }

    func testPreviousFinalDoesNotClearNextSourceDraft() throws {
        let state = OverlayState()
        state.applyTranscription(try event("""
            {"event_type":"partial","source_key":"next","revision":1,"text":"Next sentence"}
            """))
        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"next","revision":1,
             "source_text":"Next sentence","translated_full_text":"下一句",
             "is_streaming_token":true,"partial_complete":true}
            """))

        state.apply(try event("""
            {"event_type":"translation_final","source_key":"previous","revision":3,
             "source_text":"Previous sentence","translated_full_text":"上一句"}
            """))

        XCTAssertEqual(state.draftSourceText, "Next sentence")
        XCTAssertEqual(state.draftTranslatedText, "下一句")
    }

    func testPreviousTranslationDraftCannotReplaceNextSourceDraft() throws {
        let state = OverlayState()
        state.applyTranscription(try event("""
            {"event_type":"partial","source_key":"next","revision":1,"text":"Next sentence"}
            """))

        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"previous","revision":3,
             "source_text":"Previous sentence","translated_full_text":"上一句草稿",
             "is_streaming_token":true}
            """))
        state.apply(try event("""
            {"event_type":"translation_final","source_key":"previous","revision":3,
             "source_text":"Previous sentence","translated_full_text":"上一句"}
            """))

        XCTAssertEqual(state.draftSourceText, "Next sentence")
        XCTAssertNil(state.draftTranslatedText)
    }

    func testOlderTranslationRevisionCannotReplaceNewerSourceRevision() throws {
        let state = OverlayState()
        state.applyTranscription(try event("""
            {"event_type":"partial","source_key":"same","revision":5,"text":"Newest source"}
            """))

        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"same","revision":4,
             "source_text":"Stale source","translated_full_text":"过期译文",
             "is_streaming_token":true}
            """))

        XCTAssertEqual(state.draftSourceText, "Newest source")
        XCTAssertNil(state.draftTranslatedText)
    }

    func testNextSourceDraftClearsPreviousTranslationBeforePreviousFinalArrives() throws {
        let state = OverlayState()
        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"previous","revision":3,
             "source_text":"Previous sentence","translated_full_text":"上一句草稿",
             "is_streaming_token":true}
            """))

        state.applyTranscription(try event("""
            {"event_type":"partial","source_key":"next","revision":1,"text":"Next sentence"}
            """))

        XCTAssertEqual(state.draftSourceText, "Next sentence")
        XCTAssertNil(state.draftTranslatedText)

        state.apply(try event("""
            {"event_type":"translation_final","source_key":"previous","revision":3,
             "source_text":"Previous sentence","translated_full_text":"上一句"}
            """))

        XCTAssertEqual(state.draftSourceText, "Next sentence")
        XCTAssertNil(state.draftTranslatedText)
    }

    func testPartialCannotReappearAfterSameSourceFinal() throws {
        let state = OverlayState()
        state.apply(try event("""
            {"event_type":"translation_final","source_key":"same","revision":3,
             "source_text":"Stable source","translated_full_text":"稳定译文"}
            """))

        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"same","revision":3,
             "source_text":"Stale draft","translated_full_text":"迟到草稿",
             "is_streaming_token":true}
            """))

        XCTAssertEqual(state.committed?.sourceText, "Stable source")
        XCTAssertNil(state.draftSourceText)
        XCTAssertNil(state.draftTranslatedText)
    }

    func testTranslationEventDecodesOptionalPartialComplete() throws {
        XCTAssertTrue(try event("""
            {"event_type":"translation_partial","partial_complete":true}
            """).partialComplete == true)
        XCTAssertNil(try event("""
            {"event_type":"translation_partial"}
            """).partialComplete)
    }

    func testNewSourceRevisionKeepsTranslationAndTranslationNeverRevertsSource() throws {
        let state = OverlayState()
        state.applyTranscription(try event("""
            {"event_type":"partial","source_key":"same","revision":1,"text":"First source"}
            """))
        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"same","revision":1,
             "source_text":"First source","translated_full_text":"第一版译文",
             "partial_complete":true}
            """))

        state.applyTranscription(try event("""
            {"event_type":"partial","source_key":"same","revision":2,"text":"Newest source"}
            """))
        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"same","revision":1,
             "source_text":"First source","translated_full_text":"第一版译文已经完成",
             "partial_complete":true}
            """))

        XCTAssertEqual(state.draftSourceText, "Newest source")
        XCTAssertEqual(state.draftTranslatedText, "第一版译文已经完成")
    }

    func testNewTranslationRevisionWaitsUntilCompleteBeforeAtomicReplacement() throws {
        let state = OverlayState()
        state.applyTranscription(try event("""
            {"event_type":"partial","source_key":"same","revision":1,"text":"Source"}
            """))
        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"same","revision":1,
             "translated_full_text":"旧版完整译文","partial_complete":true}
            """))
        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"same","revision":2,
             "translated_full_text":"新"}
            """))
        XCTAssertEqual(state.draftTranslatedText, "旧版完整译文")

        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"same","revision":2,
             "translated_full_text":"新版","partial_complete":true}
            """))
        XCTAssertEqual(state.draftTranslatedText, "新版")
    }

    func testFirstTranslationWaitsForEightCharactersButCompleteShortSentenceDisplays() throws {
        let state = OverlayState()
        state.applyTranscription(try event("""
            {"event_type":"partial","source_key":"same","revision":1,"text":"Source"}
            """))
        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"same","revision":1,
             "translated_full_text":"1234567"}
            """))
        XCTAssertNil(state.draftTranslatedText)

        state.apply(try event("""
            {"event_type":"translation_partial","source_key":"same","revision":1,
             "translated_full_text":"12345678"}
            """))
        XCTAssertEqual(state.draftTranslatedText, "12345678")

        let shortState = OverlayState()
        shortState.apply(try event("""
            {"event_type":"translation_partial","source_key":"short","revision":1,
             "translated_full_text":"好。","partial_complete":true}
            """))
        XCTAssertEqual(shortState.draftTranslatedText, "好。")
    }

    func testNemotronSettingsOnlyShowForExplicitNemotronSlot() {
        XCTAssertFalse(ModelPane.usesNemotron(nonChineseASR: ""))
        XCTAssertTrue(ModelPane.usesNemotron(
            nonChineseASR: "mlx-community/nemotron-3.5-asr-streaming-0.6b"))
        XCTAssertFalse(ModelPane.usesNemotron(
            nonChineseASR: "mlx-community/Qwen3-ASR-0.6B-4bit"))
    }
}
