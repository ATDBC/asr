from funasr_transcriber.punc_scheduler import (
    PuncRecord,
    assemble_punc_items,
    compact_aed_record,
    pack_punc_records,
)


def _record(index: int, token_count: int) -> PuncRecord:
    return PuncRecord(
        index=index,
        uttid=f"job-{index}",
        timestamp=tuple(("字", 0.0, 0.1) for _ in range(token_count)),
        confidence=0.9,
    )


def test_compact_aed_record_keeps_only_punc_inputs():
    compact = compact_aed_record(
        {
            "index": 7,
            "result": {
                "uttid": "job_s0_e1000",
                "text": "你好",
                "confidence": 0.8,
                "timestamp": [("你", 0.0, 0.2), ("好", 0.2, 0.4)],
                "unused": "discarded",
            },
        }
    )

    assert compact == PuncRecord(
        index=7,
        uttid="job_s0_e1000",
        timestamp=(("你", 0.0, 0.2), ("好", 0.2, 0.4)),
        confidence=0.8,
    )


def test_pack_punc_records_is_deterministic_and_token_bounded():
    records = [_record(0, 3), _record(1, 2), _record(2, 6), _record(3, 1)]

    batches = pack_punc_records(records, token_budget=6, max_items=3)

    assert [[record.index for record in batch] for batch in batches] == [
        [0, 1],
        [2],
        [3],
    ]
    assert all(
        sum(len(record.timestamp) for record in batch) <= 6
        for batch in batches
    )


def test_pack_punc_records_allows_a_single_oversized_record():
    records = [_record(4, 20)]

    batches = pack_punc_records(records, token_budget=6, max_items=3)

    assert [[record.index for record in batch] for batch in batches] == [[4]]


def test_assemble_punc_items_restores_original_index_order():
    raw = assemble_punc_items(
        "job",
        [
            {
                "index": 1,
                "uttid": "job_s1000_e2000",
                "confidence": 0.8,
                "timestamp": [["乙", 0.0, 1.0]],
                "punc_sentences": [
                    {"punc_text": "乙。", "start_s": 0.0, "end_s": 1.0}
                ],
            },
            {
                "index": 0,
                "uttid": "job_s0_e1000",
                "confidence": 0.9,
                "timestamp": [["甲", 0.0, 1.0]],
                "punc_sentences": [
                    {"punc_text": "甲。", "start_s": 0.0, "end_s": 1.0}
                ],
            },
        ],
        duration_ms=2000,
        vad_segments_ms=[[0, 1000], [1000, 2000]],
    )

    assert [item["text"] for item in raw["sentences"]] == ["甲。", "乙。"]
    assert [item["start_ms"] for item in raw["sentences"]] == [0, 1000]
    assert raw["text"] == "甲。乙。"
