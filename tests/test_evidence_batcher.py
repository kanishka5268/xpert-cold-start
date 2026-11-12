"""Unit tests for the evidence batching module."""

import copy
from typing import Any, Dict, List, Optional

import pytest

from src.ingestion.loader import load_dataset
from src.interpretation import (
    EvidenceBatch,
    EvidenceBatcher,
    build_evidence_batches,
)
from src.preprocessing.context_builder import build_conversation_context


class MockContext:
    """Minimal mock for ConversationContext."""

    def __init__(self, chats: Dict[str, List[Dict[str, Any]]]):
        self.chats = {
            cid: {"id": cid, "messages": msgs}
            for cid, msgs in chats.items()
        }

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        return self.chats.get(chat_id)


@pytest.fixture
def synthetic_context():
    """Synthetic multi-chat conversation context with 20 messages per chat."""
    c1_messages = [
        {"id": f"C1_M{i:02d}", "chat_id": "C1", "sender_id": "P1", "ts": 1000 + i * 60, "body": f"C1 message {i}"}
        for i in range(20)
    ]
    c2_messages = [
        {"id": f"C2_M{i:02d}", "chat_id": "C2", "sender_id": "P2", "ts": 2000 + i * 60, "body": f"C2 message {i}"}
        for i in range(20)
    ]
    return MockContext({"C1": c1_messages, "C2": c2_messages})


def test_empty_candidate_input(synthetic_context):
    """Verify that empty candidate input returns an empty list of batches."""
    batcher = EvidenceBatcher()
    batches = batcher.batch(synthetic_context, [])
    assert batches == []

    batches_none = build_evidence_batches(synthetic_context, [])
    assert batches_none == []


def test_candidates_from_multiple_chats(synthetic_context):
    """Verify that candidates from different chats are partitioned into separate batches."""
    candidates = [
        {"message_id": "C1_M05", "chat_id": "C1", "category": "commitment", "timestamp": 1300},
        {"message_id": "C2_M05", "chat_id": "C2", "category": "question", "timestamp": 2300},
    ]
    batches = build_evidence_batches(synthetic_context, candidates)

    assert len(batches) == 2
    chat_ids = [b.chat_id for b in batches]
    assert "C1" in chat_ids
    assert "C2" in chat_ids

    b1 = next(b for b in batches if b.chat_id == "C1")
    assert all(m["chat_id"] == "C1" for m in b1.messages)
    assert all(c["chat_id"] == "C1" for c in b1.candidates)

    b2 = next(b for b in batches if b.chat_id == "C2")
    assert all(m["chat_id"] == "C2" for m in b2.messages)
    assert all(c["chat_id"] == "C2" for c in b2.candidates)


def test_candidates_in_same_chat(synthetic_context):
    """Verify grouping of multiple candidates belonging to the same chat."""
    candidates = [
        {"message_id": "C1_M02", "chat_id": "C1", "category": "commitment", "timestamp": 1120},
        {"message_id": "C1_M03", "chat_id": "C1", "category": "deadline", "timestamp": 1180},
    ]
    batches = build_evidence_batches(synthetic_context, candidates, window_before=1, window_after=1)

    assert len(batches) == 1
    batch = batches[0]
    assert batch.chat_id == "C1"
    assert len(batch.candidates) == 2
    assert "C1_M02" in batch.candidate_message_ids
    assert "C1_M03" in batch.candidate_message_ids
    assert batch.categories == ["commitment", "deadline"]


def test_overlapping_context_windows_merged(synthetic_context):
    """Verify that overlapping context windows in the same chat are merged into a single batch."""
    # M03 with window [-2, +2] -> indices [1..5]
    # M05 with window [-2, +2] -> indices [3..7]
    # Indices 3, 4, 5 overlap!
    candidates = [
        {"message_id": "C1_M03", "chat_id": "C1", "category": "question", "timestamp": 1180},
        {"message_id": "C1_M05", "chat_id": "C1", "category": "commitment", "timestamp": 1300},
    ]
    batches = build_evidence_batches(synthetic_context, candidates, window_before=2, window_after=2)

    assert len(batches) == 1
    batch = batches[0]
    # Merged window from index 1 to index 7 -> 7 messages total
    expected_ids = [f"C1_M{i:02d}" for i in range(1, 8)]
    assert [m["id"] for m in batch.messages] == expected_ids
    assert len(batch.candidates) == 2


def test_non_overlapping_context_windows_remain_separate(synthetic_context):
    """Verify that distant candidates with non-overlapping context windows remain separate batches."""
    # M02 with window [-1, +1] -> indices [1..3]
    # M15 with window [-1, +1] -> indices [14..16]
    # No overlap!
    candidates = [
        {"message_id": "C1_M02", "chat_id": "C1", "category": "question", "timestamp": 1120},
        {"message_id": "C1_M15", "chat_id": "C1", "category": "followup", "timestamp": 1900},
    ]
    batches = build_evidence_batches(synthetic_context, candidates, window_before=1, window_after=1)

    assert len(batches) == 2
    b1, b2 = batches[0], batches[1]

    assert [m["id"] for m in b1.messages] == ["C1_M01", "C1_M02", "C1_M03"]
    assert b1.candidate_message_ids == ["C1_M02"]

    assert [m["id"] for m in b2.messages] == ["C1_M14", "C1_M15", "C1_M16"]
    assert b2.candidate_message_ids == ["C1_M15"]


def test_chronological_ordering(synthetic_context):
    """Verify that context messages and batches strictly preserve chronological ordering."""
    candidates = [
        {"message_id": "C1_M10", "chat_id": "C1", "category": "commitment", "timestamp": 1600},
        {"message_id": "C1_M02", "chat_id": "C1", "category": "question", "timestamp": 1120},
        {"message_id": "C1_M06", "chat_id": "C1", "category": "deadline", "timestamp": 1360},
    ]
    batches = build_evidence_batches(synthetic_context, candidates, window_before=1, window_after=1)

    # All messages in each batch must have non-decreasing timestamps
    for batch in batches:
        for i in range(len(batch.messages) - 1):
            assert batch.messages[i]["ts"] <= batch.messages[i + 1]["ts"]

    # Across separate batches in the same chat, batch start timestamps must increase
    for i in range(len(batches) - 1):
        assert batches[i].messages[0]["ts"] < batches[i + 1].messages[0]["ts"]


def test_no_duplicate_context_messages(synthetic_context):
    """Verify that merged batches never contain duplicate messages."""
    candidates = [
        {"message_id": "C1_M04", "chat_id": "C1", "category": "question", "timestamp": 1240},
        {"message_id": "C1_M05", "chat_id": "C1", "category": "commitment", "timestamp": 1300},
        {"message_id": "C1_M06", "chat_id": "C1", "category": "deadline", "timestamp": 1360},
    ]
    batches = build_evidence_batches(synthetic_context, candidates, window_before=2, window_after=2)

    assert len(batches) == 1
    batch = batches[0]
    msg_ids = [m["id"] for m in batch.messages]
    assert len(msg_ids) == len(set(msg_ids)), "Duplicate messages found in merged context window"


def test_candidate_to_message_association(synthetic_context):
    """Verify that every candidate in a batch correctly associates with context messages."""
    candidates = [
        {"message_id": "C1_M03", "chat_id": "C1", "category": "commitment", "timestamp": 1180},
        {"message_id": "C1_M03", "chat_id": "C1", "category": "deadline", "timestamp": 1180},
    ]
    batches = build_evidence_batches(synthetic_context, candidates, window_before=1, window_after=1)

    assert len(batches) == 1
    batch = batches[0]
    assert batch.candidate_message_ids == ["C1_M03"]
    assert len(batch.candidates) == 2
    assert set(batch.categories) == {"commitment", "deadline"}

    # Source message exists in batch.messages
    assert any(m["id"] == "C1_M03" for m in batch.messages)


def test_candidate_source_message_always_included(synthetic_context):
    """Verify that every candidate's own source message is strictly present in its batch."""
    # Test boundary candidate at the very start (index 0) and very end (index 19)
    candidates = [
        {"message_id": "C1_M00", "chat_id": "C1", "category": "question", "timestamp": 1000},
        {"message_id": "C1_M19", "chat_id": "C1", "category": "commitment", "timestamp": 2140},
    ]
    batches = build_evidence_batches(synthetic_context, candidates, window_before=3, window_after=3)

    assert len(batches) == 2
    b1 = next(b for b in batches if "C1_M00" in b.candidate_message_ids)
    assert any(m["id"] == "C1_M00" for m in b1.messages)

    b2 = next(b for b in batches if "C1_M19" in b.candidate_message_ids)
    assert any(m["id"] == "C1_M19" for m in b2.messages)


def test_maximum_candidate_limit(synthetic_context):
    """Verify that batches split when candidate count reaches max_candidates_per_batch."""
    # 5 contiguous candidates in C1
    candidates = [
        {"message_id": f"C1_M{i:02d}", "chat_id": "C1", "category": "commitment", "timestamp": 1000 + i * 60}
        for i in range(5)
    ]
    # Limit max_candidates_per_batch to 2
    batches = build_evidence_batches(
        synthetic_context, candidates, window_before=2, window_after=2, max_candidates_per_batch=2
    )

    # 5 candidates with max 2 per batch -> ceil(5 / 2) = 3 batches
    assert len(batches) == 3
    for b in batches:
        assert len(b.candidates) <= 2
        for c in b.candidates:
            assert any(m["id"] == c["message_id"] for m in b.messages)


def test_maximum_message_limit(synthetic_context):
    """Verify that batches split when merged context messages exceed max_messages_per_batch."""
    # Candidates at M01, M03, M05
    # If merged together: window [-1, +1] -> M00..M06 (7 messages)
    candidates = [
        {"message_id": "C1_M01", "chat_id": "C1", "category": "question", "timestamp": 1060},
        {"message_id": "C1_M03", "chat_id": "C1", "category": "commitment", "timestamp": 1180},
        {"message_id": "C1_M05", "chat_id": "C1", "category": "deadline", "timestamp": 1300},
    ]
    # Set max_messages_per_batch to 4:
    # M01 window: M00..M02 (3 msgs, fits)
    # Merging M03 would make M00..M04 (5 msgs > 4) -> must split!
    batches = build_evidence_batches(
        synthetic_context, candidates, window_before=1, window_after=1, max_messages_per_batch=4
    )

    assert len(batches) >= 2
    for b in batches:
        assert b.message_count <= 4
        for c in b.candidates:
            assert any(m["id"] == c["message_id"] for m in b.messages)


def test_oversized_context_window_handling(synthetic_context):
    """Verify explicit handling when a single candidate window exceeds max_messages_per_batch."""
    # Candidate M05 with window [-4, +4] spans 9 messages (M01..M09)
    candidate = [{"message_id": "C1_M05", "chat_id": "C1", "category": "goal", "timestamp": 1300}]

    # Strategy 1 (default: keep_intact): preserves all 9 messages and sets is_oversized=True
    batches_intact = build_evidence_batches(
        synthetic_context, candidate, window_before=4, window_after=4, max_messages_per_batch=5, oversized_strategy="keep_intact"
    )
    assert len(batches_intact) == 1
    b_intact = batches_intact[0]
    assert b_intact.is_oversized is True
    assert b_intact.message_count == 9
    assert any(m["id"] == "C1_M05" for m in b_intact.messages)

    # Strategy 2 (trim_to_limit): trims to max_messages_per_batch while guaranteeing candidate message is present
    batches_trim = build_evidence_batches(
        synthetic_context, candidate, window_before=4, window_after=4, max_messages_per_batch=5, oversized_strategy="trim_to_limit"
    )
    assert len(batches_trim) == 1
    b_trim = batches_trim[0]
    assert b_trim.is_oversized is True
    assert b_trim.message_count == 5
    assert any(m["id"] == "C1_M05" for m in b_trim.messages)


def test_no_cross_chat_mixing(synthetic_context):
    """Verify that messages and candidates from different chats are never intermixed."""
    candidates = [
        {"message_id": "C1_M01", "chat_id": "C1", "category": "commitment", "timestamp": 1060},
        {"message_id": "C2_M02", "chat_id": "C2", "category": "question", "timestamp": 1120},
        {"message_id": "C1_M03", "chat_id": "C1", "category": "deadline", "timestamp": 1180},
        {"message_id": "C2_M04", "chat_id": "C2", "category": "followup", "timestamp": 1240},
    ]
    batches = build_evidence_batches(synthetic_context, candidates)

    for batch in batches:
        # All messages in batch match batch.chat_id
        for m in batch.messages:
            assert m["chat_id"] == batch.chat_id, f"Cross-chat message found in batch {batch.chat_id}"
        # All candidates in batch match batch.chat_id
        for c in batch.candidates:
            assert c["chat_id"] == batch.chat_id, f"Cross-chat candidate found in batch {batch.chat_id}"


def test_deterministic_output(synthetic_context):
    """Verify that identical input produces strictly identical batch output and ordering."""
    candidates = [
        {"message_id": "C2_M08", "chat_id": "C2", "category": "commitment", "timestamp": 2480},
        {"message_id": "C1_M03", "chat_id": "C1", "category": "question", "timestamp": 1180},
        {"message_id": "C1_M05", "chat_id": "C1", "category": "deadline", "timestamp": 1300},
        {"message_id": "C2_M02", "chat_id": "C2", "category": "followup", "timestamp": 2120},
    ]

    batcher = EvidenceBatcher(window_before=2, window_after=2)
    run1 = batcher.batch(synthetic_context, candidates)
    run2 = batcher.batch(synthetic_context, candidates)

    assert len(run1) == len(run2)
    for b1, b2 in zip(run1, run2):
        assert b1.to_dict() == b2.to_dict()


def test_source_messages_and_dataset_immutability(synthetic_context):
    """Verify that source dataset and candidate records are not mutated during batching."""
    candidates = [
        {"message_id": "C1_M02", "chat_id": "C1", "category": "commitment", "timestamp": 1120},
        {"message_id": "C1_M04", "chat_id": "C1", "category": "question", "timestamp": 1240},
    ]
    candidates_copy = copy.deepcopy(candidates)
    context_copy = copy.deepcopy(synthetic_context.chats)

    batcher = EvidenceBatcher()
    _ = batcher.batch(synthetic_context, candidates)

    assert candidates == candidates_copy
    assert synthetic_context.chats == context_copy


def test_mutating_returned_batch_does_not_mutate_original_data(synthetic_context):
    """Verify that mutating returned EvidenceBatch objects does not affect the input dataset or candidates."""
    candidates = [
        {"message_id": "C1_M02", "chat_id": "C1", "category": "commitment", "timestamp": 1120},
        {"message_id": "C1_M04", "chat_id": "C1", "category": "question", "timestamp": 1240},
    ]
    batcher = EvidenceBatcher()
    batches = batcher.batch(synthetic_context, candidates)

    assert len(batches) > 0
    first_batch = batches[0]

    # Mutate a message inside the batch
    first_batch.messages[0]["body"] = "MUTATED BODY CONTENT"
    first_batch.messages[0]["extra_field"] = "ADDED IN BATCH"

    # Mutate a candidate inside the batch
    first_batch.candidates[0]["category"] = "MUTATED CATEGORY"

    # Verify original context message remains unchanged
    c1_msgs = synthetic_context.get_chat("C1")["messages"]
    assert c1_msgs[0]["body"] != "MUTATED BODY CONTENT"
    assert "extra_field" not in c1_msgs[0]

    # Verify original candidate list remains unchanged
    assert candidates[0]["category"] == "commitment"


def test_missing_candidate_source_message(synthetic_context):
    """Verify that a candidate referencing a nonexistent message ID raises ValueError."""
    candidate = [{"message_id": "NONEXISTENT_MSG", "chat_id": "C1", "category": "commitment", "timestamp": 1000}]
    with pytest.raises(ValueError, match="not found in conversation context"):
        build_evidence_batches(synthetic_context, candidate)


def test_candidate_source_message_different_chat(synthetic_context):
    """Verify that a candidate specifying chat C1 for a message in C2 raises ValueError."""
    candidate = [{"message_id": "C2_M05", "chat_id": "C1", "category": "commitment", "timestamp": 1000}]
    with pytest.raises(ValueError, match="belongs to chat 'C2', but candidate specifies chat 'C1'"):
        build_evidence_batches(synthetic_context, candidate)


def test_missing_or_empty_chat_id(synthetic_context):
    """Verify that candidates with missing or empty chat_id raise ValueError."""
    with pytest.raises(ValueError, match="missing valid non-empty 'chat_id'"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01", "chat_id": "", "category": "commitment"}])

    with pytest.raises(ValueError, match="missing valid non-empty 'chat_id'"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01", "chat_id": None, "category": "commitment"}])


def test_whitespace_in_chat_id(synthetic_context):
    """Verify that candidates with leading or trailing whitespace in chat_id raise ValueError."""
    # Leading whitespace
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01", "chat_id": " C1", "category": "commitment"}])

    # Trailing whitespace
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01", "chat_id": "C1 ", "category": "commitment"}])


def test_missing_or_empty_message_id(synthetic_context):
    """Verify that candidates with missing or empty message_id raise ValueError."""
    with pytest.raises(ValueError, match="missing valid non-empty 'message_id'"):
        build_evidence_batches(synthetic_context, [{"message_id": "", "chat_id": "C1", "category": "commitment"}])

    with pytest.raises(ValueError, match="missing valid non-empty 'message_id'"):
        build_evidence_batches(synthetic_context, [{"message_id": None, "chat_id": "C1", "category": "commitment"}])


def test_whitespace_in_message_id(synthetic_context):
    """Verify that candidates with leading or trailing whitespace in message_id raise ValueError."""
    # Leading whitespace
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        build_evidence_batches(synthetic_context, [{"message_id": " C1_M01", "chat_id": "C1", "category": "commitment"}])

    # Trailing whitespace
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01 ", "chat_id": "C1", "category": "commitment"}])


def test_missing_or_invalid_category(synthetic_context):
    """Verify that candidates with missing or invalid category raise ValueError."""
    with pytest.raises(ValueError, match="missing valid non-empty 'category'"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01", "chat_id": "C1", "category": ""}])

    with pytest.raises(ValueError, match="missing valid non-empty 'category'"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01", "chat_id": "C1", "category": None}])

    with pytest.raises(ValueError, match="invalid category"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01", "chat_id": "C1", "category": "invalid_cat"}])


def test_whitespace_in_category(synthetic_context):
    """Verify that candidates with leading or trailing whitespace in category raise ValueError."""
    # Leading whitespace
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01", "chat_id": "C1", "category": " commitment"}])

    # Trailing whitespace
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        build_evidence_batches(synthetic_context, [{"message_id": "C1_M01", "chat_id": "C1", "category": "commitment "}])


def test_no_silent_candidate_loss(synthetic_context):
    """Verify that across various boundary limits, no candidates are ever silently dropped."""
    # 10 candidates in C1
    candidates = [
        {"message_id": f"C1_M{i:02d}", "chat_id": "C1", "category": "commitment", "timestamp": 1000 + i * 60}
        for i in range(10)
    ]
    for max_cands in [1, 2, 3, 5, 8]:
        for max_msgs in [5, 10, 15, 25]:
            batches = build_evidence_batches(
                synthetic_context,
                candidates,
                max_candidates_per_batch=max_cands,
                max_messages_per_batch=max_msgs,
            )
            total_batched = sum(len(b.candidates) for b in batches)
            assert total_batched == len(candidates), f"Lost candidates with max_cands={max_cands}, max_msgs={max_msgs}"


def test_batching_on_example_dataset():
    """Verify batching works seamlessly on the real example dataset with CandidateEvidenceExtractor."""
    from src.extraction.evidence_extractor import CandidateEvidenceExtractor

    ds = load_dataset("example")
    ctx = build_conversation_context(ds)
    extractor = CandidateEvidenceExtractor()
    candidates = extractor.extract(ctx)

    assert len(candidates) > 0

    batcher = EvidenceBatcher(window_before=2, window_after=4, max_candidates_per_batch=6, max_messages_per_batch=20)
    batches = batcher.batch(ctx, candidates)

    assert len(batches) > 0
    total_batched_candidates = sum(len(b.candidates) for b in batches)
    assert total_batched_candidates == len(candidates), "Candidate records were lost during batching"

    for b in batches:
        assert b.message_count > 0
        assert b.candidate_count > 0
        assert all(m["chat_id"] == b.chat_id for m in b.messages)
        for c in b.candidates:
            assert any(m["id"] == c["message_id"] for m in b.messages)


def test_batching_on_dev_dataset():
    """Verify batching works seamlessly on the full development dataset."""
    from src.extraction.evidence_extractor import CandidateEvidenceExtractor

    ds = load_dataset("data/dev")
    ctx = build_conversation_context(ds)
    extractor = CandidateEvidenceExtractor()
    candidates = extractor.extract(ctx)

    assert len(candidates) > 0

    batcher = EvidenceBatcher(window_before=2, window_after=4, max_candidates_per_batch=8, max_messages_per_batch=25)
    batches = batcher.batch(ctx, candidates)

    assert len(batches) > 0
    total_batched_candidates = sum(len(b.candidates) for b in batches)
    assert total_batched_candidates == len(candidates), "Candidate records were lost during batching"

    for b in batches:
        assert b.message_count > 0
        assert b.candidate_count > 0
        assert all(m["chat_id"] == b.chat_id for m in b.messages)
        for c in b.candidates:
            assert any(m["id"] == c["message_id"] for m in b.messages)


def test_invalid_configuration_parameters():
    """Verify that invalid batcher configurations raise ValueError."""
    with pytest.raises(ValueError, match="window_before"):
        EvidenceBatcher(window_before=-1)

    with pytest.raises(ValueError, match="window_after"):
        EvidenceBatcher(window_after=-1)

    with pytest.raises(ValueError, match="max_candidates_per_batch"):
        EvidenceBatcher(max_candidates_per_batch=0)

    with pytest.raises(ValueError, match="max_messages_per_batch"):
        EvidenceBatcher(max_messages_per_batch=0)

    with pytest.raises(ValueError, match="oversized_strategy"):
        EvidenceBatcher(oversized_strategy="invalid_mode")
