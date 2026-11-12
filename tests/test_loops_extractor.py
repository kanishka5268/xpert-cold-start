"""Comprehensive end-to-end unit tests for open-loop extraction pipeline and schema emission."""

import copy
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.interpretation.interpreter import BatchFailureDiagnostic, LLMConfigurationError
from src.interpretation.records import InterpretedOpenLoop
from src.loops import (
    OPEN_LOOP_CATEGORIES,
    CandidateOpenLoop,
    HelpAction,
    OpenLoopExtractionConfig,
    OpenLoopExtractionDiagnostics,
    OpenLoopExtractionResult,
    extract_open_loops,
    interpret_open_loop_evidence,
    rank_and_clamp_open_loops,
)
from src.preprocessing.context_builder import ConversationContext


# ==============================================================================
# Test Helpers and Mocks
# ==============================================================================


class MockEvidenceInterpreter:
    """Mock evidence interpreter for deterministic pipeline testing without network calls."""

    def __init__(
        self,
        chat_responses: Optional[Dict[str, Tuple[List[Any], List[Any], Optional[BatchFailureDiagnostic]]]] = None,
        default_loops: Optional[List[Any]] = None,
        raise_error: Optional[Exception] = None,
    ):
        self.chat_responses = chat_responses or {}
        self.default_loops = default_loops or []
        self.raise_error = raise_error
        self.batch_calls: List[Any] = []

    def interpret_chat_batch(
        self, batch: Any, context: Optional[ConversationContext] = None
    ) -> Tuple[List[Any], List[Any], Optional[BatchFailureDiagnostic]]:
        self.batch_calls.append(batch)
        if self.raise_error is not None:
            raise self.raise_error

        chat_id = getattr(batch, "chat_id", None)
        if chat_id and chat_id in self.chat_responses:
            return self.chat_responses[chat_id]

        return list(self.default_loops), [], None


def _create_test_context(
    chat_messages: Dict[str, List[Dict[str, Any]]],
    chats_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    participants_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    timezone_str: str = "Asia/Kolkata",
) -> ConversationContext:
    """Build a valid ConversationContext from per-chat messages and optional metadata."""
    all_msgs: List[Dict[str, Any]] = []
    messages_by_chat: Dict[str, List[Dict[str, Any]]] = {}
    messages_by_id: Dict[str, Dict[str, Any]] = {}
    chats_by_id: Dict[str, Dict[str, Any]] = {}

    chats_meta = chats_metadata or {}
    for cid, msgs in chat_messages.items():
        meta = chats_meta.get(cid, {})
        chats_by_id[cid] = {
            "id": cid,
            "name": meta.get("name", f"Chat {cid}"),
            "is_group": meta.get("is_group", False),
            "participants": meta.get("participants", ["ME", "P01"]),
            "messages": msgs,
        }
        messages_by_chat[cid] = msgs
        for m in msgs:
            all_msgs.append(m)
            messages_by_id[m["id"]] = m

    all_msgs.sort(key=lambda m: m.get("ts", 0) if isinstance(m.get("ts"), (int, float)) else 0)

    participants = participants_metadata or {
        "ME": {"name": "Arjun", "is_silent": False, "sender_names": ["Arjun", "Me"], "direct_chat_names": ["Arjun"], "names": ["Arjun"]},
        "P01": {"name": "Neha", "is_silent": False, "sender_names": ["Neha"], "direct_chat_names": ["Neha"], "names": ["Neha"]},
        "P02": {"name": "Rahul", "is_silent": False, "sender_names": ["Rahul"], "direct_chat_names": ["Rahul"], "names": ["Rahul"]},
    }

    dataset: Dict[str, Any] = {
        "chats": list(chats_by_id.values()),
        "chats_by_id": chats_by_id,
        "messages": all_msgs,
        "messages_by_chat": messages_by_chat,
        "messages_by_id": messages_by_id,
        "timezone": timezone_str,
        "participants": participants,
    }

    return ConversationContext(dataset=dataset, chats=chats_by_id, participants=participants)


def _make_interpreted_loop(
    source_msg: str,
    chat_id: str,
    title: str,
    type: str = "commitment",
    owner: str = "ME",
    target_person: Optional[str] = "P01",
    status: str = "open",
    closed_by: Optional[str] = None,
    due_date: Optional[str] = None,
    suggested_action: Optional[str] = "draft proposal",
    confidence: Optional[float] = 0.9,
    due_anchor_msg: Optional[str] = None,
    is_genuine: bool = True,
) -> InterpretedOpenLoop:
    """Convenience factory for InterpretedOpenLoop with all contract arguments."""
    return InterpretedOpenLoop(
        source_msg=source_msg,
        chat_id=chat_id,
        type=type,
        is_genuine=is_genuine,
        title=title,
        owner=owner,
        target_person=target_person,
        status=status,
        closed_by=closed_by,
        due_date=due_date,
        suggested_action=suggested_action,
        confidence=confidence,
        validation_flags=None,
        due_anchor_msg=due_anchor_msg,
    )


# ==============================================================================
# 1. Empty Dataset and No-Candidate Scenarios
# ==============================================================================


def test_extract_open_loops_none_context():
    """Verify extract_open_loops handles context=None safely without errors."""
    mock = MockEvidenceInterpreter()
    diag = OpenLoopExtractionDiagnostics()
    result = extract_open_loops(None, mock, diagnostics=diag)  # type: ignore[arg-type]

    assert isinstance(result, OpenLoopExtractionResult)
    assert result.open_loops == []
    assert result.loop_count == 0
    assert len(mock.batch_calls) == 0


def test_extract_open_loops_no_candidates():
    """Verify conversation without open-loop evidence bypasses LLM interpretation."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Hello there", "kind": "text"},
            {"id": "M02", "chat_id": "C01", "sender_id": "P01", "ts": 110, "body": "Hi, all good!", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)
    mock = MockEvidenceInterpreter()
    diag = OpenLoopExtractionDiagnostics()

    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.open_loops == []
    assert result.loop_count == 0
    assert len(mock.batch_calls) == 0
    assert diag.candidates_received == 0
    assert diag.batches_created == 0


# ==============================================================================
# 2. First-Person Commitment (1:1 Chat)
# ==============================================================================


def test_extract_open_loops_first_person_commitment_1on1():
    """Verify valid 1:1 first-person commitment from ME creates schema-compliant open loop."""
    chat_msgs = {
        "C01": [
            {
                "id": "M01",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will draft the revised proposal by tomorrow",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C01",
        title="Draft revised proposal",
        type="commitment",
        owner="ME",
        suggested_action="draft the proposal",
        confidence=0.9,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    diag = OpenLoopExtractionDiagnostics()

    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["title"] == "Draft revised proposal"
    assert loop["type"] == "commitment"
    assert loop["owner"] == "ME"
    assert loop["chat_id"] == "C01"
    assert loop["source_msg"] == "M01"
    assert loop["status"] == "open"
    assert loop["closed_by"] is None
    assert isinstance(loop["help"], dict)
    assert loop["help"]["action"] == "assist"
    assert diag.batches_succeeded == 1
    assert diag.final_loops_emitted == 1


# ==============================================================================
# 3. Direct-Chat Request Assigned to ME
# ==============================================================================


def test_extract_open_loops_direct_chat_request_to_me():
    """Verify contact request in 1:1 chat assigns ownership to ME with valid open status."""
    chat_msgs = {
        "C01": [
            {
                "id": "M01",
                "chat_id": "C01",
                "sender_id": "P01",
                "ts": 1700000000,
                "body": "Can you review the client contract please?",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C01",
        title="Review client contract",
        type="followup",
        owner="ME",
        target_person="ME",
        suggested_action="review the contract document",
        confidence=0.88,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    result = extract_open_loops(context, mock)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["owner"] == "ME"
    assert loop["status"] == "open"
    assert loop["help"]["action"] in {"assist", "remind"}


# ==============================================================================
# 4. Group-Chat First-Person Commitment
# ==============================================================================


def test_extract_open_loops_group_chat_first_person_commitment():
    """Verify group-chat first-person commitment is retained and assigned to sender."""
    chat_msgs = {
        "C02": [
            {
                "id": "M10",
                "chat_id": "C02",
                "sender_id": "P02",
                "ts": 1700000000,
                "body": "I will prepare the presentation slides for the summit",
                "kind": "text",
            }
        ]
    }
    chats_meta = {
        "C02": {"name": "Summit Planning", "is_group": True, "participants": ["ME", "P01", "P02"]}
    }
    context = _create_test_context(chat_msgs, chats_metadata=chats_meta)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M10",
        chat_id="C02",
        title="Prepare presentation slides",
        type="commitment",
        owner="P02",
        target_person="ME",
        suggested_action=None,
        confidence=0.85,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    result = extract_open_loops(context, mock)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["owner"] == "P02"
    assert loop["status"] == "open"
    # Contact-owned loop gets leave action with grounded contact ownership reason
    assert loop["help"]["action"] == "leave"
    assert "P02" in loop["help"]["reason"]


# ==============================================================================
# 5. Group-Chat Directed Assignment
# ==============================================================================


def test_extract_open_loops_group_chat_directed_assignment():
    """Verify group-chat directed assignment to grounded contact is retained."""
    chat_msgs = {
        "C02": [
            {
                "id": "M11",
                "chat_id": "C02",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "Rahul please check the financial sheet",
                "kind": "text",
            }
        ]
    }
    chats_meta = {
        "C02": {"name": "Finance Team", "is_group": True, "participants": ["ME", "P01", "P02"]}
    }
    context = _create_test_context(chat_msgs, chats_metadata=chats_meta)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M11",
        chat_id="C02",
        title="Check financial sheet",
        type="followup",
        owner="P02",
        target_person="P02",
        suggested_action=None,
        confidence=0.85,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    result = extract_open_loops(context, mock)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["owner"] == "P02"
    assert loop["help"]["action"] == "leave"


# ==============================================================================
# 6. Actionable Unanswered Question
# ==============================================================================


def test_extract_open_loops_actionable_unanswered_question():
    """Verify unanswered actionable question is extracted with type=question."""
    chat_msgs = {
        "C01": [
            {
                "id": "M20",
                "chat_id": "C01",
                "sender_id": "P01",
                "ts": 1700000000,
                "body": "Can you check where should we host the offsite meeting?",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M20",
        chat_id="C01",
        title="Where should we host the offsite meeting?",
        type="question",
        owner="ME",
        suggested_action="research venues for the offsite meeting",
        confidence=0.8,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    result = extract_open_loops(context, mock)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["type"] == "question"
    assert loop["status"] == "open"


# ==============================================================================
# 7. Follow-up or Deadline with Grounded Due Date
# ==============================================================================


def test_extract_open_loops_deadline_grounded_due():
    """Verify deadline loop retains grounded ISO calendar date YYYY-MM-DD."""
    chat_msgs = {
        "C01": [
            {
                "id": "M30",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will submit the tax return by 2025-10-31",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M30",
        chat_id="C01",
        title="Submit tax return",
        type="deadline",
        owner="ME",
        due_date="2025-10-31",
        due_anchor_msg="M30",
        suggested_action="finalize and submit the tax return",
        confidence=0.9,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    result = extract_open_loops(context, mock)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["due"] == "2025-10-31"
    assert loop["type"] == "deadline"


# ==============================================================================
# 8. Completed Obligation (status="done")
# ==============================================================================


def test_extract_open_loops_completed_obligation():
    """Verify actor-authorized completion marks status=done with valid closed_by and help=None."""
    chat_msgs = {
        "C01": [
            {
                "id": "M40",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will send the pitch deck today",
                "kind": "text",
            },
            {
                "id": "M41",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700003600,
                "body": "Done, sent the pitch deck to your inbox",
                "kind": "text",
            },
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M40",
        chat_id="C01",
        title="Send pitch deck",
        type="commitment",
        owner="ME",
        status="done",
        closed_by="M41",
        confidence=0.9,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    diag = OpenLoopExtractionDiagnostics()
    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["status"] == "done"
    assert loop["closed_by"] == "M41"
    assert loop["help"] is None
    assert diag.loops_marked_done == 1


# ==============================================================================
# 9. Explicitly Cancelled Obligation (status="dropped")
# ==============================================================================


def test_extract_open_loops_cancelled_obligation():
    """Verify requester revocation marks status=dropped with valid closed_by and help=None."""
    chat_msgs = {
        "C01": [
            {
                "id": "M50",
                "chat_id": "C01",
                "sender_id": "P01",
                "ts": 1700000000,
                "body": "Can you book tickets for the conference?",
                "kind": "text",
            },
            {
                "id": "M51",
                "chat_id": "C01",
                "sender_id": "P01",
                "ts": 1700003600,
                "body": "Please cancel that conference ticket request, the event was postponed",
                "kind": "text",
            },
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M50",
        chat_id="C01",
        title="Book conference tickets",
        type="followup",
        owner="ME",
        status="dropped",
        closed_by="M51",
        confidence=0.88,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    diag = OpenLoopExtractionDiagnostics()
    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["status"] == "dropped"
    assert loop["closed_by"] == "M51"
    assert loop["help"] is None
    assert diag.loops_marked_dropped == 1


# ==============================================================================
# 10. Unrelated Completion Message Non-Closure
# ==============================================================================


def test_extract_open_loops_unrelated_completion_remains_open():
    """Verify unrelated completion message does not close the obligation."""
    chat_msgs = {
        "C01": [
            {
                "id": "M60",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will review the pitch deck",
                "kind": "text",
            },
            {
                "id": "M61",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700003600,
                "body": "Done with lunch, back at my desk",
                "kind": "text",
            },
        ]
    }
    context = _create_test_context(chat_msgs)
    # Even if LLM claimed closed_by="M61", resolver rejects unrelated linkage
    interpreted_loop = _make_interpreted_loop(
        source_msg="M60",
        chat_id="C01",
        title="Review pitch deck",
        type="commitment",
        owner="ME",
        status="done",
        closed_by="M61",
        suggested_action="review the pitch deck",
        confidence=0.85,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    result = extract_open_loops(context, mock)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["status"] == "open"
    assert loop["closed_by"] is None
    assert isinstance(loop["help"], dict)


# ==============================================================================
# 11. Invalid or Missing Source References
# ==============================================================================


def test_extract_open_loops_invalid_source_references():
    """Verify candidate with nonexistent source_msg or chat_id is discarded."""
    chat_msgs = {
        "C01": [
            {
                "id": "M70",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will call the architect tomorrow",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    bad_source = _make_interpreted_loop(
        source_msg="NONEXISTENT_MSG",
        chat_id="C01",
        title="Call architect",
        type="commitment",
        owner="ME",
        confidence=0.9,
    )
    bad_chat = _make_interpreted_loop(
        source_msg="M70",
        chat_id="NONEXISTENT_CHAT",
        title="Call architect",
        type="commitment",
        owner="ME",
        confidence=0.9,
    )
    mock = MockEvidenceInterpreter(default_loops=[bad_source, bad_chat])
    diag = OpenLoopExtractionDiagnostics()
    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.loop_count == 0
    assert diag.candidates_rejected >= 2


# ==============================================================================
# 12. Invalid Loop Types Discarded
# ==============================================================================


def test_extract_open_loops_invalid_loop_type_discarded():
    """Verify invalid or unsupported loop type is rejected rather than coerced."""
    chat_msgs = {
        "C01": [
            {
                "id": "M80",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will handle this payment",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    invalid_type_loop = _make_interpreted_loop(
        source_msg="M80",
        chat_id="C01",
        title="Handle payment",
        type="unsupported_loop_type",
        owner="ME",
        confidence=0.9,
    )
    mock = MockEvidenceInterpreter(default_loops=[invalid_type_loop])
    diag = OpenLoopExtractionDiagnostics()
    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.loop_count == 0
    assert diag.candidates_rejected >= 1


# ==============================================================================
# 13. Missing or Invalid Confidence
# ==============================================================================


def test_extract_open_loops_invalid_confidence_discarded():
    """Verify loops with confidence below min_confidence or non-numeric confidence are discarded."""
    chat_msgs = {
        "C01": [
            {
                "id": "M90",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will inspect the site",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    low_conf = _make_interpreted_loop(
        source_msg="M90",
        chat_id="C01",
        title="Inspect site",
        type="commitment",
        owner="ME",
        confidence=0.3,  # Below default min_confidence=0.5
    )
    none_conf = _make_interpreted_loop(
        source_msg="M90",
        chat_id="C01",
        title="Inspect site",
        type="commitment",
        owner="ME",
        confidence=None,  # Missing confidence rejected
    )
    mock = MockEvidenceInterpreter(default_loops=[low_conf, none_conf])
    diag = OpenLoopExtractionDiagnostics()
    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.loop_count == 0
    assert diag.candidates_rejected >= 2


# ==============================================================================
# 14. Invalid Due-Date Anchors and Unsupported Timezone Assumptions
# ==============================================================================


def test_extract_open_loops_ungrounded_due_date_becomes_none():
    """Verify ungrounded relative date without authoritative anchor sets due=None."""
    chat_msgs = {
        "C01": [
            {
                "id": "M100",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will deliver the report soon",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M100",
        chat_id="C01",
        title="Deliver report",
        type="commitment",
        owner="ME",
        due_date="2025-13-45",  # Invalid calendar date
        due_anchor_msg=None,
        suggested_action="draft the report",
        confidence=0.85,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    result = extract_open_loops(context, mock)

    assert result.loop_count == 1
    assert result.open_loops[0]["due"] is None


# ==============================================================================
# 15. Sensitive Help-Action Behavior
# ==============================================================================


def test_extract_open_loops_sensitive_topic_leave_action():
    """Verify sensitive personal / spouse boundary topic assigns help action=leave."""
    chat_msgs = {
        "C01": [
            {
                "id": "M110",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will buy an anniversary gift for my spouse",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M110",
        chat_id="C01",
        title="Anniversary gift for spouse",
        type="commitment",
        owner="ME",
        confidence=0.9,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    result = extract_open_loops(context, mock)

    assert result.loop_count == 1
    loop = result.open_loops[0]
    assert loop["help"]["action"] == "leave"
    assert "personal" in loop["help"]["reason"].lower() or "spouse" in loop["help"]["reason"].lower() or "boundary" in loop["help"]["reason"].lower()


# ==============================================================================
# 16. Deduplication and Recurring Episodes
# ==============================================================================


def test_extract_open_loops_deduplication_reminder_merges():
    """Verify reminder of open task merges and preserves original source_msg."""
    chat_msgs = {
        "C01": [
            {
                "id": "M120",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will send the contract by Friday",
                "kind": "text",
            },
            {
                "id": "M121",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700003600,
                "body": "Reminder: I will send the contract by Friday",
                "kind": "text",
            },
        ]
    }
    context = _create_test_context(chat_msgs)
    l1 = _make_interpreted_loop(
        source_msg="M120",
        chat_id="C01",
        title="Send contract",
        type="commitment",
        owner="ME",
        confidence=0.85,
    )
    l2 = _make_interpreted_loop(
        source_msg="M121",
        chat_id="C01",
        title="Send contract",
        owner="ME",
        confidence=0.85,
    )
    mock = MockEvidenceInterpreter(default_loops=[l1, l2])
    diag = OpenLoopExtractionDiagnostics()
    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.loop_count == 1
    assert result.open_loops[0]["source_msg"] == "M120"
    assert diag.loops_deduplicated == 1


# ==============================================================================
# 17. Uncapped Global Output & Deterministic Ranking (Milestone A)
# ==============================================================================


def test_extract_open_loops_retains_more_than_10_loops_by_default():
    """Verify that by default (max_open_loops=None), more than 10 valid loops are retained without global clamping."""
    task_templates = [
        ("Send invoice", "I will send the invoice"),
        ("Review contract", "I will review the contract"),
        ("Book flight tickets", "I will book flight tickets"),
        ("Call doctor", "I will call doctor"),
        ("Submit report", "I will submit the report"),
        ("Order groceries", "I will order groceries"),
        ("Fix faucet", "I will fix the faucet"),
        ("File tax return", "I will file the tax return"),
        ("Draft proposal", "I will draft the proposal"),
        ("Pay electricity bill", "I will pay the electricity bill"),
        ("Inspect site", "I will inspect the site"),
        ("Reserve dinner table", "I will reserve dinner table"),
        ("Update website", "I will update the website"),
        ("Deliver package", "I will deliver the package"),
        ("Clean garage", "I will clean the garage"),
    ]

    msgs = []
    loops = []
    for i, (title, body) in enumerate(task_templates):
        mid = f"M{200 + i}"
        msgs.append({
            "id": mid,
            "chat_id": "C01",
            "sender_id": "ME",
            "ts": 1700000000 + i * 10,
            "body": body,
            "kind": "text",
        })
        loops.append(
            _make_interpreted_loop(
                source_msg=mid,
                chat_id="C01",
                title=title,
                type="commitment",
                owner="ME",
                suggested_action=f"action for {title}",
                confidence=0.80 + (i * 0.01),
            )
        )

    chat_msgs = {"C01": msgs}
    context = _create_test_context(chat_msgs)

    class DynamicInterpreter:
        def __init__(self, all_loops):
            self.loop_map = {l.source_msg: l for l in all_loops}

        def interpret_chat_batch(self, batch, context=None):
            b_msgs = getattr(batch, "messages", [])
            b_ids = {m["id"] for m in b_msgs if isinstance(m, dict) and "id" in m}
            b_loops = [self.loop_map[mid] for mid in b_ids if mid in self.loop_map]
            return b_loops, [], None

    mock = DynamicInterpreter(loops)
    diag = OpenLoopExtractionDiagnostics()

    # 1. Default config: max_open_loops=None (unlimited)
    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.loop_count == 15
    assert len(result.open_loops) == 15
    assert diag.loops_truncated_by_cap == 0
    assert diag.final_loops_emitted == 15
    # Deterministic ranking preserved: highest confidence (0.94) first
    assert result.open_loops[0]["source_msg"] == "M214"

    # 2. Explicit custom cap: max_open_loops=10 clamps to 10
    diag_capped = OpenLoopExtractionDiagnostics()
    result_capped = extract_open_loops(
        context,
        mock,
        config=OpenLoopExtractionConfig(max_open_loops=10),
        diagnostics=diag_capped,
    )
    assert result_capped.loop_count == 10
    assert len(result_capped.open_loops) == 10
    assert diag_capped.loops_truncated_by_cap == 5
    assert diag_capped.final_loops_emitted == 10
    assert result_capped.open_loops[0]["source_msg"] == "M214"


def test_extract_open_loops_retains_exactly_10_when_10_exist():
    """Verify that exactly 10 valid open loops are retained when there are exactly 10."""
    templates = [
        ("Send invoice", "I will send the invoice"),
        ("Review contract", "I will review the contract"),
        ("Book flight tickets", "I will book flight tickets"),
        ("Call doctor", "I will call doctor"),
        ("Submit report", "I will submit the report"),
        ("Order groceries", "I will order groceries"),
        ("Fix faucet", "I will fix the faucet"),
        ("File tax return", "I will file the tax return"),
        ("Draft proposal", "I will draft the proposal"),
        ("Pay electricity bill", "I will pay the electricity bill"),
    ]
    msgs = [
        {"id": f"M_EX_{i}", "chat_id": "C01", "sender_id": "ME", "ts": 1700000000 + i * 10, "body": b, "kind": "text"}
        for i, (t, b) in enumerate(templates)
    ]
    loops = [
        _make_interpreted_loop(
            source_msg=f"M_EX_{i}",
            chat_id="C01",
            title=t,
            type="commitment",
            owner="ME",
            suggested_action=f"do {t}",
            confidence=0.85,
        )
        for i, (t, b) in enumerate(templates)
    ]
    context = _create_test_context({"C01": msgs})

    class DynamicInterpreter:
        def __init__(self, all_loops):
            self.loop_map = {l.source_msg: l for l in all_loops}

        def interpret_chat_batch(self, batch, context=None):
            b_msgs = getattr(batch, "messages", [])
            b_ids = {m["id"] for m in b_msgs if isinstance(m, dict) and "id" in m}
            b_loops = [self.loop_map[mid] for mid in b_ids if mid in self.loop_map]
            return b_loops, [], None

    diag = OpenLoopExtractionDiagnostics()
    result = extract_open_loops(context, DynamicInterpreter(loops), diagnostics=diag)

    assert result.loop_count == 10
    assert len(result.open_loops) == 10
    assert diag.loops_truncated_by_cap == 0
    assert diag.final_loops_emitted == 10


def test_extract_open_loops_large_sets_25_and_74_not_globally_truncated():
    """Verify large datasets (such as 25 and 74 valid loops matching gold dev count) are not globally truncated."""
    class DynamicInterpreter:
        def __init__(self, all_loops):
            self.loop_map = {l.source_msg: l for l in all_loops}

        def interpret_chat_batch(self, batch, context=None):
            b_msgs = getattr(batch, "messages", [])
            b_ids = {m["id"] for m in b_msgs if isinstance(m, dict) and "id" in m}
            b_loops = [self.loop_map[mid] for mid in b_ids if mid in self.loop_map]
            return b_loops, [], None

    # Test 74 loops across 74 distinct chats (matching gold dev count)
    chat_msgs74 = {}
    loops74 = []
    for i in range(74):
        cid = f"C{i:03d}"
        mid = f"M_GOLD_{i:03d}"
        chat_msgs74[cid] = [{
            "id": mid,
            "chat_id": cid,
            "sender_id": "ME",
            "ts": 1700000000 + i * 10,
            "body": f"I will deliver deliverable {i}",
            "kind": "text",
        }]
        loops74.append(
            _make_interpreted_loop(
                source_msg=mid,
                chat_id=cid,
                title=f"Deliver item {i}",
                type="commitment",
                owner="ME",
                suggested_action=f"deliver {i}",
                confidence=0.70 + (i % 30) * 0.01,
            )
        )

    context74 = _create_test_context(chat_msgs74)
    diag74 = OpenLoopExtractionDiagnostics()
    result74 = extract_open_loops(context74, DynamicInterpreter(loops74), diagnostics=diag74)

    assert result74.loop_count == 74
    assert len(result74.open_loops) == 74
    assert diag74.loops_truncated_by_cap == 0
    assert diag74.final_loops_emitted == 74

    # Also test 25 loops across 25 chats
    chat_msgs25 = {
        f"C{i:02d}": [{
            "id": f"M_25_{i}",
            "chat_id": f"C{i:02d}",
            "sender_id": "ME",
            "ts": 1700000000 + i * 10,
            "body": f"I will deliver item {i}",
            "kind": "text",
        }]
        for i in range(25)
    }
    loops25 = [
        _make_interpreted_loop(
            source_msg=f"M_25_{i}",
            chat_id=f"C{i:02d}",
            title=f"Deliver item {i}",
            type="commitment",
            owner="ME",
            suggested_action=f"action {i}",
            confidence=0.85,
        )
        for i in range(25)
    ]
    context25 = _create_test_context(chat_msgs25)
    diag25 = OpenLoopExtractionDiagnostics()
    result25 = extract_open_loops(context25, DynamicInterpreter(loops25), diagnostics=diag25)

    assert result25.loop_count == 25
    assert len(result25.open_loops) == 25
    assert diag25.loops_truncated_by_cap == 0
    assert diag25.final_loops_emitted == 25


def test_deterministic_ranking_stability_without_cap():
    """Verify deterministic ranking (-confidence, ts, source_msg) remains completely stable across uncapped results."""
    cands = [
        CandidateOpenLoop(
            title="Task A",
            type="commitment",
            owner="ME",
            chat_id="C01",
            source_msg="M01",
            confidence=0.80,
            status="open",
            help=HelpAction(action="assist", next_step="do step", needs_yes=True),
        ),
        CandidateOpenLoop(
            title="Task B",
            type="commitment",
            owner="ME",
            chat_id="C01",
            source_msg="M02",
            confidence=0.95,
            status="open",
            help=HelpAction(action="assist", next_step="do step", needs_yes=True),
        ),
        CandidateOpenLoop(
            title="Task C",
            type="commitment",
            owner="ME",
            chat_id="C01",
            source_msg="M03",
            confidence=0.95,
            status="open",
            help=HelpAction(action="assist", next_step="do step", needs_yes=True),
        ),
        CandidateOpenLoop(
            title="Task D",
            type="commitment",
            owner="ME",
            chat_id="C01",
            source_msg="M04",
            confidence=0.60,
            status="open",
            help=HelpAction(action="assist", next_step="do step", needs_yes=True),
        ),
    ]

    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "ts": 100},
            {"id": "M02", "chat_id": "C01", "ts": 200},  # Higher confidence (0.95), ts 200
            {"id": "M03", "chat_id": "C01", "ts": 150},  # Higher confidence (0.95), earlier ts 150 -> wins over M02!
            {"id": "M04", "chat_id": "C01", "ts": 50},
        ]
    }
    context = _create_test_context(chat_msgs)
    config = OpenLoopExtractionConfig()  # unlimited

    ranked = rank_and_clamp_open_loops(cands, context, config)

    assert len(ranked) == 4
    # Highest confidence (0.95): M03 has ts=150, M02 has ts=200 -> M03 first, then M02
    assert ranked[0].source_msg == "M03"
    assert ranked[1].source_msg == "M02"
    # Next confidence (0.80) -> M01
    assert ranked[2].source_msg == "M01"
    # Lowest confidence (0.60) -> M04
    assert ranked[3].source_msg == "M04"


def test_per_chat_limits_with_unlimited_global():
    """Verify max_loops_per_chat clamps per chat while global output remains unlimited (>10)."""
    class DynamicInterpreter:
        def __init__(self, all_loops):
            self.loop_map = {l.source_msg: l for l in all_loops}

        def interpret_chat_batch(self, batch, context=None):
            b_msgs = getattr(batch, "messages", [])
            b_ids = {m["id"] for m in b_msgs if isinstance(m, dict) and "id" in m}
            b_loops = [self.loop_map[mid] for mid in b_ids if mid in self.loop_map]
            return b_loops, [], None

    actions = ["Send invoice", "Review contract", "Fix faucet", "Submit report"]
    chat_msgs = {}
    loops = []
    for c_idx in range(5):
        cid = f"C{c_idx:02d}"
        chat_msgs[cid] = []
        for t_idx in range(4):
            mid = f"M_{cid}_{t_idx}"
            title = actions[t_idx]
            chat_msgs[cid].append({
                "id": mid,
                "chat_id": cid,
                "sender_id": "ME",
                "ts": 1700000000 + t_idx * 10,
                "body": f"I will {title.lower()}",
                "kind": "text",
            })
            loops.append(
                _make_interpreted_loop(
                    source_msg=mid,
                    chat_id=cid,
                    title=title,
                    type="commitment",
                    owner="ME",
                    suggested_action=f"do {title}",
                    confidence=0.85,
                )
            )

    context = _create_test_context(chat_msgs)
    config = OpenLoopExtractionConfig(max_loops_per_chat=3, max_open_loops=None)
    diag = OpenLoopExtractionDiagnostics()

    result = extract_open_loops(context, DynamicInterpreter(loops), config=config, diagnostics=diag)

    assert result.loop_count == 15
    assert len(result.open_loops) == 15
    assert diag.loops_truncated_by_cap == 5 * 1  # 1 discarded per chat
    assert diag.final_loops_emitted == 15


def test_max_loops_per_chat_none_is_unlimited():
    """Verify max_loops_per_chat=None imposes no chat-level limit."""
    class DynamicInterpreter:
        def __init__(self, all_loops):
            self.loop_map = {l.source_msg: l for l in all_loops}

        def interpret_chat_batch(self, batch, context=None):
            b_msgs = getattr(batch, "messages", [])
            b_ids = {m["id"] for m in b_msgs if isinstance(m, dict) and "id" in m}
            b_loops = [self.loop_map[mid] for mid in b_ids if mid in self.loop_map]
            return b_loops, [], None

    templates = [
        ("Send invoice", "I will send the invoice"),
        ("Review contract", "I will review the contract"),
        ("Book flight tickets", "I will book flight tickets"),
        ("Call doctor", "I will call doctor"),
        ("Submit report", "I will submit the report"),
        ("Order groceries", "I will order groceries"),
        ("Fix faucet", "I will fix the faucet"),
        ("File tax return", "I will file the tax return"),
        ("Draft proposal", "I will draft the proposal"),
        ("Pay electricity bill", "I will pay the electricity bill"),
        ("Inspect site", "I will inspect the site"),
        ("Clean garage", "I will clean the garage"),
    ]
    msgs = [
        {"id": f"M_NL_{i}", "chat_id": "C01", "sender_id": "ME", "ts": 1700000000 + i * 10, "body": b, "kind": "text"}
        for i, (t, b) in enumerate(templates)
    ]
    loops = [
        _make_interpreted_loop(
            source_msg=f"M_NL_{i}",
            chat_id="C01",
            title=t,
            type="commitment",
            owner="ME",
            suggested_action=f"do {t}",
            confidence=0.85,
        )
        for i, (t, b) in enumerate(templates)
    ]
    context = _create_test_context({"C01": msgs})
    config = OpenLoopExtractionConfig(max_loops_per_chat=None, max_open_loops=None)
    diag = OpenLoopExtractionDiagnostics()

    result = extract_open_loops(context, DynamicInterpreter(loops), config=config, diagnostics=diag)

    assert result.loop_count == 12
    assert diag.loops_truncated_by_cap == 0


def test_invalid_candidates_rejected_at_schema_barrier_without_cap():
    """Verify invalid CandidateOpenLoop records are rejected at schema barrier even with uncapped output."""
    valid_loops = [
        CandidateOpenLoop(
            title=f"Valid task {i}",
            type="commitment",
            owner="ME",
            chat_id="C01",
            source_msg=f"M_VAL_{i}",
            confidence=0.85,
            status="open",
            help=HelpAction(action="assist", next_step="do task", needs_yes=True),
        )
        for i in range(12)
    ]
    invalid_candidate = CandidateOpenLoop(
        title="",  # Empty title -> validate() returns False!
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M_INVAL",
        confidence=0.85,
        status="open",
        help=HelpAction(action="assist", next_step="do task", needs_yes=True),
    )

    all_cands = valid_loops + [invalid_candidate]
    chat_msgs = {
        "C01": [{"id": f"M_VAL_{i}", "chat_id": "C01", "ts": 100 + i} for i in range(12)]
        + [{"id": "M_INVAL", "chat_id": "C01", "ts": 200}]
    }
    context = _create_test_context(chat_msgs)
    config = OpenLoopExtractionConfig(max_open_loops=None)
    diag = OpenLoopExtractionDiagnostics()

    clamped = rank_and_clamp_open_loops(all_cands, context, config, diagnostics=diag)
    assert len(clamped) == 13

    # Barrier validation
    final_loops = []
    for cand in clamped:
        if cand.validate():
            final_loops.append(cand.to_schema_dict())
        else:
            diag.candidates_rejected += 1

    assert len(final_loops) == 12
    assert diag.candidates_rejected == 1


def test_every_output_contains_exact_nine_schema_fields_in_large_set():
    """Verify that every single emitted open loop across a large result set contains exactly the 9 required schema fields."""
    class DynamicInterpreter:
        def __init__(self, all_loops):
            self.loop_map = {l.source_msg: l for l in all_loops}

        def interpret_chat_batch(self, batch, context=None):
            b_msgs = getattr(batch, "messages", [])
            b_ids = {m["id"] for m in b_msgs if isinstance(m, dict) and "id" in m}
            b_loops = [self.loop_map[mid] for mid in b_ids if mid in self.loop_map]
            return b_loops, [], None

    chat_msgs = {}
    loops = []
    for i in range(25):
        cid = f"C{i:02d}"
        mid = f"M_SCH_{i}"
        chat_msgs[cid] = [{
            "id": mid,
            "chat_id": cid,
            "sender_id": "ME",
            "ts": 1700000000 + i * 10,
            "body": f"I will deliver item {i}",
            "kind": "text",
        }]
        loops.append(
            _make_interpreted_loop(
                source_msg=mid,
                chat_id=cid,
                title=f"Deliver item {i}",
                type="commitment",
                owner="ME",
                suggested_action=f"action {i}",
                confidence=0.85,
            )
        )

    context = _create_test_context(chat_msgs)
    result = extract_open_loops(context, DynamicInterpreter(loops))

    assert result.loop_count == 25
    expected_fields = {"title", "type", "owner", "chat_id", "source_msg", "due", "status", "closed_by", "help"}
    for loop in result.open_loops:
        assert set(loop.keys()) == expected_fields
        assert isinstance(loop["title"], str) and loop["title"]
        assert loop["type"] in OPEN_LOOP_CATEGORIES
        assert loop["owner"] in ("ME", "P01") or loop["owner"].startswith("P")
        assert loop["chat_id"].startswith("C")
        assert loop["source_msg"].startswith("M_SCH_")
        assert loop["status"] in ("open", "done", "dropped")


def test_goal_extraction_isolation_still_enforces_10():
    """Verify that Step 6 goal extraction remains strictly isolated and continues enforcing max_goals=10."""
    from src.goals.models import GoalExtractionConfig
    goal_config = GoalExtractionConfig()
    assert goal_config.max_goals == 10


def test_goal_extraction_isolation_still_enforces_10():
    """Verify that Step 6 goal extraction remains strictly isolated and continues enforcing max_goals=10."""
    from src.goals.models import GoalExtractionConfig
    goal_config = GoalExtractionConfig()
    assert goal_config.max_goals == 10


# ==============================================================================
# 18. Exact Nine-Field Schema Compliance
# ==============================================================================


def test_extract_open_loops_exact_nine_fields_schema():
    """Verify every emitted open loop contains exactly the nine official schema properties."""
    chat_msgs = {
        "C01": [
            {
                "id": "M300",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will deliver the draft agreement by Monday",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_loop = _make_interpreted_loop(
        source_msg="M300",
        chat_id="C01",
        title="Deliver draft agreement",
        type="commitment",
        owner="ME",
        suggested_action="draft the agreement document",
        confidence=0.9,
    )
    mock = MockEvidenceInterpreter(default_loops=[interpreted_loop])
    result = extract_open_loops(context, mock)

    assert result.loop_count == 1
    loop = result.open_loops[0]

    expected_keys = {
        "title",
        "type",
        "owner",
        "chat_id",
        "source_msg",
        "due",
        "status",
        "closed_by",
        "help",
    }
    assert set(loop.keys()) == expected_keys
    assert isinstance(loop["title"], str) and loop["title"]
    assert loop["type"] in OPEN_LOOP_CATEGORIES
    assert loop["owner"] == "ME"
    assert loop["chat_id"] == "C01"
    assert loop["source_msg"] == "M300"
    assert loop["due"] is None
    assert loop["status"] == "open"
    assert loop["closed_by"] is None
    assert isinstance(loop["help"], dict)
    assert loop["help"]["action"] in {"assist", "remind", "leave"}


# ==============================================================================
# 19. LLM Batch Failure and Retry Handling
# ==============================================================================


def test_extract_open_loops_batch_failure_and_continuation():
    """Verify non-fatal batch failure increments batches_failed and processing continues."""
    chat_msgs = {
        "C01": [
            {
                "id": "M401",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will send file one",
                "kind": "text",
            }
        ],
        "C02": [
            {
                "id": "M402",
                "chat_id": "C02",
                "sender_id": "ME",
                "ts": 1700001000,
                "body": "I will send file two",
                "kind": "text",
            }
        ],
    }
    context = _create_test_context(chat_msgs)

    loop2 = _make_interpreted_loop(
        source_msg="M402",
        chat_id="C02",
        title="Send file two",
        type="commitment",
        owner="ME",
        suggested_action="send file two",
        confidence=0.85,
    )

    diagnostic = BatchFailureDiagnostic(
        phase="interpretation",
        batch_id="C01_b1",
        attempt_count=1,
        error_type="RateLimitError",
        error_message="Rate limit exceeded",
        timestamp=1700000000.0,
    )

    chat_responses = {
        "C01": ([], [], diagnostic),
        "C02": ([loop2], [], None),
    }

    mock = MockEvidenceInterpreter(chat_responses=chat_responses)
    diag = OpenLoopExtractionDiagnostics()

    result = extract_open_loops(context, mock, diagnostics=diag)

    assert result.loop_count == 1
    assert result.open_loops[0]["source_msg"] == "M402"
    assert diag.batches_created == 2
    assert diag.batches_failed == 1
    assert diag.batches_succeeded == 1


def test_extract_open_loops_fatal_configuration_error_bubbles_up():
    """Verify LLMConfigurationError bubbles up immediately."""
    chat_msgs = {
        "C01": [
            {
                "id": "M500",
                "chat_id": "C01",
                "sender_id": "ME",
                "ts": 1700000000,
                "body": "I will send the report",
                "kind": "text",
            }
        ]
    }
    context = _create_test_context(chat_msgs)
    mock = MockEvidenceInterpreter(raise_error=LLMConfigurationError("Missing API key"))

    with pytest.raises(LLMConfigurationError, match="Missing API key"):
        extract_open_loops(context, mock)


# ==============================================================================
# 20. Configurable max_loops_per_chat
# ==============================================================================


def test_extract_open_loops_max_loops_per_chat_cap():
    """Verify max_loops_per_chat truncates excess loops in that chat and updates diagnostics."""
    tasks = [
        ("Send invoice", "I will send invoice"),
        ("Review contract", "I will review contract"),
        ("Fix faucet", "I will fix faucet"),
        ("Submit report", "I will submit report"),
        ("Clean garage", "I will clean garage"),
    ]
    msgs = []
    loops = []
    for i, (title, body) in enumerate(tasks):
        mid = f"M60{i}"
        msgs.append({
            "id": mid,
            "chat_id": "C01",
            "sender_id": "ME",
            "ts": 1700000000 + i * 10,
            "body": body,
            "kind": "text",
        })
        loops.append(
            _make_interpreted_loop(
                source_msg=mid,
                chat_id="C01",
                title=title,
                type="commitment",
                owner="ME",
                suggested_action=f"finish {title}",
                confidence=0.85,
            )
        )

    chat_msgs = {"C01": msgs}
    context = _create_test_context(chat_msgs)
    mock = MockEvidenceInterpreter(default_loops=loops)
    config = OpenLoopExtractionConfig(max_loops_per_chat=2)
    diag = OpenLoopExtractionDiagnostics()

    result = extract_open_loops(context, mock, config=config, diagnostics=diag)

    assert result.loop_count == 2
    assert diag.loops_truncated_by_cap == 3


def test_rank_and_clamp_rejects_invalid_cap_values():
    """Verify rank_and_clamp_open_loops strictly rejects invalid cap values instead of treating them as unlimited."""
    cands = [
        CandidateOpenLoop(
            title="Task A",
            type="commitment",
            owner="ME",
            chat_id="C01",
            source_msg="M01",
            confidence=0.85,
            status="open",
            help=HelpAction(action="assist", next_step="do step", needs_yes=True),
        )
    ]
    chat_msgs = {"C01": [{"id": "M01", "chat_id": "C01", "ts": 100}]}
    context = _create_test_context(chat_msgs)

    class MockDuckConfig:
        def __init__(self, max_open_loops=None, max_loops_per_chat=None):
            self.max_open_loops = max_open_loops
            self.max_loops_per_chat = max_loops_per_chat

    # 1. Zero values must be rejected with ValueError
    with pytest.raises(ValueError, match="strictly positive integer"):
        rank_and_clamp_open_loops(cands, context, MockDuckConfig(max_open_loops=0))

    with pytest.raises(ValueError, match="strictly positive integer"):
        rank_and_clamp_open_loops(cands, context, MockDuckConfig(max_loops_per_chat=0))

    # 2. Negative values must be rejected with ValueError
    with pytest.raises(ValueError, match="strictly positive integer"):
        rank_and_clamp_open_loops(cands, context, MockDuckConfig(max_open_loops=-5))

    with pytest.raises(ValueError, match="strictly positive integer"):
        rank_and_clamp_open_loops(cands, context, MockDuckConfig(max_loops_per_chat=-1))

    # 3. Booleans must be rejected with TypeError
    with pytest.raises(TypeError, match="cannot be a boolean"):
        rank_and_clamp_open_loops(cands, context, MockDuckConfig(max_open_loops=True))

    with pytest.raises(TypeError, match="cannot be a boolean"):
        rank_and_clamp_open_loops(cands, context, MockDuckConfig(max_loops_per_chat=False))

    # 4. Floats and strings must be rejected with TypeError
    with pytest.raises(TypeError, match="must be an integer or None"):
        rank_and_clamp_open_loops(cands, context, MockDuckConfig(max_open_loops=10.5))

    with pytest.raises(TypeError, match="must be an integer or None"):
        rank_and_clamp_open_loops(cands, context, MockDuckConfig(max_loops_per_chat="3"))


def test_rank_and_clamp_empty_candidate_list_validation():
    """Verify rank_and_clamp_open_loops validates config even when candidate list is empty."""
    chat_msgs = {"C01": [{"id": "M01", "chat_id": "C01", "ts": 100}]}
    context = _create_test_context(chat_msgs)

    class MockDuckConfig:
        def __init__(self, max_open_loops=None, max_loops_per_chat=None):
            self.max_open_loops = max_open_loops
            self.max_loops_per_chat = max_loops_per_chat

    # 1. Empty candidate list + invalid global cap raises appropriate exception
    with pytest.raises(ValueError, match="strictly positive integer"):
        rank_and_clamp_open_loops([], context, MockDuckConfig(max_open_loops=0))

    with pytest.raises(ValueError, match="strictly positive integer"):
        rank_and_clamp_open_loops([], context, MockDuckConfig(max_open_loops=-10))

    with pytest.raises(TypeError, match="cannot be a boolean"):
        rank_and_clamp_open_loops([], context, MockDuckConfig(max_open_loops=True))

    with pytest.raises(TypeError, match="must be an integer or None"):
        rank_and_clamp_open_loops([], context, MockDuckConfig(max_open_loops="invalid"))

    # 2. Empty candidate list + invalid per-chat cap raises appropriate exception
    with pytest.raises(ValueError, match="strictly positive integer"):
        rank_and_clamp_open_loops([], context, MockDuckConfig(max_loops_per_chat=0))

    with pytest.raises(ValueError, match="strictly positive integer"):
        rank_and_clamp_open_loops([], context, MockDuckConfig(max_loops_per_chat=-2))

    with pytest.raises(TypeError, match="cannot be a boolean"):
        rank_and_clamp_open_loops([], context, MockDuckConfig(max_loops_per_chat=False))

    with pytest.raises(TypeError, match="must be an integer or None"):
        rank_and_clamp_open_loops([], context, MockDuckConfig(max_loops_per_chat=3.5))

    # 3. Empty candidate list + valid unlimited configuration returns []
    valid_unlimited = OpenLoopExtractionConfig(max_open_loops=None, max_loops_per_chat=None)
    assert rank_and_clamp_open_loops([], context, valid_unlimited) == []

    # 4. Empty candidate list + valid explicit positive cap configuration returns []
    valid_capped = OpenLoopExtractionConfig(max_open_loops=10, max_loops_per_chat=2)
    assert rank_and_clamp_open_loops([], context, valid_capped) == []


def test_cap_validation_e2e_retains_74_and_respects_explicit_caps():
    """Verify end-to-end extraction retains 74 valid loops by default, respects explicit caps, and preserves ranking."""
    class DynamicInterpreter:
        def __init__(self, all_loops):
            self.loop_map = {l.source_msg: l for l in all_loops}

        def interpret_chat_batch(self, batch, context=None):
            b_msgs = getattr(batch, "messages", [])
            b_ids = {m["id"] for m in b_msgs if isinstance(m, dict) and "id" in m}
            b_loops = [self.loop_map[mid] for mid in b_ids if mid in self.loop_map]
            return b_loops, [], None

    # Construct 74 loops across 74 distinct chats
    chat_msgs = {}
    loops = []
    for i in range(74):
        cid = f"CHAT_{i:03d}"
        mid = f"MSG_{i:03d}"
        chat_msgs[cid] = [{
            "id": mid,
            "chat_id": cid,
            "sender_id": "ME",
            "ts": 1700000000 + i * 10,
            "body": f"I will complete task number {i}",
            "kind": "text",
        }]
        loops.append(
            _make_interpreted_loop(
                source_msg=mid,
                chat_id=cid,
                title=f"Complete task {i}",
                type="commitment",
                owner="ME",
                suggested_action=f"do task {i}",
                confidence=0.50 + (i * 0.005),  # MSG_073 has highest confidence (0.865)
            )
        )

    context = _create_test_context(chat_msgs)
    mock = DynamicInterpreter(loops)

    # 1. Default config (max_open_loops=None, max_loops_per_chat=None): retains all 74
    diag_default = OpenLoopExtractionDiagnostics()
    result_default = extract_open_loops(context, mock, diagnostics=diag_default)
    assert result_default.loop_count == 74
    assert len(result_default.open_loops) == 74
    assert diag_default.loops_truncated_by_cap == 0
    assert diag_default.final_loops_emitted == 74
    # Deterministic ranking: highest confidence first (MSG_073)
    assert result_default.open_loops[0]["source_msg"] == "MSG_073"

    # 2. Explicit positive global cap: max_open_loops=20 clamps to 20
    diag_capped = OpenLoopExtractionDiagnostics()
    config_capped = OpenLoopExtractionConfig(max_open_loops=20)
    result_capped = extract_open_loops(context, mock, config=config_capped, diagnostics=diag_capped)
    assert result_capped.loop_count == 20
    assert len(result_capped.open_loops) == 20
    assert diag_capped.loops_truncated_by_cap == 54
    assert diag_capped.final_loops_emitted == 20
    # Deterministic ranking preserved under cap: highest confidence still first
    assert result_capped.open_loops[0]["source_msg"] == "MSG_073"

    # 3. Explicit per-chat cap works correctly
    diag_chat_capped = OpenLoopExtractionDiagnostics()
    config_chat_capped = OpenLoopExtractionConfig(max_loops_per_chat=1)
    result_chat_capped = extract_open_loops(context, mock, config=config_chat_capped, diagnostics=diag_chat_capped)
    assert result_chat_capped.loop_count == 74
    assert diag_chat_capped.loops_truncated_by_cap == 0
