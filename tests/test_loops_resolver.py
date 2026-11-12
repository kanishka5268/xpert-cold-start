"""Unit tests for open-loop resolver.

Covers:
- Source and chat validation, participant presence, and exact ID preservation.
- Ownership resolution in direct and group chats.
- Due date anchoring, explicit year preservation, and timezone grounding.
- Full-chat closure scanning, actor authority, and cancellation vs completion.
- Lifecycle episode merging, reminders, deduplication, and recurrence.
- Confidence score validation, filtering, diagnostics, and model invariants.
"""

from datetime import date, datetime, timezone
import math
import pytest
from typing import Any, Dict, List, Optional

from src.interpretation.records import InterpretedOpenLoop
from src.loops.models import (
    CandidateOpenLoop,
    HelpAction,
    OpenLoopExtractionConfig,
    OpenLoopExtractionDiagnostics,
)
from src.loops.resolver import (
    _extract_task_action,
    _extract_task_entity,
    deduplicate_open_loops,
    is_valid_timezone_identifier,
    resolve_due_date,
    resolve_help_action,
    resolve_loop_owner,
    resolve_open_loops,
    resolve_relative_due_date,
    is_informational_status_update,
    is_non_actionable_banter,
    verify_full_chat_closure,
)
from src.preprocessing.context_builder import ConversationContext


def create_mock_context(
    chats: Optional[Dict[str, Dict[str, Any]]] = None,
    participants: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
) -> ConversationContext:
    """Create a lightweight synthetic ConversationContext for unit tests."""
    if participants is None:
        participants = {
            "ME": {"name": "Arjun", "is_silent": False, "chat_ids": ["C01", "C02"]},
            "P01": {"name": "Priya", "is_silent": False, "chat_ids": ["C01", "C02"]},
            "P02": {"name": "Bob", "is_silent": False, "chat_ids": ["C02"]},
            "P03": {"name": "Charlie", "is_silent": False, "chat_ids": ["C02"]},
            "P04": {"name": "Sunita", "is_silent": False, "chat_ids": ["C02"]},
        }

    if chats is None:
        # Default 1:1 chat C01
        c01_messages = [
            {"id": "M01", "chat_id": "C01", "sender_id": "P01", "text": "Can you check the draft?", "ts": 1700000000, "kind": "text"},
            {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "I will review it tomorrow", "ts": 1700000100, "kind": "text"},
            {"id": "M03", "chat_id": "C01", "sender_id": "P01", "text": "I'll call the plumber", "ts": 1700000200, "kind": "text"},
            {"id": "M04", "chat_id": "C01", "sender_id": "P01", "text": "The market is closed today", "ts": 1700000300, "kind": "text"},
            {"id": "M05", "chat_id": "C01", "sender_id": "P01", "text": "Rohan is handling the catering", "ts": 1700000400, "kind": "text"},
        ]
        # Default group chat C02
        c02_messages = [
            {"id": "G01", "chat_id": "C02", "sender_id": "P02", "text": "I will send the deck tomorrow", "ts": 1700001000, "kind": "text"},
            {"id": "G02", "chat_id": "C02", "sender_id": "P01", "text": "Does anyone know where the keys are?", "ts": 1700001100, "kind": "text"},
            {"id": "G03", "chat_id": "C02", "sender_id": "ME", "text": "Let's all review the doc by Friday", "ts": 1700001200, "kind": "text"},
            {"id": "G04", "chat_id": "C02", "sender_id": "P01", "text": "Sunita please send the invoice", "ts": 1700001300, "kind": "text"},
        ]
        chats = {
            "C01": {
                "chat_id": "C01",
                "name": "Priya",
                "is_group": False,
                "participants": ["ME", "P01"],
                "messages": c01_messages,
            },
            "C02": {
                "chat_id": "C02",
                "name": "Project Team",
                "is_group": True,
                "participants": ["ME", "P01", "P02", "P03", "P04"],
                "messages": c02_messages,
            },
        }

    # Index messages by ID
    messages_by_id: Dict[str, Dict[str, Any]] = {}
    all_msgs: List[Dict[str, Any]] = []
    for c in chats.values():
        for m in c.get("messages", []):
            messages_by_id[m["id"]] = m
            all_msgs.append(m)

    dataset = {
        "chats_by_id": chats,
        "messages_by_id": messages_by_id,
        "messages": all_msgs,
        "participants": participants,
    }
    if dataset_metadata:
        dataset.update(dataset_metadata)

    return ConversationContext(dataset=dataset, chats=chats, participants=participants)


# ==============================================================================
# 1. Source and Ownership Tests
# ==============================================================================

def test_source_and_ownership_valid():
    context = create_mock_context()
    loop = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Check the draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.85,
    )
    results = resolve_open_loops([loop], context)
    assert len(results) == 1
    cand = results[0]
    assert cand.source_msg == "M01"
    assert cand.owner == "ME"
    assert cand.chat_id == "C01"
    assert cand.status == "open"
    assert cand.help is not None
    assert cand.help.action == "assist"


def test_missing_source_message_rejected():
    context = create_mock_context()
    loop = InterpretedOpenLoop(
        source_msg="NON_EXISTENT_MSG",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Check draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action=None,
        confidence=0.85,
    )
    diag = OpenLoopExtractionDiagnostics()
    results = resolve_open_loops([loop], context, diagnostics=diag)
    assert len(results) == 0
    assert diag.candidates_rejected == 1


def test_source_message_from_another_chat_rejected():
    context = create_mock_context()
    # M01 is in C01, but loop claims chat_id="C02"
    loop = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C02",
        type="commitment",
        is_genuine=True,
        title="Check draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action=None,
        confidence=0.85,
    )
    diag = OpenLoopExtractionDiagnostics()
    results = resolve_open_loops([loop], context, diagnostics=diag)
    assert len(results) == 0
    assert diag.candidates_rejected == 1


def test_owner_not_present_in_chat_rejected():
    context = create_mock_context()
    # P99 is not a participant in C01 (only ME, P01)
    msg = context.dataset["messages_by_id"]["M01"]
    resolved = resolve_loop_owner("P99", "C01", msg, context)
    assert resolved != "P99"


def test_direct_chat_first_person_pledge():
    context = create_mock_context()
    # M03: P01 says "I'll call the plumber" in 1:1 chat C01
    msg = context.dataset["messages_by_id"]["M03"]
    resolved = resolve_loop_owner(None, "C01", msg, context)
    assert resolved == "P01"


def test_direct_chat_explicit_request_to_me():
    context = create_mock_context()
    # M01: P01 says "Can you check the draft?" in 1:1 chat C01
    msg = context.dataset["messages_by_id"]["M01"]
    resolved = resolve_loop_owner(None, "C01", msg, context)
    assert resolved == "ME"


def test_group_first_person_pledge():
    context = create_mock_context()
    # G01: P02 says "I will send the deck tomorrow" in group C02
    msg = context.dataset["messages_by_id"]["G01"]
    resolved = resolve_loop_owner(None, "C02", msg, context)
    assert resolved == "P02"


def test_group_grounded_named_assignment():
    context = create_mock_context()
    # G04: P01 says "Sunita please send the invoice" in group C02
    msg = context.dataset["messages_by_id"]["G04"]
    resolved = resolve_loop_owner(None, "C02", msg, context)
    # Sunita is P04
    assert resolved == "P04"


def test_ambiguous_group_broadcast_rejected():
    context = create_mock_context()
    # G02: P01 says "Does anyone know where the keys are?" in group C02
    msg = context.dataset["messages_by_id"]["G02"]
    resolved = resolve_loop_owner("ME", "C02", msg, context)
    assert resolved is None


def test_third_party_task_not_attributed_to_me():
    context = create_mock_context()
    # M05: P01 says "Rohan is handling the catering" in 1:1 chat C01
    msg = context.dataset["messages_by_id"]["M05"]
    resolved = resolve_loop_owner("ME", "C01", msg, context)
    assert resolved is None


def test_situational_observation_not_attributed_to_me():
    context = create_mock_context()
    # M04: P01 says "The market is closed today" in 1:1 chat C01
    msg = context.dataset["messages_by_id"]["M04"]
    resolved = resolve_loop_owner("ME", "C01", msg, context)
    assert resolved is None


def test_exact_identifier_preservation_whitespace_rejected():
    context = create_mock_context()
    # Padded identifiers must be rejected, not stripped
    loop = InterpretedOpenLoop(
        source_msg=" M01 ",  # Leading and trailing whitespace
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Check draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action=None,
        confidence=0.85,
    )
    diag = OpenLoopExtractionDiagnostics()
    results = resolve_open_loops([loop], context, diagnostics=diag)
    assert len(results) == 0
    assert diag.candidates_rejected == 1


# ==============================================================================
# 2. Due Date and Anchor Tests
# ==============================================================================

def test_valid_due_date_and_anchor_with_explicit_year():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M10", "chat_id": "C01", "sender_id": "ME", "text": "I will submit the paper by 15 Jan 2025", "ts": 1700000000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    due, anchor = resolve_due_date("2025-01-15", "M10", source_msg, "C01", context)
    assert due == "2025-01-15"
    assert anchor == "M10"


def test_due_date_missing_anchor():
    context = create_mock_context()
    source_msg = context.dataset["messages_by_id"]["M01"]
    due, anchor = resolve_due_date("2025-01-15", "NON_EXISTENT_ANCHOR", source_msg, "C01", context)
    assert due is None
    assert anchor is None


def test_due_date_anchor_in_another_chat():
    context = create_mock_context()
    source_msg = context.dataset["messages_by_id"]["M01"]  # in C01
    # G01 is in C02
    due, anchor = resolve_due_date("2025-01-15", "G01", source_msg, "C01", context)
    assert due is None
    assert anchor is None


def test_due_date_anchor_before_source_message():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P01", "text": "Due on 2025-01-15", "ts": 1700000000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "I will send it later", "ts": 1700000500, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][1]  # M02 at ts=1700000500
    # Anchor M01 is at ts=1700000000, which predates source message
    due, anchor = resolve_due_date("2025-01-15", "M01", source_msg, "C01", context)
    assert due is None
    assert anchor is None


def test_due_date_invalid_timestamp():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Due 2025-01-15", "ts": "invalid_ts", "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    due, anchor = resolve_due_date("2025-01-15", "M01", source_msg, "C01", context)
    assert due is None
    assert anchor is None


def test_due_date_missing_year_without_grounding_returns_none():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Draft due March 25", "ts": 1700000000, "kind": "text"},
            ],
        }
    }
    # No timezone in context metadata or text
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", context)
    assert due is None
    assert anchor is None


def test_due_date_with_authoritative_timezone():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Draft due March 25 at 5:00 PM IST", "ts": 1700000000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", context)
    assert due == "2025-03-25"
    assert anchor == "M01"


# ==============================================================================
# 3. Full-Chat Closure Tests
# ==============================================================================

def test_closure_completion_by_owner():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Bob, please pay the electricity bill", "ts": 1700000000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P01", "text": "Electricity bill paid", "ts": 1700000100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P01",
        loop_type="commitment",
        title="Pay electricity bill",
        context=context,
    )
    assert status == "done"
    assert closed_by == "M02"


def test_closure_already_taken_care_of_by_owner_is_done():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Did you pay the electricity bill?", "ts": 1700000000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P01", "text": "Already taken care of", "ts": 1700000100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P01",
        loop_type="commitment",
        title="Pay electricity bill",
        context=context,
    )
    assert status == "done"
    assert closed_by == "M02"


def test_closure_already_taken_care_of_by_bystander_rejected():
    chats = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02", "P03"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "ME", "text": "Bob please pay the electricity bill", "ts": 1700000000, "kind": "text"},
                # P03 is an unrelated bystander saying "already taken care of"
                {"id": "G02", "chat_id": "C02", "sender_id": "P03", "text": "Already taken care of", "ts": 1700000100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C02"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C02",
        owner="P02",
        loop_type="commitment",
        title="Pay electricity bill",
        context=context,
    )
    assert status == "open"
    assert closed_by is None


def test_closure_authenticated_delegate():
    chats = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P02", "P04"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "ME", "text": "Bob please send the Orbit invoice", "ts": 1700000000, "kind": "text"},
                # Bob (P02) establishes Sunita (P04) as delegate
                {"id": "G02", "chat_id": "C02", "sender_id": "P02", "text": "Sunita handles billing and will send the invoice", "ts": 1700000100, "kind": "text"},
                # Sunita (P04) delivers invoice
                {"id": "G03", "chat_id": "C02", "sender_id": "P04", "text": "Orbit invoice attached", "ts": 1700000200, "kind": "document"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C02"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C02",
        owner="P02",
        loop_type="commitment",
        title="Send Orbit invoice",
        context=context,
    )
    assert status == "done"
    assert closed_by == "G03"


def test_closure_unauthorized_document_delivery_rejected():
    chats = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P02", "P03"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "ME", "text": "Bob please send the Orbit invoice", "ts": 1700000000, "kind": "text"},
                # P03 (ordinary colleague without authority) shares attachment
                {"id": "G02", "chat_id": "C02", "sender_id": "P03", "text": "Orbit invoice attached", "ts": 1700000100, "kind": "document"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C02"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C02",
        owner="P02",
        loop_type="commitment",
        title="Send Orbit invoice",
        context=context,
    )
    assert status == "open"
    assert closed_by is None


def test_closure_explicit_cancellation_by_requester():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Please book the flight to Delhi", "ts": 1700000000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "Never mind, cancel that flight, meeting got rescheduled", "ts": 1700000100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P01",
        loop_type="commitment",
        title="Book flight to Delhi",
        context=context,
    )
    assert status == "dropped"
    assert closed_by == "M02"


def test_closure_authorized_preemption_by_obligor():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P01", "text": "Can you send workout plan?", "ts": 1700000000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P01", "text": "Cancel that, trainer at gym gave me one", "ts": 1700000100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="ME",
        loop_type="commitment",
        title="Send workout plan",
        context=context,
    )
    assert status == "dropped"
    assert closed_by == "M02"


def test_closure_postponement_retains_open():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Call the plumber for the tap", "ts": 1700000000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P01", "text": "Leave it for now, let's see later", "ts": 1700000100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P01",
        loop_type="commitment",
        title="Call plumber",
        context=context,
    )
    assert status == "open"
    assert closed_by is None


# ==============================================================================
# 4. Lifecycle and Episode Merging Tests
# ==============================================================================

def test_reminder_merges_into_existing_open_loop():
    context = create_mock_context()
    cand1 = CandidateOpenLoop(
        title="Review pitch deck",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-15T10:00:00Z"),
        source_msgs_all=["M01"],
    )
    # Cand2 is a reminder referencing same task
    cand2 = CandidateOpenLoop(
        title="Checking in on pitch deck review",
        type="followup",
        owner="ME",
        chat_id="C01",
        source_msg="M02",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-16T10:00:00Z"),
        source_msgs_all=["M02"],
    )
    diag = OpenLoopExtractionDiagnostics()
    merged = deduplicate_open_loops([cand1, cand2], context, diagnostics=diag)
    assert len(merged) == 1
    assert merged[0].source_msg == "M01"  # Original source preserved
    assert "M02" in merged[0].source_msgs_all  # Reminder appended
    assert diag.loops_deduplicated == 1


def test_completed_obligation_followed_by_new_commitment_creates_new_episode():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Please send October invoice", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P01", "text": "October invoice attached", "ts": 2000, "kind": "document"},
                {"id": "M03", "chat_id": "C01", "sender_id": "ME", "text": "Please send November invoice", "ts": 3000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    ep1 = CandidateOpenLoop(
        title="Send October invoice",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M01",
        status="done",
        closed_by="M02",
        source_msgs_all=["M01"],
    )
    ep2 = CandidateOpenLoop(
        title="Send November invoice",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M03",
        status="open",
        help=HelpAction(action="leave", reason="P01 owns this task"),
        source_msgs_all=["M03"],
    )
    merged = deduplicate_open_loops([ep1, ep2], context)
    assert len(merged) == 2
    # Verify both episodes are retained and second episode keeps its own initiating source_msg
    assert merged[0].source_msg == "M01"
    assert merged[0].status == "done"
    assert merged[1].source_msg == "M03"
    assert merged[1].status == "open"


# ==============================================================================
# 5. Confidence, Diagnostics, and Model Invariants Tests
# ==============================================================================

def test_confidence_threshold_filtering():
    context = create_mock_context()
    loop_low_conf = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Review draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action=None,
        confidence=0.3,  # Below default min_confidence=0.5
    )
    config = OpenLoopExtractionConfig(min_confidence=0.5)
    diag = OpenLoopExtractionDiagnostics()
    results = resolve_open_loops([loop_low_conf], context, config=config, diagnostics=diag)
    assert len(results) == 0
    assert diag.candidates_rejected == 1


def test_boolean_and_non_finite_confidence_rejected():
    context = create_mock_context()
    loop_bool_conf = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Review draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action=None,
        confidence=True,  # type: ignore[arg-type]
    )
    diag = OpenLoopExtractionDiagnostics()
    results = resolve_open_loops([loop_bool_conf], context, diagnostics=diag)
    assert len(results) == 0
    assert diag.candidates_rejected == 1


def test_help_action_resolution_for_me_assist_vs_remind():
    context = create_mock_context()
    msg = context.dataset["messages_by_id"]["M01"]
    config = OpenLoopExtractionConfig()

    # Research / drafting task resolves to assist
    help_assist = resolve_help_action("Draft proposal for partnership", "ME", msg, context, config)
    assert help_assist.action == "assist"
    assert help_assist.next_step is not None
    assert "\n" not in help_assist.next_step
    assert help_assist.validate() is True

    # Deadline / payment task resolves to remind
    help_remind = resolve_help_action("Pay electricity bill", "ME", msg, context, config)
    assert help_remind.action == "remind"
    assert help_remind.remind_at is not None
    assert help_remind.remind_at.endswith("Z")
    assert help_remind.validate() is True


def test_help_action_resolution_for_contact_leaves():
    context = create_mock_context()
    msg = context.dataset["messages_by_id"]["M01"]
    config = OpenLoopExtractionConfig()

    help_leave = resolve_help_action("Call the plumber", "P01", msg, context, config)
    assert help_leave.action == "leave"
    assert "P01" in help_leave.reason  # Grounded reason
    assert help_leave.validate() is True


def test_end_to_end_candidate_model_validation_and_serialization():
    context = create_mock_context()
    loop = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Review draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.9,
    )
    results = resolve_open_loops([loop], context)
    assert len(results) == 1
    cand = results[0]

    # Validate model
    assert cand.validate() is True

    # Validate schema dict serialization
    schema_dict = cand.to_schema_dict()
    assert schema_dict["title"] == "Review draft"
    assert schema_dict["type"] == "commitment"
    assert schema_dict["owner"] == "ME"
    assert schema_dict["chat_id"] == "C01"
    assert schema_dict["source_msg"] == "M01"
    assert schema_dict["status"] == "open"
    assert schema_dict["closed_by"] is None
    assert schema_dict["help"]["action"] == "assist"
    # Ensure internal fields are not serialized
    assert "due_anchor_msg" not in schema_dict
    assert "source_msgs_all" not in schema_dict
    assert "confidence" not in schema_dict


def test_deterministic_resolver_output_across_repeated_runs():
    context = create_mock_context()
    loop = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Check the draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.88,
    )
    run1 = resolve_open_loops([loop], context)
    run2 = resolve_open_loops([loop], context)
    assert len(run1) == len(run2) == 1
    assert run1[0].to_schema_dict() == run2[0].to_schema_dict()


def test_no_speculative_year_roll_forward():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P01", "text": "The invoice due Sept 15 was unpaid", "ts": 1700000000, "kind": "text"},
            ],
        }
    }
    # Dataset grounded in 2024
    context = create_mock_context(chats=chats, dataset_metadata={"timezone": "UTC"})
    source_msg = chats["C01"]["messages"][0]
    # Stated date Sept 15 in past should not be rolled forward to 2025
    due, anchor = resolve_due_date("2024-09-15", "M01", source_msg, "C01", context)
    assert due == "2024-09-15"
    assert anchor == "M01"


def test_closure_confirmation_by_original_requester():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Bob",
            "is_group": False,
            "participants": ["ME", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P02", "text": "I'll pay the electricity bill today", "ts": 1700000000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "Already taken care of, thanks!", "ts": 1700000100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P02",
        loop_type="commitment",
        title="Pay electricity bill",
        context=context,
    )
    assert status == "done"
    assert closed_by == "M02"


def test_closure_unrelated_completion_does_not_close():
    chats = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "ME", "text": "Bob please pay the electricity bill", "ts": 1700000000, "kind": "text"},
                {"id": "G02", "chat_id": "C02", "sender_id": "P01", "text": "Where should we have lunch?", "ts": 1700000100, "kind": "text"},
                {"id": "G03", "chat_id": "C02", "sender_id": "P02", "text": "Pizza place", "ts": 1700000200, "kind": "text"},
                {"id": "G04", "chat_id": "C02", "sender_id": "P01", "text": "Sounds good", "ts": 1700000300, "kind": "text"},
                # Unquoted generic done on unrelated lunch conversation
                {"id": "G05", "chat_id": "C02", "sender_id": "P02", "text": "done", "ts": 1700000400, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C02"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C02",
        owner="P02",
        loop_type="commitment",
        title="Pay electricity bill",
        context=context,
    )
    assert status == "open"
    assert closed_by is None


def test_closure_message_from_another_chat_rejected():
    context = create_mock_context()
    source_msg = context.dataset["messages_by_id"]["M01"]  # in C01
    # G01 is in C02
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="ME",
        loop_type="commitment",
        title="Check the draft",
        context=context,
        raw_status="done",
        raw_closed_by="G01",
    )
    assert status == "open"
    assert closed_by is None


def test_invalid_or_missing_closed_by_does_not_produce_terminal_candidate():
    context = create_mock_context()
    # A candidate cannot have status done with closed_by None
    cand = CandidateOpenLoop(
        title="Check draft",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="done",
        closed_by=None,
    )
    assert cand.validate() is False


def test_status_update_merges_into_same_episode():
    context = create_mock_context()
    ep1 = CandidateOpenLoop(
        title="Submit research paper",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        due="2025-01-15",
        due_anchor_msg="M01",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-14T10:00:00Z"),
        source_msgs_all=["M01"],
    )
    # ep2 amends deadline to 2025-01-20
    ep2 = CandidateOpenLoop(
        title="Research paper submission updated",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M02",
        due="2025-01-20",
        due_anchor_msg="M02",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-19T10:00:00Z"),
        source_msgs_all=["M02"],
    )
    diag = OpenLoopExtractionDiagnostics()
    merged = deduplicate_open_loops([ep1, ep2], context, diagnostics=diag)
    assert len(merged) == 1
    assert merged[0].source_msg == "M01"  # Retains original source
    assert merged[0].due == "2025-01-20"  # Updated due date
    assert merged[0].due_anchor_msg == "M02"
    assert "M02" in merged[0].source_msgs_all
    assert diag.loops_deduplicated == 1


def test_duplicate_interpretations_do_not_create_duplicate_loops():
    context = create_mock_context()
    loop1 = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Review draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.9,
    )
    loop2 = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Review draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.85,
    )
    diag = OpenLoopExtractionDiagnostics()
    results = resolve_open_loops([loop1, loop2], context, diagnostics=diag)
    assert len(results) == 1
    assert diag.loops_deduplicated == 1


def test_unrelated_later_completion_does_not_close_earlier_obligation():
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Please book flight to Mumbai", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "Can you also check weather?", "ts": 2000, "kind": "text"},
                {"id": "M03", "chat_id": "C01", "sender_id": "P01", "text": "Weather checked, it will rain", "ts": 3000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]  # M01: Book flight
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P01",
        loop_type="commitment",
        title="Book flight to Mumbai",
        context=context,
    )
    # Weather answer does not close flight booking
    assert status == "open"
    assert closed_by is None


# ==============================================================================
# Targeted Correction Tests (Milestone 3 Source Review)
# ==============================================================================

def test_messages_using_body_field_ownership():
    """Verify ownership resolution works when messages use 'body' instead of 'text'."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P01", "body": "Can you check the draft?", "ts": 1700000000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P01", "body": "I'll call the plumber", "ts": 1700000100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    m01 = chats["C01"]["messages"][0]
    m02 = chats["C01"]["messages"][1]

    # Explicit request in body resolves to ME
    assert resolve_loop_owner(None, "C01", m01, context) == "ME"
    # First-person pledge in body resolves to P01
    assert resolve_loop_owner(None, "C01", m02, context) == "P01"


def test_messages_using_body_field_closure():
    """Verify closure resolution works when messages use 'body' instead of 'text'."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "body": "Please book flight to Delhi", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P01", "body": "Delhi flight booked, confirmation attached", "ts": 2000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P01",
        loop_type="commitment",
        title="Book flight to Delhi",
        context=context,
    )
    assert status == "done"
    assert closed_by == "M02"


def test_raw_closed_by_unrelated_completion_rejected():
    """raw_closed_by pointing to an unrelated completion message by owner must be rejected."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Bob",
            "is_group": False,
            "participants": ["ME", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Bob please pay the electricity bill", "ts": 1000, "kind": "text"},
                # M05 is sent much later about lunch, containing bare 'done'
                {"id": "M05", "chat_id": "C01", "sender_id": "P02", "text": "pizza was great, done", "ts": 9000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P02",
        loop_type="commitment",
        title="Pay electricity bill",
        context=context,
        raw_status="done",
        raw_closed_by="M05",
    )
    assert status == "open"
    assert closed_by is None


def test_raw_closed_by_unrelated_cancellation_rejected():
    """raw_closed_by pointing to an unrelated cancellation must be rejected."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Bob",
            "is_group": False,
            "participants": ["ME", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Please draft the partnership proposal", "ts": 1000, "kind": "text"},
                # M08 is cancelling an unrelated lunch
                {"id": "M08", "chat_id": "C01", "sender_id": "ME", "text": "cancel that lunch plan, I have another meeting", "ts": 8000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P02",
        loop_type="commitment",
        title="Draft partnership proposal",
        context=context,
        raw_status="dropped",
        raw_closed_by="M08",
    )
    assert status == "open"
    assert closed_by is None


def test_unrelated_document_delivery_shortly_after_source_rejected():
    """An unrelated attachment shortly after source message must not close document delivery."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Vikram",
            "is_group": False,
            "participants": ["ME", "P05"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Vikram please send the Orbit October invoice", "ts": 1000, "kind": "text"},
                # Vikram sends a personal photo 1 message later without linkage to invoice
                {"id": "M02", "chat_id": "C01", "sender_id": "P05", "text": "Here is the vacation photo", "ts": 1100, "kind": "document", "caption": "vacation.jpg"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P05",
        loop_type="commitment",
        title="Send Orbit October invoice",
        context=context,
    )
    assert status == "open"
    assert closed_by is None


def test_genuinely_linked_completion_and_cancellation_pass():
    """Genuinely linked completion and cancellation messages pass verification."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Bob",
            "is_group": False,
            "participants": ["ME", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Please send the Q3 marketing presentation", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P02", "text": "Q3 marketing presentation attached", "ts": 2000, "kind": "document", "caption": "q3_marketing.pdf"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P02",
        loop_type="commitment",
        title="Send Q3 marketing presentation",
        context=context,
        raw_status="done",
        raw_closed_by="M02",
    )
    assert status == "done"
    assert closed_by == "M02"


def test_different_invoices_similar_titles_remain_separate():
    """Different month invoices sharing generic words 'send' and 'invoice' must remain separate episodes."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Vendor",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Please send October invoice", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "Please send November invoice", "ts": 3000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    ep1 = CandidateOpenLoop(
        title="Send October invoice",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M01",
        status="open",
        help=HelpAction(action="leave", reason="P01 owns this task"),
        source_msgs_all=["M01"],
    )
    ep2 = CandidateOpenLoop(
        title="Send November invoice",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M02",
        status="open",
        help=HelpAction(action="leave", reason="P01 owns this task"),
        source_msgs_all=["M02"],
    )
    merged = deduplicate_open_loops([ep1, ep2], context)
    assert len(merged) == 2
    assert merged[0].source_msg == "M01"
    assert merged[1].source_msg == "M02"


def test_different_reports_differing_targets_remain_separate():
    """Different reports with distinct targets must remain separate episodes."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Team",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Send sales report", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "Send marketing report", "ts": 2000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    ep1 = CandidateOpenLoop(
        title="Send sales report",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M01",
        status="open",
        help=HelpAction(action="leave", reason="P01 owns this task"),
        source_msgs_all=["M01"],
    )
    ep2 = CandidateOpenLoop(
        title="Send marketing report",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M02",
        status="open",
        help=HelpAction(action="leave", reason="P01 owns this task"),
        source_msgs_all=["M02"],
    )
    merged = deduplicate_open_loops([ep1, ep2], context)
    assert len(merged) == 2
    assert merged[0].source_msg == "M01"
    assert merged[1].source_msg == "M02"


def test_reminder_merges_into_correct_open_episode():
    """When multiple open episodes exist, a reminder merges into the matching open episode."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Team",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Book flight to Delhi", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "Order the book from Amazon", "ts": 2000, "kind": "text"},
                {"id": "M03", "chat_id": "C01", "sender_id": "ME", "text": "Did you book the flight to Delhi?", "ts": 3000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    ep_flight = CandidateOpenLoop(
        title="Book flight to Delhi",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M01",
        status="open",
        help=HelpAction(action="leave", reason="P01 owns this task"),
        source_msgs_all=["M01"],
    )
    ep_book = CandidateOpenLoop(
        title="Order book from Amazon",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M02",
        status="open",
        help=HelpAction(action="leave", reason="P01 owns this task"),
        source_msgs_all=["M02"],
    )
    cand_reminder = CandidateOpenLoop(
        title="Checking in on flight to Delhi",
        type="followup",
        owner="P01",
        chat_id="C01",
        source_msg="M03",
        status="open",
        help=HelpAction(action="leave", reason="P01 owns this task"),
        source_msgs_all=["M03"],
    )
    diag = OpenLoopExtractionDiagnostics()
    merged = deduplicate_open_loops([ep_flight, ep_book, cand_reminder], context, diagnostics=diag)
    assert len(merged) == 2
    # Verify flight episode received M03 in source_msgs_all and kept M01 as source_msg
    flight_res = next(ep for ep in merged if "Delhi" in ep.title)
    assert flight_res.source_msg == "M01"
    assert "M03" in flight_res.source_msgs_all
    assert diag.loops_deduplicated == 1


def test_renewed_obligation_with_multiple_later_updates():
    """Verify renewed obligation creates one new episode and later updates merge into it."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Client",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P01", "text": "Please send monthly timesheet", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "Timesheet sent", "ts": 2000, "kind": "text"},
                {"id": "M03", "chat_id": "C01", "sender_id": "P01", "text": "Please send monthly timesheet", "ts": 3000, "kind": "text"},
                {"id": "M04", "chat_id": "C01", "sender_id": "P01", "text": "Checking in on timesheet", "ts": 4000, "kind": "text"},
                {"id": "M05", "chat_id": "C01", "sender_id": "P01", "text": "Timesheet deadline is Friday", "ts": 5000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    ep1 = CandidateOpenLoop(
        title="Send monthly timesheet",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="done",
        closed_by="M02",
        source_msgs_all=["M01"],
    )
    # M03: Renewed obligation after closure
    cand_renewed = CandidateOpenLoop(
        title="Send monthly timesheet",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M03",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-15T10:00:00Z"),
        source_msgs_all=["M03"],
    )
    # M04: First update for renewed obligation
    cand_update1 = CandidateOpenLoop(
        title="Checking in on timesheet",
        type="followup",
        owner="ME",
        chat_id="C01",
        source_msg="M04",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-16T10:00:00Z"),
        source_msgs_all=["M04"],
    )
    # M05: Second update for renewed obligation
    cand_update2 = CandidateOpenLoop(
        title="Timesheet deadline reminder",
        type="followup",
        owner="ME",
        chat_id="C01",
        source_msg="M05",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-17T10:00:00Z"),
        source_msgs_all=["M05"],
    )

    merged = deduplicate_open_loops([ep1, cand_renewed, cand_update1, cand_update2], context)
    assert len(merged) == 2
    # First episode stays done with M01
    assert merged[0].source_msg == "M01"
    assert merged[0].status == "done"
    # Renewed episode retains its own initiating source_msg M03, and merged M04 and M05
    assert merged[1].source_msg == "M03"
    assert merged[1].status == "open"
    assert "M04" in merged[1].source_msgs_all
    assert "M05" in merged[1].source_msgs_all


def test_incomplete_or_ambiguous_assist_never_fabricates_proposal_text():
    """Assist suggestions without grounded actionable text must not fabricate synthetic proposal text."""
    context = create_mock_context()
    msg = context.dataset["messages_by_id"]["M01"]
    config = OpenLoopExtractionConfig()

    # Vague title with suggested_action="assist" should NOT generate "Draft proposal for General pricing question"
    help_res = resolve_help_action(
        loop_title="General pricing question",
        owner="ME",
        source_msg=msg,
        context=context,
        config=config,
        suggested_action="assist",
    )
    # Falls back to remind or leave; never fabricates "Draft proposal for ..."
    assert help_res.action != "assist" or not help_res.next_step.startswith("Draft proposal for")
    assert "Draft proposal for General pricing question" not in str(help_res)
    assert help_res.validate() is True


def test_missing_confidence_rejected():
    """Loop with missing confidence (None) must be rejected rather than receiving artificial high score."""
    context = create_mock_context()
    loop_none_conf = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Review draft",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=None,  # Missing confidence
    )
    diag = OpenLoopExtractionDiagnostics()
    results = resolve_open_loops([loop_none_conf], context, diagnostics=diag)
    assert len(results) == 0
    assert diag.candidates_rejected == 1


def test_invalid_confidence_values_rejected():
    """All non-numeric, non-finite, and out-of-range confidence scores must be rejected."""
    context = create_mock_context()
    for bad_conf in [True, False, float("nan"), float("inf"), float("-inf"), -0.1, 1.05]:
        loop = InterpretedOpenLoop(
            source_msg="M01",
            chat_id="C01",
            type="commitment",
            is_genuine=True,
            title="Review draft",
            owner="ME",
            target_person=None,
            status="open",
            closed_by=None,
            due_date=None,
            suggested_action="assist",
            confidence=bad_conf,  # type: ignore[arg-type]
        )
        diag = OpenLoopExtractionDiagnostics()
        results = resolve_open_loops([loop], context, diagnostics=diag)
        assert len(results) == 0
        assert diag.candidates_rejected == 1


# ==============================================================================
# Final Targeted Resolver Tests (Milestone 3 Pre-Approval)
# ==============================================================================

def test_cancellation_by_owner_bare_phrase_rejected():
    """An obligor saying only a bare cancellation phrase without preemption reason cannot drop the loop."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Bob",
            "is_group": False,
            "participants": ["ME", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Bob please pay the electricity bill", "ts": 1000, "kind": "text"},
                # Bob is obligor, saying bare 'cancel that' without preemption reason
                {"id": "M02", "chat_id": "C01", "sender_id": "P02", "text": "Cancel that", "ts": 1100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P02",
        loop_type="commitment",
        title="Pay electricity bill",
        context=context,
    )
    # Must remain open because obligor lacks authority for bare cancellation
    assert status == "open"
    assert closed_by is None


def test_cancellation_by_owner_with_concrete_preemption_accepted():
    """An obligor reporting an authorized preemption or external reason can drop the loop."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Bob",
            "is_group": False,
            "participants": ["ME", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Bob please draft the client proposal", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P02", "text": "Cancel that, client cancelled the meeting and project was scrapped", "ts": 1100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P02",
        loop_type="commitment",
        title="Draft client proposal",
        context=context,
    )
    assert status == "dropped"
    assert closed_by == "M02"


def test_unsupported_raw_owner_in_direct_chat_rejected():
    """Do not accept raw_owner merely because the participant is in the direct chat."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "The weather is very sunny today", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P01", "text": "I had some pasta for lunch", "ts": 2000, "kind": "text"},
                {"id": "M03", "chat_id": "C01", "sender_id": "ME", "text": "I will review the pull request by 5pm", "ts": 3000, "kind": "text"},
                {"id": "M04", "chat_id": "C01", "sender_id": "ME", "text": "Can you send the contract?", "ts": 4000, "kind": "text"},
                {"id": "M05", "chat_id": "C01", "sender_id": "P01", "text": "I'll call the plumber", "ts": 5000, "kind": "text"},
                {"id": "M06", "chat_id": "C01", "sender_id": "P01", "text": "Can you check the draft?", "ts": 6000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    m01 = chats["C01"]["messages"][0]
    m02 = chats["C01"]["messages"][1]
    m03 = chats["C01"]["messages"][2]
    m04 = chats["C01"]["messages"][3]
    m05 = chats["C01"]["messages"][4]
    m06 = chats["C01"]["messages"][5]

    # Non-actionable statement by ME: unsupported raw_owner in both directions rejected
    assert resolve_loop_owner("ME", "C01", m01, context) is None
    assert resolve_loop_owner("P01", "C01", m01, context) is None

    # Non-actionable statement by contact: unsupported raw_owner in both directions rejected
    assert resolve_loop_owner("ME", "C01", m02, context) is None
    assert resolve_loop_owner("P01", "C01", m02, context) is None

    # Valid first-person commitment by ME -> ME
    assert resolve_loop_owner("ME", "C01", m03, context) == "ME"
    # Mismatched attribution to P01 on ME's pledge -> rejected
    assert resolve_loop_owner("P01", "C01", m03, context) is None

    # Valid explicit request by ME -> P01
    assert resolve_loop_owner("P01", "C01", m04, context) == "P01"
    # Mismatched attribution to ME on request to P01 -> rejected
    assert resolve_loop_owner("ME", "C01", m04, context) is None

    # Valid first-person pledge by P01 -> P01
    assert resolve_loop_owner("P01", "C01", m05, context) == "P01"
    # Mismatched attribution to ME on P01's pledge -> rejected
    assert resolve_loop_owner("ME", "C01", m05, context) is None

    # Valid explicit request by P01 to ME -> ME
    assert resolve_loop_owner("ME", "C01", m06, context) == "ME"
    # Mismatched attribution to P01 on request to ME -> rejected
    assert resolve_loop_owner("P01", "C01", m06, context) is None


def test_orbit_invoice_and_orbit_contract_remain_separate():
    """Obligations sharing a single company token like 'Orbit' with different entities must remain separate."""
    context = create_mock_context()
    ep1 = CandidateOpenLoop(
        title="Send Orbit invoice",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-15T10:00:00Z"),
        source_msgs_all=["M01"],
    )
    ep2 = CandidateOpenLoop(
        title="Review Orbit contract",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M02",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-16T10:00:00Z"),
        source_msgs_all=["M02"],
    )
    merged = deduplicate_open_loops([ep1, ep2], context)
    assert len(merged) == 2
    assert merged[0].source_msg == "M01"
    assert merged[1].source_msg == "M02"


def test_different_actions_on_same_target_remain_separate():
    """Distinct substantive actions on the same entity (e.g. 'Send Orbit invoice' vs 'Review Orbit invoice') must remain separate."""
    context = create_mock_context()
    ep1 = CandidateOpenLoop(
        title="Send Orbit invoice",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-15T10:00:00Z"),
        source_msgs_all=["M01"],
    )
    ep2 = CandidateOpenLoop(
        title="Review Orbit invoice",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M02",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-16T10:00:00Z"),
        source_msgs_all=["M02"],
    )
    merged = deduplicate_open_loops([ep1, ep2], context)
    assert len(merged) == 2
    assert merged[0].source_msg == "M01"
    assert merged[1].source_msg == "M02"


def test_two_related_reminders_for_same_orbit_invoice_merge():
    """Genuinely related reminders for the same Orbit invoice must merge into one episode."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Orbit Client",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P01", "text": "Please send Orbit invoice", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "P01", "text": "Did you send the Orbit invoice?", "ts": 2000, "kind": "text"},
                {"id": "M03", "chat_id": "C01", "sender_id": "P01", "text": "Checking in on Orbit invoice", "ts": 3000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    ep1 = CandidateOpenLoop(
        title="Send Orbit invoice",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-15T10:00:00Z"),
        source_msgs_all=["M01"],
    )
    ep2 = CandidateOpenLoop(
        title="Did you send the Orbit invoice?",
        type="followup",
        owner="ME",
        chat_id="C01",
        source_msg="M02",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-16T10:00:00Z"),
        source_msgs_all=["M02"],
    )
    ep3 = CandidateOpenLoop(
        title="Checking in on Orbit invoice",
        type="followup",
        owner="ME",
        chat_id="C01",
        source_msg="M03",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-17T10:00:00Z"),
        source_msgs_all=["M03"],
    )
    diag = OpenLoopExtractionDiagnostics()
    merged = deduplicate_open_loops([ep1, ep2, ep3], context, diagnostics=diag)
    assert len(merged) == 1
    assert merged[0].source_msg == "M01"
    assert "M02" in merged[0].source_msgs_all
    assert "M03" in merged[0].source_msgs_all
    assert diag.loops_deduplicated == 2


def test_timezone_metadata_validation():
    """Verify that only valid, explicitly supported timezone metadata is accepted."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "Draft due March 25", "ts": 1700000000, "kind": "text"},
            ],
        }
    }
    source_msg = chats["C01"]["messages"][0]

    # Valid timezones: accepted
    ctx_utc = create_mock_context(chats=chats, dataset_metadata={"timezone": "UTC"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_utc)
    assert due == "2025-03-25"
    assert anchor == "M01"

    ctx_kolkata = create_mock_context(chats=chats, dataset_metadata={"timezone": "Asia/Kolkata"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_kolkata)
    assert due == "2025-03-25"
    assert anchor == "M01"

    ctx_offset = create_mock_context(chats=chats, dataset_metadata={"timezone": "+05:30"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_offset)
    assert due == "2025-03-25"
    assert anchor == "M01"

    # Empty metadata: rejected
    ctx_empty = create_mock_context(chats=chats, dataset_metadata={"timezone": ""})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_empty)
    assert due is None
    assert anchor is None

    ctx_none = create_mock_context(chats=chats, dataset_metadata={"timezone": None})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_none)
    assert due is None
    assert anchor is None

    # Invalid metadata: rejected
    ctx_country = create_mock_context(chats=chats, dataset_metadata={"timezone": "India"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_country)
    assert due is None
    assert anchor is None

    ctx_invalid = create_mock_context(chats=chats, dataset_metadata={"timezone": "not-a-timezone"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_invalid)
    assert due is None
    assert anchor is None

    # Unrecognized IANA timezone names: rejected
    ctx_fake_iana = create_mock_context(chats=chats, dataset_metadata={"timezone": "Asia/NotAZone"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_fake_iana)
    assert due is None
    assert anchor is None

    ctx_fake_city = create_mock_context(chats=chats, dataset_metadata={"timezone": "America/NotARealCity"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_fake_city)
    assert due is None
    assert anchor is None

    # Invalid boundary offsets: rejected
    ctx_over_offset = create_mock_context(chats=chats, dataset_metadata={"timezone": "+14:30"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_over_offset)
    assert due is None
    assert anchor is None

    ctx_neg_over = create_mock_context(chats=chats, dataset_metadata={"timezone": "-15:00"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_neg_over)
    assert due is None
    assert anchor is None

    # Valid boundary offset: accepted
    ctx_max_offset = create_mock_context(chats=chats, dataset_metadata={"timezone": "+14:00"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_max_offset)
    assert due == "2025-03-25"
    assert anchor == "M01"

    # Unknown timezone offset -00:00: rejected
    ctx_unknown_offset = create_mock_context(chats=chats, dataset_metadata={"timezone": "-00:00"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_unknown_offset)
    assert due is None
    assert anchor is None

    # UTC numeric offset +00:00: accepted
    ctx_utc_offset = create_mock_context(chats=chats, dataset_metadata={"timezone": "+00:00"})
    due, anchor = resolve_due_date("2025-03-25", "M01", source_msg, "C01", ctx_utc_offset)
    assert due == "2025-03-25"
    assert anchor == "M01"


def test_group_pledge_cancellation_bare_phrase_rejected():
    """A group first-person pledge where sender is resolved owner cannot be dropped by bare cancel."""
    chats = {
        "C02": {
            "chat_id": "C02",
            "name": "Project Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "P01", "text": "I will submit the quarterly report tomorrow", "ts": 1000, "kind": "text"},
                {"id": "G02", "chat_id": "C02", "sender_id": "P01", "text": "Cancel that", "ts": 1100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C02"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C02",
        owner="P01",
        loop_type="commitment",
        title="Submit quarterly report",
        context=context,
    )
    # P01 is obligor, not requester; bare phrase without preemption reason leaves it open
    assert status == "open"
    assert closed_by is None

    # Same check when sender is ME in group
    chats_me = {
        "C02": {
            "chat_id": "C02",
            "name": "Project Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G10", "chat_id": "C02", "sender_id": "ME", "text": "I'll draft the architecture proposal", "ts": 2000, "kind": "text"},
                {"id": "G11", "chat_id": "C02", "sender_id": "ME", "text": "Cancel that", "ts": 2100, "kind": "text"},
            ],
        }
    }
    context_me = create_mock_context(chats=chats_me)
    source_msg_me = chats_me["C02"]["messages"][0]
    status_me, closed_by_me = verify_full_chat_closure(
        source_msg=source_msg_me,
        chat_id="C02",
        owner="ME",
        loop_type="commitment",
        title="Draft architecture proposal",
        context=context_me,
    )
    assert status_me == "open"
    assert closed_by_me is None


def test_group_pledge_cancellation_with_authorized_preemption_accepted():
    """A group first-person pledge sender reporting an authorized preemption can drop the loop."""
    chats = {
        "C02": {
            "chat_id": "C02",
            "name": "Project Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "P01", "text": "I will prepare the presentation slides", "ts": 1000, "kind": "text"},
                {"id": "G02", "chat_id": "C02", "sender_id": "P01", "text": "Cancel that, client cancelled the meeting and project was scrapped", "ts": 1100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C02"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C02",
        owner="P01",
        loop_type="commitment",
        title="Prepare presentation slides",
        context=context,
    )
    assert status == "dropped"
    assert closed_by == "G02"


def test_group_directed_assignment_requester_revocation_accepted():
    """In group chats, the requester who assigned a task to another participant can revoke it."""
    chats = {
        "C02": {
            "chat_id": "C02",
            "name": "Project Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "ME", "text": "Bob please draft the client proposal", "ts": 1000, "kind": "text"},
                {"id": "G02", "chat_id": "C02", "sender_id": "ME", "text": "Never mind, cancel that client proposal, no longer needed", "ts": 1100, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    source_msg = chats["C02"]["messages"][0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C02",
        owner="P02",
        loop_type="commitment",
        title="Draft client proposal",
        context=context,
    )
    # ME is requester; unilateral requester revocation accepted
    assert status == "dropped"
    assert closed_by == "G02"

    # Conversely, if obligor Bob says bare "Cancel that", it must be rejected
    chats_bob = {
        "C02": {
            "chat_id": "C02",
            "name": "Project Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "ME", "text": "Bob please draft the client proposal", "ts": 1000, "kind": "text"},
                {"id": "G02", "chat_id": "C02", "sender_id": "P02", "text": "Cancel that", "ts": 1100, "kind": "text"},
            ],
        }
    }
    context_bob = create_mock_context(chats=chats_bob)
    source_msg_bob = chats_bob["C02"]["messages"][0]
    status_bob, closed_by_bob = verify_full_chat_closure(
        source_msg=source_msg_bob,
        chat_id="C02",
        owner="P02",
        loop_type="commitment",
        title="Draft client proposal",
        context=context_bob,
    )
    assert status_bob == "open"
    assert closed_by_bob is None


def test_is_valid_timezone_identifier_iana_and_offsets():
    """Verify strict validation of official IANA identifiers and numeric offset bounds."""
    # Valid recognized IANA identifiers
    valid_iana = [
        "Asia/Kolkata",
        "America/New_York",
        "Europe/London",
        "Australia/Sydney",
        "Asia/Tokyo",
        "Pacific/Auckland",
        "Africa/Cairo",
        "Etc/UTC",
        "UTC",
        "GMT",
    ]
    for tz in valid_iana:
        assert is_valid_timezone_identifier(tz) is True, f"Expected {tz} to be valid"

    # Syntactically plausible but unrecognized IANA names: rejected
    invalid_iana = [
        "Asia/NotAZone",
        "America/NotARealCity",
        "Europe/FakeCity",
        "Africa/ImaginaryCity",
        "Antarctica/Atlantis",
    ]
    for tz in invalid_iana:
        assert is_valid_timezone_identifier(tz) is False, f"Expected {tz} to be rejected"

    # Valid boundary offsets: [-14:00, +14:00], excluding unknown -00:00
    valid_offsets = [
        "+14:00",
        "-14:00",
        "+00:00",
        "+13:45",
        "-11:00",
        "+05:30",
        "-08:00",
        "+0530",
        "-0800",
    ]
    for offset in valid_offsets:
        assert is_valid_timezone_identifier(offset) is True, f"Expected {offset} to be valid"
    assert is_valid_timezone_identifier("+00:00") is True

    # Invalid offsets: out of bounds, malformed, or unknown offset -00:00
    invalid_offsets = [
        "-00:00",   # Unknown offset per RFC 3339 / ISO 8601
        "-0000",
        "+14:30",   # Nonzero minute at 14h boundary
        "-14:30",   # Nonzero minute at -14h boundary
        "+14:01",   # Nonzero minute at 14h boundary
        "-14:01",   # Nonzero minute at -14h boundary
        "+15:00",   # Hour > 14
        "-15:00",   # Hour > 14
        "+16:00",
        "+14:60",   # Minute >= 60
        "+13:60",   # Minute >= 60
        "+5:30",    # Single digit hour
        "14:00",    # Missing sign
        "-14",      # Missing minutes
        "not-an-offset",
        "",
        None,
        False,
        True,
        123,
    ]
    for offset in invalid_offsets:
        assert is_valid_timezone_identifier(offset) is False, f"Expected {offset} to be rejected"
    assert is_valid_timezone_identifier("-00:00") is False


def test_action_entity_extraction_deterministic_title_token_order():
    """Verify action and entity extraction preserves title token order deterministically."""
    # Title with multiple action verbs: first token in title order is selected
    title1 = "Review and draft client proposal"
    for _ in range(20):
        assert _extract_task_action(title1) == "review"
        assert _extract_task_entity(title1) == "proposal"

    # Reversed action verbs: first token in title order is selected
    title2 = "Draft and review client proposal"
    for _ in range(20):
        assert _extract_task_action(title2) == "draft"
        assert _extract_task_entity(title2) == "proposal"

    # Title with multiple task entities: first token in title order is selected
    title3 = "Send Orbit invoice and agreement"
    for _ in range(20):
        assert _extract_task_action(title3) == "send"
        assert _extract_task_entity(title3) == "invoice"

    # Reversed task entities: first token in title order is selected
    title4 = "Send Orbit agreement and invoice"
    for _ in range(20):
        assert _extract_task_action(title4) == "send"
        assert _extract_task_entity(title4) == "agreement"


def test_episode_deduplication_stability_multiple_actions_entities():
    """Verify deduplication stability for titles containing multiple action verbs or entities."""
    context = create_mock_context()
    ep1 = CandidateOpenLoop(
        title="Review and draft client proposal",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-15T10:00:00Z"),
        source_msgs_all=["M01"],
    )
    ep2 = CandidateOpenLoop(
        title="Review and draft client proposal",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M02",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-16T10:00:00Z"),
        source_msgs_all=["M02"],
    )
    ep3 = CandidateOpenLoop(
        title="Send Orbit invoice and agreement",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M03",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-17T10:00:00Z"),
        source_msgs_all=["M03"],
    )
    ep4 = CandidateOpenLoop(
        title="Send Orbit contract and report",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M04",
        status="open",
        help=HelpAction(action="remind", remind_at="2025-01-18T10:00:00Z"),
        source_msgs_all=["M04"],
    )

    # ep1 and ep2 merge (same episode title/action/entity), ep3 and ep4 remain separate
    for _ in range(10):
        merged = deduplicate_open_loops([ep1, ep2, ep3, ep4], context)
        assert len(merged) == 3
        assert merged[0].source_msg == "M01"
        assert merged[0].source_msgs_all == ["M01", "M02"]
        assert merged[1].source_msg == "M03"
        assert merged[2].source_msg == "M04"



def test_resolve_open_loops_rejects_invalid_loop_types():
    """Verify that invalid, missing, non-string, or unsupported loop types are strictly rejected."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Priya",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "I will review the pull request by 5pm", "ts": 1700000000, "kind": "text"},
            ],
        }
    }
    context = create_mock_context(chats=chats)
    config = OpenLoopExtractionConfig(min_confidence=0.7)

    # 1. Invalid string type
    diag1 = OpenLoopExtractionDiagnostics()
    loop_invalid_str = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="invalid_type",
        is_genuine=True,
        title="Review pull request",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.9,
    )
    res1 = resolve_open_loops([loop_invalid_str], context, config, diag1)
    assert len(res1) == 0
    assert diag1.candidates_rejected == 1

    # 2. None / missing type
    diag2 = OpenLoopExtractionDiagnostics()
    loop_none = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type=None,
        is_genuine=True,
        title="Review pull request",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.9,
    )
    res2 = resolve_open_loops([loop_none], context, config, diag2)
    assert len(res2) == 0
    assert diag2.candidates_rejected == 1

    # 3. Non-string type (int, bool, list)
    diag3 = OpenLoopExtractionDiagnostics()
    loop_int = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type=123,
        is_genuine=True,
        title="Review pull request",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.9,
    )
    res3 = resolve_open_loops([loop_int], context, config, diag3)
    assert len(res3) == 0
    assert diag3.candidates_rejected == 1

    diag3_bool = OpenLoopExtractionDiagnostics()
    loop_bool = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type=False,
        is_genuine=True,
        title="Review pull request",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.9,
    )
    res3_bool = resolve_open_loops([loop_bool], context, config, diag3_bool)
    assert len(res3_bool) == 0
    assert diag3_bool.candidates_rejected == 1

    # 4. Unsupported obligation type 'request' (not in VALID_OPEN_LOOP_TYPES) is rejected
    diag_req = OpenLoopExtractionDiagnostics()
    loop_request = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="request",
        is_genuine=True,
        title="Review pull request",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.9,
    )
    res_req = resolve_open_loops([loop_request], context, config, diag_req)
    assert len(res_req) == 0
    assert diag_req.candidates_rejected == 1

    # 5. Valid types continue to resolve normally
    valid_types = ["commitment", "question", "followup", "deadline"]
    for v_type in valid_types:
        diag = OpenLoopExtractionDiagnostics()
        valid_loop = InterpretedOpenLoop(
            source_msg="M01",
            chat_id="C01",
            type=v_type,
            is_genuine=True,
            title="Review pull request",
            owner="ME",
            target_person=None,
            status="open",
            closed_by=None,
            due_date=None,
            suggested_action="assist",
            confidence=0.9,
        )
        res = resolve_open_loops([valid_loop], context, config, diag)
        assert len(res) == 1
        assert res[0].type == v_type
        assert diag.candidates_rejected == 0
        assert diag.candidates_created == 1


def test_group_chat_consecutive_turn_proximity():
    """Verify group-chat proximity rules, intervening message invalidation, and linkage fallbacks."""
    # Scenario 1: Original sender's consecutive group turn within 1,800s with preemption: accepted
    chats_consecutive = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "P01", "text": "I will prepare the presentation slides", "ts": 1000, "kind": "text"},
                {"id": "G02", "chat_id": "C02", "sender_id": "P01", "text": "Cancel that, client cancelled the meeting and project was scrapped", "ts": 1100, "kind": "text"},
            ],
        }
    }
    ctx1 = create_mock_context(chats=chats_consecutive)
    status1, closed_by1 = verify_full_chat_closure(
        source_msg=chats_consecutive["C02"]["messages"][0],
        chat_id="C02",
        owner="P01",
        loop_type="commitment",
        title="Prepare presentation slides",
        context=ctx1,
    )
    assert status1 == "dropped"
    assert closed_by1 == "G02"

    # Scenario 2: Intervening participant message before sender's preemption: proximity exception rejected
    chats_intervening = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "P01", "text": "I will prepare the presentation slides", "ts": 1000, "kind": "text"},
                # P02 intervenes
                {"id": "G02", "chat_id": "C02", "sender_id": "P02", "text": "Sounds good!", "ts": 1050, "kind": "text"},
                # P01 sends preemption reason without quote or semantic overlap with title
                {"id": "G03", "chat_id": "C02", "sender_id": "P01", "text": "Cancel that, client cancelled the meeting and project was scrapped", "ts": 1100, "kind": "text"},
            ],
        }
    }
    ctx2 = create_mock_context(chats=chats_intervening)
    status2, closed_by2 = verify_full_chat_closure(
        source_msg=chats_intervening["C02"]["messages"][0],
        chat_id="C02",
        owner="P01",
        loop_type="commitment",
        title="Prepare presentation slides",
        context=ctx2,
    )
    assert status2 == "open"
    assert closed_by2 is None

    # Scenario 3: Bystander cancellation: rejected
    chats_bystander = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "P01", "text": "I will prepare the presentation slides", "ts": 1000, "kind": "text"},
                # P02 (bystander) tries to cancel with preemption phrase
                {"id": "G02", "chat_id": "C02", "sender_id": "P02", "text": "Cancel that, client cancelled the meeting and project was scrapped", "ts": 1100, "kind": "text"},
            ],
        }
    }
    ctx3 = create_mock_context(chats=chats_bystander)
    status3, closed_by3 = verify_full_chat_closure(
        source_msg=chats_bystander["C02"]["messages"][0],
        chat_id="C02",
        owner="P01",
        loop_type="commitment",
        title="Prepare presentation slides",
        context=ctx3,
    )
    assert status3 == "open"
    assert closed_by3 is None

    # Scenario 4: Original sender's preemption after more than 1,800 seconds: proximity exception rejected
    chats_delayed = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "P01", "text": "I will prepare the presentation slides", "ts": 1000, "kind": "text"},
                # Preemption sent 1,900s later (no quote, no semantic overlap with title)
                {"id": "G02", "chat_id": "C02", "sender_id": "P01", "text": "Cancel that, client cancelled the meeting and project was scrapped", "ts": 2900, "kind": "text"},
            ],
        }
    }
    ctx4 = create_mock_context(chats=chats_delayed)
    status4, closed_by4 = verify_full_chat_closure(
        source_msg=chats_delayed["C02"]["messages"][0],
        chat_id="C02",
        owner="P01",
        loop_type="commitment",
        title="Prepare presentation slides",
        context=ctx4,
    )
    assert status4 == "open"
    assert closed_by4 is None

    # Scenario 5: Explicit semantic or quoted linkage continues to work where independently authorized
    # 5a: Quoted linkage overcomes intervening message and delay
    chats_quoted = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "P01", "text": "I will prepare the presentation slides", "ts": 1000, "kind": "text"},
                {"id": "G02", "chat_id": "C02", "sender_id": "P02", "text": "Sounds good!", "ts": 1050, "kind": "text"},
                {"id": "G03", "chat_id": "C02", "sender_id": "P01", "text": "Cancel that, client cancelled the meeting and project was scrapped", "ts": 5000, "quoted_id": "G01", "kind": "text"},
            ],
        }
    }
    ctx5a = create_mock_context(chats=chats_quoted)
    status5a, closed_by5a = verify_full_chat_closure(
        source_msg=chats_quoted["C02"]["messages"][0],
        chat_id="C02",
        owner="P01",
        loop_type="commitment",
        title="Prepare presentation slides",
        context=ctx5a,
    )
    assert status5a == "dropped"
    assert closed_by5a == "G03"

    # 5b: Semantic overlap linkage overcomes intervening message and delay
    chats_semantic = {
        "C02": {
            "chat_id": "C02",
            "name": "Team",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "G01", "chat_id": "C02", "sender_id": "P01", "text": "I will prepare the presentation slides", "ts": 1000, "kind": "text"},
                {"id": "G02", "chat_id": "C02", "sender_id": "P02", "text": "Sounds good!", "ts": 1050, "kind": "text"},
                {"id": "G03", "chat_id": "C02", "sender_id": "P01", "text": "Cancel that, client cancelled the meeting, no presentation slides needed", "ts": 5000, "kind": "text"},
            ],
        }
    }
    ctx5b = create_mock_context(chats=chats_semantic)
    status5b, closed_by5b = verify_full_chat_closure(
        source_msg=chats_semantic["C02"]["messages"][0],
        chat_id="C02",
        owner="P01",
        loop_type="commitment",
        title="Prepare presentation slides",
        context=ctx5b,
    )
    assert status5b == "dropped"
    assert closed_by5b == "G03"


# ==============================================================================
# Milestone C.3 Safeguards: Banter and Informational Status Regression Tests
# ==============================================================================

def _make_interpreted_loop(
    source_msg: str,
    chat_id: str,
    title: str,
    owner: str = "ME",
    type: str = "commitment",
    is_genuine: bool = True,
    confidence: float = 0.9,
    status: str = "open",
    closed_by: Optional[str] = None,
    due_date: Optional[str] = None,
    suggested_action: Optional[str] = None,
) -> InterpretedOpenLoop:
    return InterpretedOpenLoop(
        source_msg=source_msg,
        chat_id=chat_id,
        type=type,
        is_genuine=is_genuine,
        title=title,
        owner=owner,
        target_person=None,
        status=status,
        closed_by=closed_by,
        due_date=due_date,
        suggested_action=suggested_action,
        confidence=confidence,
    )


def test_banter_clear_playful_threat_rejected():
    """Verify clear playful threat in conversational banter is rejected as non-actionable."""
    chats = {
        "C10": {
            "chat_id": "C10",
            "name": "Social Group",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C10", "sender_id": "P01", "text": "Haha did you finish all the pizza?", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C10", "sender_id": "ME", "text": "I'll slap u if i see u 😂", "ts": 1050, "kind": "text"},
                {"id": "M03", "chat_id": "C10", "sender_id": "P01", "text": "😂😂", "ts": 1100, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C10"]["messages"][1]

    loop = _make_interpreted_loop(
        source_msg="M02",
        chat_id="C10",
        type="commitment",
        is_genuine=True,
        title="Slap ME if they see me",
        owner="ME",
        confidence=0.9,
    )

    # 1. Direct safeguard check
    assert is_non_actionable_banter(loop, source_msg, ctx) is True

    # 2. End-to-end resolver execution rejects candidate
    resolved = resolve_open_loops([loop], ctx)
    assert resolved == [], f"Expected banter to be rejected by resolver, got: {resolved}"


def test_banter_genuine_informal_commitment_retained():
    """Verify genuine informal commitment is preserved and not rejected by banter safeguard."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Work 1:1",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P01", "text": "When will you send the file?", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "I'll send it tomorrow.", "ts": 1050, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][1]

    loop = _make_interpreted_loop(
        source_msg="M02",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Send file tomorrow",
        owner="ME",
        confidence=0.9,
    )

    assert is_non_actionable_banter(loop, source_msg, ctx) is False
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1
    assert resolved[0].source_msg == "M02"
    assert resolved[0].owner == "ME"


def test_banter_legitimate_task_with_misleading_verbs_retained():
    """Verify tasks containing action words like 'kill', 'run', 'drop', or 'hit' are retained."""
    tasks = [
        ("M10", "I will kill the background process by 5pm", "Kill background process"),
        ("M11", "I will run the weekly analytics report tomorrow", "Run analytics report"),
        ("M12", "I will drop the keys at reception", "Drop keys at reception"),
        ("M13", "I will hit you up on WhatsApp tomorrow", "Contact on WhatsApp"),
        ("M14", "I will hit the gym at 6am", "Go to gym"),
    ]
    for msg_id, body, title in tasks:
        chats = {
            "C01": {
                "chat_id": "C01",
                "name": "Team",
                "is_group": False,
                "participants": ["ME", "P01"],
                "messages": [
                    {"id": msg_id, "chat_id": "C01", "sender_id": "ME", "text": body, "ts": 1000, "kind": "text"},
                ],
            }
        }
        ctx = create_mock_context(chats=chats)
        source_msg = chats["C01"]["messages"][0]

        loop = _make_interpreted_loop(
            source_msg=msg_id,
            chat_id="C01",
            type="commitment",
            is_genuine=True,
            title=title,
            owner="ME",
            confidence=0.9,
        )

        assert is_non_actionable_banter(loop, source_msg, ctx) is False, f"Failed for {body}"
        resolved = resolve_open_loops([loop], ctx)
        assert len(resolved) == 1, f"Expected task with '{body}' to be retained"


def test_banter_missing_or_ambiguous_context_defaults_to_retention():
    """Verify that if context is missing, incomplete, or ambiguous, the candidate is conservatively retained."""
    # Single isolated message in chat without surrounding laughter or conversational context
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Direct",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "ME", "text": "I'll slap u if i see u", "ts": 1000, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Slap ME if they see me",
        owner="ME",
        confidence=0.9,
    )

    # Missing framing cues -> conservative default is retention (False)
    assert is_non_actionable_banter(loop, source_msg, ctx) is False


def test_informational_status_immediate_linked_acknowledgement_rejected():
    """Verify attendance update with an immediately following, clearly linked acknowledgement in direct chat is rejected."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Direct 1:1",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P01", "text": "Due to personal work I will join the 2nd half", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "Noted, thanks for informing", "ts": 1050, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C01"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Join the second half",
        owner="P01",
        confidence=0.9,
    )

    assert is_informational_status_update(loop, source_msg, ctx) is True
    resolved = resolve_open_loops([loop], ctx)
    assert resolved == []





def test_informational_status_explicit_quoted_reply_recognized():
    """Verify attendance update acknowledged via explicit quoted reply is rejected even across intervening messages."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Chat",
            "is_group": True,
            "participants": ["ME", "P01", "P02", "P03"],
            "messages": [
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "I have doctor check up today will not be able to join today", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "P02", "text": "Who is running the sprint review?", "ts": 1100, "kind": "text"},
                {"id": "M03", "chat_id": "C03", "sender_id": "P03", "text": "I can handle it", "ts": 1150, "kind": "text"},
                {"id": "M04", "chat_id": "C03", "sender_id": "ME", "text": "Take care, get well soon", "ts": 1200, "quoted_id": "M01", "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Will not be able to join today",
        owner="P01",
        confidence=0.9,
    )

    assert is_informational_status_update(loop, source_msg, ctx) is True
    resolved = resolve_open_loops([loop], ctx)
    assert resolved == []





def test_informational_status_structured_roll_call_recognized():
    """Verify attendance message participating in an explicit structured roll-call is recognized and rejected."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Standup",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "M00", "chat_id": "C03", "sender_id": "ME", "text": "Team, please share today's availability", "ts": 900, "kind": "text"},
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "Due to personal work I will join the 2nd half", "ts": 1000, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][1]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Join second half",
        owner="P01",
        confidence=0.9,
    )

    assert is_informational_status_update(loop, source_msg, ctx) is True
    resolved = resolve_open_loops([loop], ctx)
    assert resolved == []





def test_informational_status_intervening_message_does_not_qualify():
    """Verify that an unquoted acknowledgement separated by even one intervening message does NOT qualify."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Chat",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "Due to personal work I will join the 2nd half", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "P02", "text": "Did anyone see the client email?", "ts": 1050, "kind": "text"},
                {"id": "M03", "chat_id": "C03", "sender_id": "ME", "text": "Noted, thanks for informing", "ts": 1100, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Join the second half",
        owner="P01",
        confidence=0.9,
    )

    # Intervening message breaks Route 2; no quoted_id for Route 1 -> must NOT qualify as informational
    assert is_informational_status_update(loop, source_msg, ctx) is False
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1
    assert resolved[0].source_msg == "M01"


def test_informational_status_unrelated_generic_ack_does_not_qualify():
    """Verify that an unrelated generic 'ok', 'noted', 'sure', or 'thanks' without demonstrable linkage does not qualify."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Chat",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "Due to personal work I will join the 2nd half", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "P02", "text": "Can someone check the deployment pipeline?", "ts": 1050, "kind": "text"},
                {"id": "M03", "chat_id": "C03", "sender_id": "ME", "text": "sure", "ts": 1100, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Join the second half",
        owner="P01",
        confidence=0.9,
    )

    assert is_informational_status_update(loop, source_msg, ctx) is False
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1


def test_informational_status_two_nearby_announcements_do_not_establish_roll_call():
    """Verify that two nearby attendance announcements by themselves do NOT establish a structured roll-call."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Chat",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "Due to personal work I will join the 2nd half", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "P02", "text": "I will also be joining in the 2nd half", "ts": 1050, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Join the second half",
        owner="P01",
        confidence=0.9,
    )

    # Proximity without a trigger prompt must NOT be treated as a roll-call
    assert is_informational_status_update(loop, source_msg, ctx) is False
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1


def test_informational_status_isolated_ambiguous_statement_retained():
    """Verify that an isolated attendance statement without affirmative contextual evidence is conservatively retained."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Chat",
            "is_group": True,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "Due to personal work I will join the 2nd half", "ts": 1000, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Join the second half",
        owner="P01",
        confidence=0.9,
    )

    # Isolated with no response -> conservative fallback retains
    assert is_informational_status_update(loop, source_msg, ctx) is False
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1


def test_informational_status_combined_with_deliverable_remains_actionable():
    """Verify attendance combined with a concrete deliverable or request remains an actionable open loop."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Chat",
            "is_group": True,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "I will join the 2nd half and present the Q3 deck", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "ME", "text": "Noted, thanks for informing", "ts": 1050, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Present Q3 deck in second half",
        owner="P01",
        confidence=0.9,
    )

    # Deliverable present -> NOT purely informational
    assert is_informational_status_update(loop, source_msg, ctx) is False
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1
    assert resolved[0].source_msg == "M01"
    assert resolved[0].owner == "P01"


def test_safeguards_preserve_approved_ownership_resolution():
    """Verify that banter and informational status safeguards do not alter approved ownership resolution."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Work 1:1",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C01", "sender_id": "P01", "text": "Can you review the draft proposal?", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C01", "sender_id": "ME", "text": "I will review it by tomorrow", "ts": 1050, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)

    # Loop 1: directed request to ME -> resolved to ME
    loop1 = _make_interpreted_loop(source_msg="M01", chat_id="C01", type="commitment", is_genuine=True, title="Review draft proposal", owner="ME", confidence=0.9)
    # Loop 2: first-person pledge by ME -> resolved to ME
    loop2 = _make_interpreted_loop(source_msg="M02", chat_id="C01", type="commitment", is_genuine=True, title="Review draft by tomorrow", owner="ME", confidence=0.9)

    resolved = resolve_open_loops([loop1, loop2], ctx)
    assert len(resolved) == 2
    assert resolved[0].owner == "ME"
    assert resolved[1].owner == "ME"


def test_banter_playful_threat_with_task_noun_not_automatically_retained():
    """Verify a playful threat containing a task-like noun (ticket, pr, report) is not retained solely because of that noun."""
    chats = {
        "C10": {
            "chat_id": "C10",
            "name": "Dev Chat",
            "is_group": False,
            "participants": ["ME", "P01"],
            "messages": [
                {"id": "M01", "chat_id": "C10", "sender_id": "P01", "text": "I rejected your PR haha", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C10", "sender_id": "ME", "text": "I will murder you for this ticket lol", "ts": 1050, "kind": "text"},
                {"id": "M03", "chat_id": "C10", "sender_id": "P01", "text": "hahaha try me", "ts": 1100, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C10"]["messages"][1]

    loop = _make_interpreted_loop(
        source_msg="M02",
        chat_id="C10",
        type="commitment",
        is_genuine=True,
        title="Murder counterparty for ticket",
        owner="ME",
        confidence=0.9,
    )

    # Must be recognized as banter and rejected despite the presence of task nouns ('ticket', 'pr')
    assert is_non_actionable_banter(loop, source_msg, ctx) is True
    resolved = resolve_open_loops([loop], ctx)
    assert resolved == []


def test_banter_genuine_informal_commitment_without_formal_task_noun_retained():
    """Verify genuine informal commitment lacking a formal task noun is retained."""
    tasks = [
        ("M20", "I will handle it tomorrow", "Handle it tomorrow"),
        ("M21", "I will look into it later", "Look into it later"),
        ("M22", "I will take care of it tonight", "Take care of it tonight"),
    ]
    for msg_id, body, title in tasks:
        chats = {
            "C01": {
                "chat_id": "C01",
                "name": "Direct 1:1",
                "is_group": False,
                "participants": ["ME", "P01"],
                "messages": [
                    {"id": msg_id, "chat_id": "C01", "sender_id": "ME", "text": body, "ts": 1000, "kind": "text"},
                ],
            }
        }
        ctx = create_mock_context(chats=chats)
        source_msg = chats["C01"]["messages"][0]

        loop = _make_interpreted_loop(
            source_msg=msg_id,
            chat_id="C01",
            type="commitment",
            is_genuine=True,
            title=title,
            owner="ME",
            confidence=0.9,
        )

        assert is_non_actionable_banter(loop, source_msg, ctx) is False, f"Failed for {body}"
        resolved = resolve_open_loops([loop], ctx)
        assert len(resolved) == 1, f"Expected informal commitment '{body}' to be retained"


def test_informational_status_immediate_ack_from_intended_counterparty_in_group_qualifies():
    """Verify immediate acknowledgement from an established intended counterparty in a group chat qualifies."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Chat",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "@ME Due to personal work I will join the 2nd half", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "ME", "text": "Noted, thanks for informing", "ts": 1050, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Join the second half",
        owner="P01",
        confidence=0.9,
    )

    # ME is explicitly addressed, so ME is the intended counterparty and immediate acknowledgement qualifies
    assert is_informational_status_update(loop, source_msg, ctx) is True
    resolved = resolve_open_loops([loop], ctx)
    assert resolved == []





def test_informational_status_immediate_ack_from_unrelated_group_participant_does_not_qualify():
    """Verify immediate acknowledgement from an unrelated group participant does not qualify."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Chat",
            "is_group": True,
            "participants": ["ME", "P01", "P02", "P03"],
            "messages": [
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "@P02 I will be delayed by 15 mins", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "P03", "text": "Noted, thanks for letting us know", "ts": 1050, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Delayed by 15 mins",
        owner="P01",
        confidence=0.9,
    )

    # P03 is NOT the intended counterparty (which is P02) -> immediate response route does NOT qualify
    assert is_informational_status_update(loop, source_msg, ctx) is False
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1


def test_informational_status_missing_recipient_identity_in_group_defaults_to_retention():
    """Verify missing recipient identity in a group broadcast defaults to retention."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Chat",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "M01", "chat_id": "C03", "sender_id": "P01", "text": "Due to personal work I will join the 2nd half", "ts": 1000, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "ME", "text": "Noted, thanks for informing", "ts": 1050, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][0]

    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Join the second half",
        owner="P01",
        confidence=0.9,
    )

    # General broadcast without grounded intended recipient -> immediate response route does NOT qualify
    assert is_informational_status_update(loop, source_msg, ctx) is False
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1


def test_informational_status_unrelated_intervening_conversation_breaks_roll_call_linkage():
    """Verify an unrelated intervening conversation breaks apparent roll-call linkage and retains candidate."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Standup",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {"id": "M00", "chat_id": "C03", "sender_id": "ME", "text": "Team, please share today's availability", "ts": 900, "kind": "text"},
                {"id": "M01", "chat_id": "C03", "sender_id": "P02", "text": "Did anyone review the PR for the auth service?", "ts": 950, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "P01", "text": "Due to personal work I will join the 2nd half", "ts": 1000, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][2]

    loop = _make_interpreted_loop(
        source_msg="M02",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Join second half",
        owner="P01",
        confidence=0.9,
    )

    # Intervening unrelated discussion breaks the roll-call thread -> retain candidate
    assert is_informational_status_update(loop, source_msg, ctx) is False
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1


def test_informational_status_continuous_unbroken_roll_call_exchange():
    """Verify candidate participating in an unbroken roll-call exchange with multiple peers is recognized."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "name": "Team Standup",
            "is_group": True,
            "participants": ["ME", "P01", "P02", "P03"],
            "messages": [
                {"id": "M00", "chat_id": "C03", "sender_id": "ME", "text": "Daily standup roll call: mark your attendance", "ts": 900, "kind": "text"},
                {"id": "M01", "chat_id": "C03", "sender_id": "P02", "text": "Present, in office today", "ts": 920, "kind": "text"},
                {"id": "M02", "chat_id": "C03", "sender_id": "P03", "text": "WFH today", "ts": 940, "kind": "text"},
                {"id": "M03", "chat_id": "C03", "sender_id": "P01", "text": "Due to personal work I will join the 2nd half", "ts": 960, "kind": "text"},
            ],
        }
    }
    ctx = create_mock_context(chats=chats)
    source_msg = chats["C03"]["messages"][3]

    loop = _make_interpreted_loop(
        source_msg="M03",
        chat_id="C03",
        type="commitment",
        is_genuine=True,
        title="Join second half",
        owner="P01",
        confidence=0.9,
    )

    assert is_informational_status_update(loop, source_msg, ctx) is True
    resolved = resolve_open_loops([loop], ctx)
    assert resolved == []


def test_named_assignment_resolves_via_sender_names():
    """Verify group chat named assignment (e.g. @sudar) resolves to participant ID via sender_names."""
    chats = {
        "C03": {
            "chat_id": "C03",
            "is_group": True,
            "participants": ["ME", "P24", "P10"],
            "messages": [
                {
                    "id": "M01",
                    "chat_id": "C03",
                    "sender_id": "P24",
                    "sender_name": "Garima",
                    "body": "Ad ID : 2229028@sudar plz remove the deadline urgent.",
                    "ts": 1000,
                    "kind": "text",
                },
                {
                    "id": "M02",
                    "chat_id": "C03",
                    "sender_id": "P10",
                    "sender_name": "SUDAR",
                    "body": "Done",
                    "ts": 1050,
                    "kind": "text",
                },
            ],
        }
    }
    participants = {
        "ME": {
            "id": "ME",
            "is_silent": False,
            "sender_names": [],
            "direct_chat_names": [],
            "chat_ids": {"C03"},
        },
        "P10": {
            "id": "P10",
            "is_silent": False,
            "sender_names": ["SUDAR"],
            "direct_chat_names": ["SUDAR"],
            "chat_ids": {"C03"},
        },
        "P24": {
            "id": "P24",
            "is_silent": False,
            "sender_names": ["Garima"],
            "direct_chat_names": [],
            "chat_ids": {"C03"},
        },
    }
    ctx = create_mock_context(chats=chats, participants=participants)
    source_msg = chats["C03"]["messages"][0]

    # Test direct owner resolution
    owner = resolve_loop_owner(None, "C03", source_msg, ctx)
    assert owner == "P10", f"Expected 'P10', got '{owner}'"

    # Test end-to-end open loop resolution including closure
    loop = _make_interpreted_loop(
        source_msg="M01",
        chat_id="C03",
        type="deadline",
        is_genuine=True,
        title="Remove deadline for Ad ID 2229028",
        owner="P10",
        status="done",
        closed_by="M02",
        confidence=0.95,
    )
    resolved = resolve_open_loops([loop], ctx)
    assert len(resolved) == 1
    assert resolved[0].owner == "P10"
    assert resolved[0].status == "done"
    assert resolved[0].closed_by == "M02"


def test_named_assignment_resolves_via_direct_chat_names():
    """Verify group chat named assignment resolves alias from direct_chat_names case-insensitively."""
    chats = {
        "C05": {
            "chat_id": "C05",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
            "messages": [
                {
                    "id": "M10",
                    "chat_id": "C05",
                    "sender_id": "P01",
                    "body": "Hey Rohan, please review the proposal by tomorrow.",
                    "ts": 2000,
                    "kind": "text",
                }
            ],
        }
    }
    participants = {
        "ME": {
            "id": "ME",
            "is_silent": False,
            "sender_names": [],
            "direct_chat_names": [],
            "chat_ids": {"C05"},
        },
        "P01": {
            "id": "P01",
            "is_silent": False,
            "sender_names": [],
            "direct_chat_names": [],
            "chat_ids": {"C05"},
        },
        "P02": {
            "id": "P02",
            "is_silent": False,
            "sender_names": ["R. Sharma"],
            "direct_chat_names": ["Rohan"],
            "chat_ids": {"C05"},
        },
    }
    ctx = create_mock_context(chats=chats, participants=participants)
    source_msg = chats["C05"]["messages"][0]

    owner = resolve_loop_owner(None, "C05", source_msg, ctx)
    assert owner == "P02"


def test_strict_immediate_assignee_response_succeeds():
    """Verify an assigned group participant replying 'Done' immediately closes the loop."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Team Dev",
            "is_group": True,
            "participants": ["ME", "P01", "P10"],
            "messages": [
                {
                    "id": "M01",
                    "chat_id": "C01",
                    "sender_id": "P01",
                    "body": "@sudar please fix the login bug",
                    "ts": 1000,
                    "kind": "text",
                },
                {
                    "id": "M02",
                    "chat_id": "C01",
                    "sender_id": "P10",
                    "body": "Done",
                    "ts": 1050,
                    "kind": "text",
                },
            ],
        }
    }
    participants = {
        "ME": {"id": "ME", "is_silent": False, "sender_names": [], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P01": {"id": "P01", "is_silent": False, "sender_names": ["Alice"], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P10": {"id": "P10", "is_silent": False, "sender_names": ["Sudar"], "direct_chat_names": [], "chat_ids": {"C01"}},
    }
    ctx = create_mock_context(chats=chats, participants=participants)
    source_msg = chats["C01"]["messages"][0]

    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P10",
        loop_type="commitment",
        title="Fix the login bug",
        context=ctx,
    )
    assert status == "done"
    assert closed_by == "M02"


def test_intervening_unrelated_message_prevents_proximity_closure():
    """Verify one unrelated intervening message prevents proximity-only bare closure."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Team Dev",
            "is_group": True,
            "participants": ["ME", "P01", "P02", "P10"],
            "messages": [
                {
                    "id": "M01",
                    "chat_id": "C01",
                    "sender_id": "P01",
                    "body": "@sudar please fix the login bug",
                    "ts": 1000,
                    "kind": "text",
                },
                {
                    "id": "M02",
                    "chat_id": "C01",
                    "sender_id": "P02",
                    "body": "Are we having lunch together today?",
                    "ts": 1050,
                    "kind": "text",
                },
                {
                    "id": "M03",
                    "chat_id": "C01",
                    "sender_id": "P10",
                    "body": "Done",
                    "ts": 1100,
                    "kind": "text",
                },
            ],
        }
    }
    participants = {
        "ME": {"id": "ME", "is_silent": False, "sender_names": [], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P01": {"id": "P01", "is_silent": False, "sender_names": ["Alice"], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P02": {"id": "P02", "is_silent": False, "sender_names": ["Bob"], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P10": {"id": "P10", "is_silent": False, "sender_names": ["Sudar"], "direct_chat_names": [], "chat_ids": {"C01"}},
    }
    ctx = create_mock_context(chats=chats, participants=participants)
    source_msg = chats["C01"]["messages"][0]

    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P10",
        loop_type="commitment",
        title="Fix the login bug",
        context=ctx,
    )
    assert status == "open"
    assert closed_by is None


def test_intervening_source_sender_message_prevents_proximity_closure():
    """Verify an intervening message from the source sender also prevents proximity-only bare closure."""
    chats = {
        "C01": {
            "chat_id": "C01",
            "name": "Team Dev",
            "is_group": True,
            "participants": ["ME", "P01", "P10"],
            "messages": [
                {
                    "id": "M01",
                    "chat_id": "C01",
                    "sender_id": "P01",
                    "body": "@sudar please fix the login bug",
                    "ts": 1000,
                    "kind": "text",
                },
                {
                    "id": "M02",
                    "chat_id": "C01",
                    "sender_id": "P01",
                    "body": "Also check the staging server logs",
                    "ts": 1050,
                    "kind": "text",
                },
                {
                    "id": "M03",
                    "chat_id": "C01",
                    "sender_id": "P10",
                    "body": "Done",
                    "ts": 1100,
                    "kind": "text",
                },
            ],
        }
    }
    participants = {
        "ME": {"id": "ME", "is_silent": False, "sender_names": [], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P01": {"id": "P01", "is_silent": False, "sender_names": ["Alice"], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P10": {"id": "P10", "is_silent": False, "sender_names": ["Sudar"], "direct_chat_names": [], "chat_ids": {"C01"}},
    }
    ctx = create_mock_context(chats=chats, participants=participants)
    source_msg = chats["C01"]["messages"][0]

    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P10",
        loop_type="commitment",
        title="Fix the login bug",
        context=ctx,
    )
    assert status == "open"
    assert closed_by is None


def test_non_adjacent_explicit_quote_and_semantic_linkage_closure():
    """Verify explicit quote linkage and semantic overlap still close loops when messages are not adjacent."""
    # 1. Explicit quote linkage over intervening messages
    chats_quote = {
        "C01": {
            "chat_id": "C01",
            "name": "Team Dev",
            "is_group": True,
            "participants": ["ME", "P01", "P02", "P10"],
            "messages": [
                {
                    "id": "M01",
                    "chat_id": "C01",
                    "sender_id": "P01",
                    "body": "@sudar please fix the login bug",
                    "ts": 1000,
                    "kind": "text",
                },
                {
                    "id": "M02",
                    "chat_id": "C01",
                    "sender_id": "P02",
                    "body": "Are we having lunch together today?",
                    "ts": 1050,
                    "kind": "text",
                },
                {
                    "id": "M03",
                    "chat_id": "C01",
                    "sender_id": "P10",
                    "body": "Done",
                    "quoted_id": "M01",
                    "ts": 1100,
                    "kind": "text",
                },
            ],
        }
    }
    participants = {
        "ME": {"id": "ME", "is_silent": False, "sender_names": [], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P01": {"id": "P01", "is_silent": False, "sender_names": ["Alice"], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P02": {"id": "P02", "is_silent": False, "sender_names": ["Bob"], "direct_chat_names": [], "chat_ids": {"C01"}},
        "P10": {"id": "P10", "is_silent": False, "sender_names": ["Sudar"], "direct_chat_names": [], "chat_ids": {"C01"}},
    }
    ctx_quote = create_mock_context(chats=chats_quote, participants=participants)
    status_quote, closed_by_quote = verify_full_chat_closure(
        source_msg=chats_quote["C01"]["messages"][0],
        chat_id="C01",
        owner="P10",
        loop_type="commitment",
        title="Fix the login bug",
        context=ctx_quote,
    )
    assert status_quote == "done"
    assert closed_by_quote == "M03"

    # 2. Semantic linkage over intervening messages
    chats_semantic = {
        "C01": {
            "chat_id": "C01",
            "name": "Team Dev",
            "is_group": True,
            "participants": ["ME", "P01", "P02", "P10"],
            "messages": [
                {
                    "id": "M01",
                    "chat_id": "C01",
                    "sender_id": "P01",
                    "body": "@sudar please fix the login bug",
                    "ts": 1000,
                    "kind": "text",
                },
                {
                    "id": "M02",
                    "chat_id": "C01",
                    "sender_id": "P02",
                    "body": "Are we having lunch together today?",
                    "ts": 1050,
                    "kind": "text",
                },
                {
                    "id": "M04",
                    "chat_id": "C01",
                    "sender_id": "P10",
                    "body": "The login bug is resolved and tested now",
                    "ts": 1200,
                    "kind": "text",
                },
            ],
        }
    }
    ctx_semantic = create_mock_context(chats=chats_semantic, participants=participants)
    status_sem, closed_by_sem = verify_full_chat_closure(
        source_msg=chats_semantic["C01"]["messages"][0],
        chat_id="C01",
        owner="P10",
        loop_type="commitment",
        title="Fix the login bug",
        context=ctx_semantic,
    )
    assert status_sem == "done"
    assert closed_by_sem == "M04"


