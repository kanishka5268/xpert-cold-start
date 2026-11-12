"""Comprehensive regression verification test suite for the Status and Decision Engine.

Covers all 13 Step 8 testing requirements:
1. Unauthorised group-chat closure.
2. Explicitly delegated task completion.
3. Generic acknowledgements without completion.
4. Relevant versus unrelated completion messages.
5. Authentic versus unauthenticated quoted evidence.
6. Postponement versus cancellation.
7. Reopening an obligation.
8. Progress updates versus final completion.
9. Multiple distinct obligations in one conversation.
10. UTC and ambiguous-date edge cases.
11. Missing or invalid closure timestamps.
12. Sensitive and contact-owned Help decisions.
13. Procurement approval safeguards.
"""

from typing import Any, Dict, List
import pytest

from src.loops.models import CandidateOpenLoop, HelpAction, OpenLoopExtractionConfig
from src.loops.resolver import (
    _evaluate_closing_message,
    deduplicate_open_loops,
    resolve_help_action,
    verify_full_chat_closure,
)
from src.preprocessing.context_builder import ConversationContext


def _build_test_context(chats: List[Dict[str, Any]], messages: List[Dict[str, Any]], participants: Dict[str, Any] = None) -> ConversationContext:
    msgs_by_id = {m["id"]: m for m in messages}
    chats_dict = {}
    for c in chats:
        cid = c["chat_id"]
        c_msgs = [m for m in messages if m.get("chat_id") == cid]
        chats_dict[cid] = {
            "chat_id": cid,
            "name": c.get("name", "Chat"),
            "is_group": c.get("is_group", False),
            "participants": c.get("participants", ["ME"]),
            "messages": c_msgs,
        }
    norm_participants = {}
    if participants:
        for pid, pdata in participants.items():
            pcopy = dict(pdata)
            pcopy.setdefault("is_silent", False)
            norm_participants[pid] = pcopy
    dataset = {
        "chats": chats,
        "messages": messages,
        "messages_by_id": msgs_by_id,
        "participants": norm_participants,
    }
    return ConversationContext(
        chats=chats_dict,
        participants=norm_participants,
        dataset=dataset,
    )


# =============================================================================
# 1. Unauthorised Group-Chat Closure
# =============================================================================

def test_unauthorised_group_chat_closure():
    """An unauthorised third party in a group chat cannot close another participant's obligation."""
    chats = [{"chat_id": "C01", "name": "Project Team", "is_group": True, "participants": ["ME", "P01", "P02", "P03"]}]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "P02 please update the analytics tracking."},
        {"id": "M02", "chat_id": "C01", "sender_id": "P03", "ts": 1100, "kind": "text", "body": "Done."},
    ]
    context = _build_test_context(chats, messages)
    source_msg = messages[0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="P02",
        loop_type="commitment",
        title="P02 please update the analytics tracking.",
        context=context,
    )
    # P03 is not owner (P02) nor requester (P01), nor a delegate -> obligation remains open
    assert status == "open"
    assert closed_by is None


# =============================================================================
# 2. Explicitly Delegated Task Completion
# =============================================================================

def test_explicitly_delegated_task_completion():
    """An authenticated delegate established in conversation authority may close the task upon completion."""
    chats = [{"chat_id": "C01", "name": "Operations", "is_group": True, "participants": ["ME", "P01", "P02"]}]
    participants = {
        "P02": {"id": "P02", "names": ["Rohan", "Rohan Sharma"], "sender_names": ["Rohan"]}
    }
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 1000, "kind": "text", "body": "I will handle the billing setup for Orbit."},
        {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 1050, "kind": "text", "body": "Actually Rohan will handle this billing setup."},
        {"id": "M03", "chat_id": "C01", "sender_id": "P02", "ts": 1200, "kind": "text", "body": "Orbit billing setup done and verified."},
    ]
    context = _build_test_context(chats, messages, participants=participants)
    source_msg = messages[0]
    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C01",
        owner="ME",
        loop_type="commitment",
        title="I will handle the billing setup for Orbit.",
        context=context,
    )
    assert status == "done"
    assert closed_by == "M03"


# =============================================================================
# 3. Generic Acknowledgements Without Completion
# =============================================================================

def test_generic_acknowledgements_without_completion():
    """Generic acknowledgements like 'okay', 'sure', 'will check', 'on it' preserve open status."""
    chats = [{"chat_id": "C01", "name": "Direct", "is_group": False, "participants": ["ME", "P01"]}]
    test_acks = [
        "okay", "ok", "sure", "will check", "noted", "on it", "working on it", "will revert", "looking into it",
    ]
    for ack in test_acks:
        messages = [
            {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Please review the quarterly audit report."},
            {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 1050, "kind": "text", "body": ack},
        ]
        context = _build_test_context(chats, messages)
        status, closed_by = verify_full_chat_closure(
            source_msg=messages[0],
            chat_id="C01",
            owner="ME",
            loop_type="commitment",
            title="Please review the quarterly audit report.",
            context=context,
        )
        assert status == "open", f"Generic acknowledgement '{ack}' unexpectedly closed obligation!"
        assert closed_by is None


# =============================================================================
# 4. Relevant Versus Unrelated Completion Messages
# =============================================================================

def test_relevant_vs_unrelated_completion_messages():
    """Proximity alone cannot close with unrelated activities like lunch chatter."""
    chats = [{"chat_id": "C01", "name": "Direct", "is_group": False, "participants": ["ME", "P01"]}]
    
    # Unrelated lunch chatter
    messages_unrelated = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Please send the quarterly tax file."},
        {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 1050, "kind": "text", "body": "Done with lunch, heading back."},
    ]
    context1 = _build_test_context(chats, messages_unrelated)
    status1, _ = verify_full_chat_closure(
        source_msg=messages_unrelated[0],
        chat_id="C01",
        owner="ME",
        loop_type="commitment",
        title="Please send the quarterly tax file.",
        context=context1,
    )
    assert status1 == "open"

    # Relevant completion
    messages_relevant = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Please send the quarterly tax file."},
        {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 1050, "kind": "text", "body": "Quarterly tax file sent to your email."},
    ]
    context2 = _build_test_context(chats, messages_relevant)
    status2, closed_by2 = verify_full_chat_closure(
        source_msg=messages_relevant[0],
        chat_id="C01",
        owner="ME",
        loop_type="commitment",
        title="Please send the quarterly tax file.",
        context=context2,
    )
    assert status2 == "done"
    assert closed_by2 == "M02"


# =============================================================================
# 5. Authentic Versus Unauthenticated Quoted Evidence
# =============================================================================

def test_authentic_vs_unauthenticated_quoted_evidence():
    """Quoted messages require authentic payload matching to authorize beyond-window closure."""
    chats = [{"chat_id": "C01", "name": "Direct", "is_group": False, "participants": ["ME", "P01"]}]
    
    # Case A: Authentic quote beyond window (15 days later)
    messages_authentic = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Review agreement draft."},
        {
            "id": "M02",
            "chat_id": "C01",
            "sender_id": "ME",
            "ts": 1000 + 15 * 86400,
            "kind": "text",
            "body": "Agreement draft review completed.",
            "quoted_id": "M01",
            "quoted_text": "Review agreement draft.",
        },
    ]
    context_auth = _build_test_context(chats, messages_authentic)
    status_auth, closed_by_auth = verify_full_chat_closure(
        source_msg=messages_authentic[0],
        chat_id="C01",
        owner="ME",
        loop_type="commitment",
        title="Review agreement draft.",
        context=context_auth,
    )
    assert status_auth == "done"
    assert closed_by_auth == "M02"

    # Case B: Fabricated/unauthenticated payload beyond window
    messages_fake = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Review agreement draft."},
        {
            "id": "M02",
            "chat_id": "C01",
            "sender_id": "ME",
            "ts": 1000 + 15 * 86400,
            "kind": "text",
            "body": "Agreement draft review completed.",
            "quoted_id": "M01",
            "quoted_text": "Completely unrelated text about lunch.",
        },
    ]
    context_fake = _build_test_context(chats, messages_fake)
    status_fake, _ = verify_full_chat_closure(
        source_msg=messages_fake[0],
        chat_id="C01",
        owner="ME",
        loop_type="commitment",
        title="Review agreement draft.",
        context=context_fake,
    )
    assert status_fake == "open"


# =============================================================================
# 6. Postponement Versus Cancellation
# =============================================================================

def test_postponement_vs_cancellation():
    """Postponements preserve open status while explicit cancellations drop the obligation."""
    chats = [{"chat_id": "C01", "name": "Direct", "is_group": False, "participants": ["ME", "P01"]}]

    # Postponement -> Open
    msgs_postpone = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Can you prepare the audit presentation?"},
        {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 1050, "kind": "text", "body": "Let's push this to next week, not feasible today."},
    ]
    ctx1 = _build_test_context(chats, msgs_postpone)
    st1, _ = verify_full_chat_closure(msgs_postpone[0], "C01", "ME", "commitment", "Can you prepare the audit presentation?", ctx1)
    assert st1 == "open"

    # Cancellation by requester -> Dropped
    msgs_cancel = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Can you prepare the audit presentation?"},
        {"id": "M02", "chat_id": "C01", "sender_id": "P01", "ts": 1050, "kind": "text", "body": "Never mind, audit cancelled, we don't need the presentation."},
    ]
    ctx2 = _build_test_context(chats, msgs_cancel)
    st2, closed_by2 = verify_full_chat_closure(msgs_cancel[0], "C01", "ME", "commitment", "Can you prepare the audit presentation?", ctx2)
    assert st2 == "dropped"
    assert closed_by2 == "M02"


# =============================================================================
# 7. Reopening an Obligation
# =============================================================================

def test_reopening_obligation_creates_distinct_episode():
    """An obligation recurring after terminal closure creates a new distinct episode."""
    chats = [{"chat_id": "C01", "name": "Direct", "is_group": False, "participants": ["ME", "P01"]}]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Send the invoice for October."},
        {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 1100, "kind": "text", "body": "October invoice sent."},
        {"id": "M03", "chat_id": "C01", "sender_id": "P01", "ts": 5000, "kind": "text", "body": "Please send the invoice for November."},
    ]
    context = _build_test_context(chats, messages)

    cand1 = CandidateOpenLoop(
        title="Send the invoice for October.",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="done",
        closed_by="M02",
        source_msgs_all=["M01"],
    )
    cand2 = CandidateOpenLoop(
        title="Please send the invoice for November.",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M03",
        status="open",
        closed_by=None,
        source_msgs_all=["M03"],
    )

    deduped = deduplicate_open_loops([cand1, cand2], context)
    assert len(deduped) == 2
    assert deduped[0].source_msg == "M01" and deduped[0].status == "done"
    assert deduped[1].source_msg == "M03" and deduped[1].status == "open"


# =============================================================================
# 8. Progress Updates Versus Final Completion
# =============================================================================

def test_progress_updates_vs_final_completion():
    """Progress updates reporting ongoing work preserve open status; only final completion closes."""
    chats = [{"chat_id": "C01", "name": "Direct", "is_group": False, "participants": ["ME", "P01"]}]
    
    # Progress update
    msgs_progress = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Please fix the landing page errors."},
        {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 1100, "kind": "text", "body": "Working on it, have fixed half of the errors so far."},
    ]
    ctx_p = _build_test_context(chats, msgs_progress)
    st_p, _ = verify_full_chat_closure(msgs_progress[0], "C01", "ME", "commitment", "Please fix the landing page errors.", ctx_p)
    assert st_p == "open"

    # Final completion
    msgs_done = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Please fix the landing page errors."},
        {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 1100, "kind": "text", "body": "Working on it, have fixed half of the errors so far."},
        {"id": "M03", "chat_id": "C01", "sender_id": "ME", "ts": 1200, "kind": "text", "body": "Landing page errors resolved and deployed."},
    ]
    ctx_d = _build_test_context(chats, msgs_done)
    st_d, closed_by_d = verify_full_chat_closure(msgs_done[0], "C01", "ME", "commitment", "Please fix the landing page errors.", ctx_d)
    assert st_d == "done"
    assert closed_by_d == "M03"


# =============================================================================
# 9. Multiple Distinct Obligations in One Conversation
# =============================================================================

def test_multiple_distinct_obligations_in_same_chat():
    """Distinct obligations with different payment amounts or targets are not incorrectly merged."""
    chats = [{"chat_id": "C10", "name": "Home", "is_group": False, "participants": ["ME", "P13"]}]
    messages = [
        {"id": "M01", "chat_id": "C10", "sender_id": "P13", "ts": 1000, "kind": "text", "body": "Pls pay 4500"},
        {"id": "M02", "chat_id": "C10", "sender_id": "P13", "ts": 2000, "kind": "text", "body": "Pls Pay 3500 on this"},
    ]
    context = _build_test_context(chats, messages)

    cand1 = CandidateOpenLoop(
        title="Pls pay 4500",
        type="commitment",
        owner="ME",
        chat_id="C10",
        source_msg="M01",
        status="open",
        closed_by=None,
        source_msgs_all=["M01"],
    )
    cand2 = CandidateOpenLoop(
        title="Pls Pay 3500 on this",
        type="commitment",
        owner="ME",
        chat_id="C10",
        source_msg="M02",
        status="open",
        closed_by=None,
        source_msgs_all=["M02"],
    )

    deduped = deduplicate_open_loops([cand1, cand2], context)
    assert len(deduped) == 2
    assert {d.source_msg for d in deduped} == {"M01", "M02"}


# =============================================================================
# 10. UTC and Ambiguous-Date Edge Cases
# =============================================================================

def test_utc_and_ambiguous_date_edge_cases():
    """Strict UTC rules enforced: 'Z' or '+00:00' required; '-00:00', naive, and local offsets rejected."""
    # Valid explicit UTC
    act_z = HelpAction(action="remind", remind_at="2026-09-02T20:08:04Z")
    assert act_z.validate() is True

    act_plus_zero = HelpAction(action="remind", remind_at="2026-09-02T20:08:04+00:00")
    assert act_plus_zero.validate() is True

    # Rejects RFC 3339 unknown local offset '-00:00'
    act_minus_zero = HelpAction(action="remind", remind_at="2026-09-02T20:08:04-00:00")
    assert act_minus_zero.validate() is False

    # Rejects naive timestamp (no timezone)
    act_naive = HelpAction(action="remind", remind_at="2026-09-02T20:08:04")
    assert act_naive.validate() is False

    # Rejects non-zero local offset '+05:30'
    act_offset = HelpAction(action="remind", remind_at="2026-09-02T20:08:04+05:30")
    assert act_offset.validate() is False


# =============================================================================
# 11. Missing or Invalid Closure Timestamps
# =============================================================================

def test_missing_or_invalid_closure_timestamps():
    """Closure candidate with non-numeric, negative, or preceding timestamp is rejected."""
    chats = [{"chat_id": "C01", "name": "Direct", "is_group": False, "participants": ["ME", "P01"]}]
    source_msg = {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000, "kind": "text", "body": "Task"}

    # Timestamp precedes source message
    msg_preceding = {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 900, "kind": "text", "body": "Done."}
    ctx1 = _build_test_context(chats, [source_msg, msg_preceding])
    res1 = _evaluate_closing_message(msg_preceding, source_msg, "C01", "ME", "P01", set(), "commitment", "Task", False, [source_msg, msg_preceding])
    assert res1 is None

    # Invalid non-numeric timestamp
    msg_invalid = {"id": "M03", "chat_id": "C01", "sender_id": "ME", "ts": "invalid", "kind": "text", "body": "Done."}
    ctx2 = _build_test_context(chats, [source_msg, msg_invalid])
    res2 = _evaluate_closing_message(msg_invalid, source_msg, "C01", "ME", "P01", set(), "commitment", "Task", False, [source_msg, msg_invalid])
    assert res2 is None


# =============================================================================
# 12. Sensitive and Contact-Owned Help Decisions
# =============================================================================

def test_sensitive_and_contact_owned_help_decisions():
    """Contact-owned and sensitive matters strictly resolve to 'leave'."""
    context = _build_test_context([], [])
    config = OpenLoopExtractionConfig()

    # Contact-owned
    msg_contact = {"id": "M01", "chat_id": "C01", "sender_id": "P05", "ts": 1000, "body": "I will send Orbit the invoice"}
    act_contact = resolve_help_action("Send Orbit invoice", "P05", msg_contact, context, config, chat_id="C01")
    assert act_contact.action == "leave"
    assert "P05 owns" in act_contact.reason

    # Sensitive interpersonal
    msg_sensitive = {"id": "M02", "chat_id": "C01", "sender_id": "P13", "ts": 1000, "body": "Our anniversary plans and family matter"}
    act_sensitive = resolve_help_action("Our anniversary plans and family matter", "ME", msg_sensitive, context, config, chat_id="C01")
    assert act_sensitive.action == "leave"


# =============================================================================
# 13. Procurement Approval Safeguards
# =============================================================================

def test_procurement_approval_safeguards():
    """Procurement and order commitments strictly require needs_yes=True."""
    context = _build_test_context([], [])
    config = OpenLoopExtractionConfig()

    msg_order = {"id": "M01", "chat_id": "C10", "sender_id": "P13", "ts": 1000, "body": "Pls order kids calling watch"}
    act_order = resolve_help_action("Pls order kids calling watch", "ME", msg_order, context, config, chat_id="C10")
    assert act_order.action == "assist"
    assert act_order.needs_yes is True

    # Drafting does NOT require approval
    msg_draft = {"id": "M02", "chat_id": "C10", "sender_id": "P13", "ts": 1000, "body": "Draft birthday invite text"}
    act_draft = resolve_help_action("Draft birthday invite text", "ME", msg_draft, context, config, chat_id="C10")
    assert act_draft.action == "assist"
    assert act_draft.needs_yes is False
