"""Consolidated Optimization & Lifecycle Regression Test Suite.

Consolidates all high-stakes regression tests across four core domains:
  1. Closure authority and actor verification
  2. Financial obligation and amount separation
  3. Consecutive-turn Q&A closure
  4. Appointment and arrival lifecycle
"""

import re
import pytest

from src.extraction.evidence_extractor import is_relevant_group_candidate
from src.loops.models import CandidateOpenLoop
from src.loops.extractor import (
    BANTER_PATTERN,
    DIRECTED_IMPERATIVE_PATTERN,
    SHORT_VAGUE_ACK_PATTERN,
    TRANSIT_ARRIVAL_PATTERN,
    _evaluate_fallback_actionability,
)
from src.loops.resolver import (
    DIRECTED_REQUEST_PATTERN,
    FIRST_PERSON_COMMITMENT_PATTERN,
    GENERIC_TASK_WORDS,
    STOP_WORDS,
    _are_compatible_task_episodes,
    _evaluate_closing_message,
    _extract_transaction_amounts,
    resolve_loop_owner,
    verify_full_chat_closure,
)
from src.preprocessing.context_builder import ConversationContext


# =============================================================================
# Shared Test Helpers and Mock Fixtures
# =============================================================================

def _make_msg(msg_id, chat_id, sender_id, ts, body, kind="text", quoted_id=None, sender_name=None):
    """Helper to construct synthetic message dictionaries."""
    return {
        "id": msg_id,
        "chat_id": chat_id,
        "sender_id": sender_id,
        "sender_name": sender_name or f"Person_{sender_id}",
        "ts": ts,
        "body": body,
        "kind": kind,
        "quoted_id": quoted_id,
    }


def _make_loop(title, chat_id, owner, source_msg, status="open", closed_by=None, loop_type="commitment"):
    """Helper to construct CandidateOpenLoop objects."""
    return CandidateOpenLoop(
        title=title,
        type=loop_type,
        owner=owner,
        chat_id=chat_id,
        source_msg=source_msg,
        due=None,
        status=status,
        closed_by=closed_by,
    )


class MockContext:
    """Mock ConversationContext for testing arrival and referral lifecycle resolution."""
    def __init__(self, chats=None, dataset=None):
        self.chats = chats or {}
        self.dataset = dataset or {}
        self.participants = set()
        for c in self.chats.values():
            self.participants.update(c.get("participants", []))

    def get_chat(self, chat_id):
        return self.chats.get(chat_id)

    def get_participant(self, pid):
        return {"id": pid, "name": pid}


# =============================================================================
# Section: Closure Authority and Actor Verification
# =============================================================================

# --- Closure Authority and Actor Verification (from Batch 10) (21 tests) ---

def test_legitimate_owner_completion():
    """Task assigned to P01 is closed when P01 confirms completion."""
    src = _make_msg("M10", "C1", "ME", 1000, "P01 please pay the electricity bill")
    closing = _make_msg("M11", "C1", "P01", 1200, "Paid the bill, here is the confirmation")
    msgs = [src, closing]

    res = _evaluate_closing_message(
        msg=closing,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Pay electricity bill",
        is_group=False,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M11"


def test_legitimate_delegate_completion():
    """ME-owned task explicitly delegated to P40 closes when P40 provides deliverable."""
    src = _make_msg("M20", "C1", "P01", 1000, "Arjun please share the Q3 marketing report")
    closing = _make_msg("M22", "C1", "P40", 1400, "Attached the Q3 marketing report for review", kind="document")
    msgs = [src, closing]

    res = _evaluate_closing_message(
        msg=closing,
        source_msg=src,
        chat_id="C1",
        owner="ME",
        requester="P01",
        authenticated_delegates={"P40"},
        loop_type="commitment",
        title="Share Q3 marketing report",
        is_group=True,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M22"


def test_requester_receipt_confirmation():
    """Requester receipt acknowledgement closes an active obligation."""
    src = _make_msg("M30", "C1", "P01", 1000, "Can you review the draft?")
    closing = _make_msg("M31", "C1", "ME", 1100, "Received and cleared")
    msgs = [src, closing]

    res = _evaluate_closing_message(
        msg=closing,
        source_msg=src,
        chat_id="C1",
        owner="ME",
        requester="P01",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Review the draft",
        is_group=False,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M31"


def test_requester_self_fulfillment_rejected():
    """Requester sending an inquiry/follow-up cannot close their own request."""
    src = _make_msg("M40", "C104", "ME", 1000, "Need your help to update mobile number linked to car")
    followup = _make_msg("M41", "C104", "ME", 1050, "Car is in company name. Tried doing it online")
    msgs = [src, followup]

    res = _evaluate_closing_message(
        msg=followup,
        source_msg=src,
        chat_id="C104",
        owner="ME",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Need your help to update mobile number linked to car",
        is_group=False,
        messages=msgs,
    )
    assert res is None


def test_unrelated_adjacent_execution_in_group_rejected():
    """In a group chat, an unrelated 'Done' from someone else does not close ME's task."""
    src = _make_msg("M50", "C5", "ME", 1000, "I will book the flight tickets to Delhi tonight")
    unrelated = _make_msg("M51", "C5", "P02", 1100, "Done with my lunch")
    msgs = [src, unrelated]

    res = _evaluate_closing_message(
        msg=unrelated,
        source_msg=src,
        chat_id="C5",
        owner="ME",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Book flight tickets to Delhi",
        is_group=True,
        messages=msgs,
    )
    assert res is None


def test_relevant_adjacent_execution_message():
    """Relevant execution reply immediately following commitment closes the task."""
    src = _make_msg("M60", "C6", "P01", 1000, "Please order the replacement adapter")
    closing = _make_msg("M61", "C6", "ME", 1300, "Done, bought the replacement adapter on Amazon")
    msgs = [src, closing]

    res = _evaluate_closing_message(
        msg=closing,
        source_msg=src,
        chat_id="C6",
        owner="ME",
        requester="P01",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Order replacement adapter",
        is_group=False,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M61"


def test_relevant_document_delivery():
    """Document delivery matching the task content closes the open loop."""
    src = _make_msg("M70", "C7", "P01", 1000, "Please send over the August billing statement")
    closing = _make_msg("M71", "C7", "ME", 1200, "August billing statement attached", kind="document")
    msgs = [src, closing]

    res = _evaluate_closing_message(
        msg=closing,
        source_msg=src,
        chat_id="C7",
        owner="ME",
        requester="P01",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Send August billing statement",
        is_group=False,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M71"


def test_unrelated_document_delivery_rejected():
    """Unrelated document without semantic overlap does not close the obligation."""
    src = _make_msg("M80", "C8", "P01", 1000, "Please send over the August billing statement")
    unrelated_doc = _make_msg("M81", "C8", "ME", 3000, "Sharing the gym workout routine", kind="document")
    msgs = [src, unrelated_doc]

    res = _evaluate_closing_message(
        msg=unrelated_doc,
        source_msg=src,
        chat_id="C8",
        owner="ME",
        requester="P01",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Send August billing statement",
        is_group=False,
        messages=msgs,
    )
    assert res is None


def test_relevant_direct_answer_to_question():
    """Direct contextual answer to an actionable question closes the loop."""
    src = _make_msg("M90", "C2", "P13", 1000, "Didi kal guests ke liye kya banana hai? Paneer ya chicken")
    closing = _make_msg("M91", "C2", "ME", 1100, "Paneer butter masala aur dal makhani")
    msgs = [src, closing]

    res = _evaluate_closing_message(
        msg=closing,
        source_msg=src,
        chat_id="C2",
        owner="ME",
        requester="P13",
        authenticated_delegates=set(),
        loop_type="question",
        title="Didi kal guests ke liye kya banana hai? Paneer ya chicken",
        is_group=False,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M91"


def test_generic_acknowledgement_remains_open():
    """Simple acknowledgements like 'Ok' or 'Sure' do not close an open loop."""
    src = _make_msg("M100", "C1", "ME", 1000, "Please send the finalized contract by 5pm")
    ack = _make_msg("M101", "C1", "P01", 1050, "Ok sure")
    msgs = [src, ack]

    res = _evaluate_closing_message(
        msg=ack,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Send finalized contract by 5pm",
        is_group=False,
        messages=msgs,
    )
    assert res is None


def test_authorised_requester_cancellation():
    """Requester cancelling their own request drops the loop (status='dropped')."""
    src = _make_msg("M110", "C1", "ME", 1000, "Can you book a cab for 4pm?")
    cancellation = _make_msg("M111", "C1", "ME", 1200, "Never mind, plan changed, cancel that")
    msgs = [src, cancellation]

    res = _evaluate_closing_message(
        msg=cancellation,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Book cab for 4pm",
        is_group=False,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "dropped"
    assert closed_by == "M111"


def test_unauthorised_obligor_cancellation_rejected():
    """Obligor unilaterally saying 'never mind' without preemption reason cannot drop."""
    src = _make_msg("M120", "C1", "ME", 1000, "Please submit the expense report")
    unauth = _make_msg("M121", "C1", "P01", 1200, "Never mind, drop this")
    msgs = [src, unauth]

    res = _evaluate_closing_message(
        msg=unauth,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Submit expense report",
        is_group=False,
        messages=msgs,
    )
    assert res is None


def test_legitimate_preemption_with_valid_cause():
    """Obligor reporting legitimate external preemption drops the loop."""
    src = _make_msg("M130", "C1", "ME", 1000, "Please prepare the presentation for the Acme pitch")
    preempt = _make_msg("M131", "C1", "P01", 1200, "Client cancelled the contract so we aren't doing this")
    msgs = [src, preempt]

    res = _evaluate_closing_message(
        msg=preempt,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Prepare presentation for Acme pitch",
        is_group=False,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "dropped"
    assert closed_by == "M131"


def test_postponement_must_remain_open():
    """Postponements and deferrals preserve open status."""
    src = _make_msg("M140", "C1", "ME", 1000, "Please review the vendor agreement")
    postpone = _make_msg("M141", "C1", "P01", 1100, "A bit tied up today, will do this tomorrow")
    msgs = [src, postpone]

    res = _evaluate_closing_message(
        msg=postpone,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Review vendor agreement",
        is_group=False,
        messages=msgs,
    )
    assert res is None


def test_quoted_closure_within_window():
    """Directly quoting the source message within window closes the loop."""
    src = _make_msg("M150", "C1", "ME", 1000, "Please pay Nayna ₹3500")
    closing = _make_msg("M151", "C1", "P01", 2000, "Done", quoted_id="M150")
    msgs = [src, closing]

    res = _evaluate_closing_message(
        msg=closing,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Pay Nayna ₹3500",
        is_group=False,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M151"


def test_quoted_closure_beyond_window():
    """Explicit quote bridges multi-day delay and closes the task."""
    src = _make_msg("M160", "C1", "ME", 1000, "Please pay Nayna ₹3500")
    # 4 days later (345,600s), but explicitly quotes M160
    closing = _make_msg("M161", "C1", "P01", 346600, "Done paid Nayna ₹3500", quoted_id="M160")
    msgs = [src, closing]

    res = _evaluate_closing_message(
        msg=closing,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Pay Nayna ₹3500",
        is_group=False,
        messages=msgs,
    )
    assert res is not None
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M161"


def test_unquoted_closure_beyond_window_rejected():
    """Unquoted generic closure after multi-day delay is rejected to prevent false closures."""
    src = _make_msg("M170", "C1", "ME", 1000, "Please pay Nayna ₹3500")
    # 4 days later (345,600s), bare 'Done' without quote or semantic overlap
    unquoted = _make_msg("M171", "C1", "P01", 346600, "Done")
    msgs = [src, unquoted]

    res = _evaluate_closing_message(
        msg=unquoted,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Pay Nayna ₹3500",
        is_group=False,
        messages=msgs,
    )
    assert res is None


def test_closure_in_another_chat_rejected():
    """A message from a different chat cannot close the obligation."""
    src = _make_msg("M180", "C1", "ME", 1000, "Please pay Nayna ₹3500")
    other_chat_msg = _make_msg("M181", "C99", "P01", 1100, "Paid Nayna ₹3500")
    msgs = [src]

    res = _evaluate_closing_message(
        msg=other_chat_msg,
        source_msg=src,
        chat_id="C1",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Pay Nayna ₹3500",
        is_group=False,
        messages=msgs,
    )
    assert res is None


def test_unrelated_group_participant_closure_rejected():
    """Third party without owner or delegate authority cannot close the obligation."""
    src = _make_msg("M190", "C10", "ME", 1000, "P01 please update the database index")
    p03_msg = _make_msg("M191", "C10", "P03", 1100, "Done")
    msgs = [src, p03_msg]

    res = _evaluate_closing_message(
        msg=p03_msg,
        source_msg=src,
        chat_id="C10",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Update database index",
        is_group=True,
        messages=msgs,
    )
    assert res is None


def test_clarifying_question_cannot_close_task():
    """A follow-up question or clarification request cannot close an obligation."""
    src = _make_msg("M200", "C104", "ME", 1000, "Need your help to update the mobile number linked to our car")
    clarification = _make_msg("M201", "C104", "P117", 1200, "What's the number you want linked?")
    msgs = [src, clarification]

    res = _evaluate_closing_message(
        msg=clarification,
        source_msg=src,
        chat_id="C104",
        owner="ME",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Need your help to update the mobile number linked to our car",
        is_group=False,
        messages=msgs,
    )
    assert res is None


def test_distinct_task_episodes_do_not_merge():
    """Distinct tasks with generic action verbs do not merge into unrelated episodes."""
    ep1 = CandidateOpenLoop(
        title="Ill buy the fingerprint device first and then ask him to note",
        type="commitment",
        owner="ME",
        chat_id="C10",
        source_msg="M01810",
    )
    ep2 = CandidateOpenLoop(
        title="Text him now. He will buy. Buy 14 L",
        type="commitment",
        owner="ME",
        chat_id="C10",
        source_msg="M02237",
    )
    ctx = MagicMock()
    ctx.chats = {}
    assert not _are_compatible_task_episodes(ep1, ep2, ctx)



# =============================================================================
# Section: Financial Obligation and Amount Separation
# =============================================================================

# --- Financial Obligation & Group Addressee Matching (from Batch 11) (9 tests) ---

def test_payment_amount_clash_prevents_merge():
    """Two payment requests with distinct transaction amounts must not merge into one episode."""
    ep = _make_loop("Pls Pay 4500 on this", "C10", "ME", "M10")
    cand = _make_loop("Pls Pay 3500 on this", "C10", "ME", "M20")

    context = MagicMock()
    context.dataset = {"messages_by_id": {}}
    context.get_chat.return_value = {"is_group": False}

    # Distinct amounts 4500 vs 3500 must NOT be compatible
    assert not _are_compatible_task_episodes(ep, cand, context)

    # Identical amount should be compatible
    cand_same = _make_loop("Pls Pay 4500 on this", "C10", "ME", "M21")
    assert _are_compatible_task_episodes(ep, cand_same, context)


def test_group_addressee_matching_assigns_me():
    """Group chat message explicitly addressing ME with 'we need' assigns ownership to ME."""
    context = MagicMock()
    context.get_participant.side_effect = lambda pid: {"name": "Arjun Khetan"} if pid == "ME" else {"name": "Other"}
    context.get_chat.return_value = {
        "is_group": True,
        "participants": ["ME", "P10", "P20"],
    }
    context.chats = {"C24": {"is_group": True, "participants": ["ME", "P10", "P20"]}}

    msg = _make_msg("M3358", "C24", "P10", 1000, "Arjun, we need the PI Copy to raise the PO")
    owner = resolve_loop_owner(
        raw_owner=None,
        chat_id="C24",
        source_msg=msg,
        context=context,
    )
    assert owner == "ME"


def test_two_person_group_directed_request():
    """In a group chat with only 2 participants, directed requests from counterpart resolve to ME."""
    context = MagicMock()
    context.get_participant.side_effect = lambda pid: {"name": "Arjun Khetan"} if pid == "ME" else {"name": "Trainer"}
    context.get_chat.return_value = {
        "is_group": True,
        "participants": ["ME", "P136"],
    }
    context.chats = {"C107": {"is_group": True, "participants": ["ME", "P136"]}}

    msg = _make_msg(
        "M2516",
        "C107",
        "P136",
        1000,
        "Sept morning batch. Fee: 6500/-. Please confirm your spot by sharing payment screenshot by 2nd Sep.",
    )
    owner = resolve_loop_owner(
        raw_owner=None,
        chat_id="C107",
        source_msg=msg,
        context=context,
    )
    assert owner == "ME"


def test_small_group_dinner_attendance_relevance():
    """Small group dinner query ('You both coming for dinner...') is retained as actionable."""
    context = MagicMock()
    context.chats = {
        "C73": {
            "chat_id": "C73",
            "is_group": True,
            "participants": ["ME", "P39", "P40"],
        }
    }
    context.dataset = {"messages_by_id": {}}

    cand = {
        "message_id": "M1874",
        "chat_id": "C73",
        "sender_id": "P39",
        "category": "question",
    }
    msg = _make_msg("M1874", "C73", "P39", 1000, "You both coming for dinner on 31st Aug / Monday right? @123 @456")

    assert is_relevant_group_candidate(cand, msg, context)


def test_business_discussion_request_owner_me():
    """When ME sends 'Please discuss with me tomorrow', owner remains ME."""
    context = MagicMock()
    context.get_chat.return_value = {"is_group": False, "participants": ["ME", "P65"]}
    context.chats = {"C62": {"is_group": False, "participants": ["ME", "P65"]}}

    msg = _make_msg("M2794", "C62", "ME", 1000, "Please discuss with me tomorrow")
    owner = resolve_loop_owner(
        raw_owner=None,
        chat_id="C62",
        source_msg=msg,
        context=context,
    )
    assert owner == "ME"


def test_closing_fulfillment_subsumes_preliminary_request():
    """When counterpart confirms tomorrow's meeting, it is recognized as compatible with preliminary request."""
    ep = _make_loop("Please discuss with me tomorrow", "C82", "ME", "M2800", status="done", closed_by="M2803")
    cand = _make_loop(
        "Also I got the errors fixed we talked about today and my work on Orbit is also done so we can sit tomorrow",
        "C82",
        "ME",
        "M2803",
        status="open",
    )

    context = MagicMock()
    context.dataset = {"messages_by_id": {
        "M2800": {"id": "M2800", "ts": 1000},
        "M2803": {"id": "M2803", "ts": 1500},
    }}
    context.get_chat.return_value = {"is_group": False, "participants": ["ME", "P169"]}
    context.chats = {"C82": {"is_group": False, "participants": ["ME", "P169"]}}

    assert _are_compatible_task_episodes(ep, cand, context)


def test_polite_prefix_imperatives():
    """Imperatives with 'Pl', 'Pls', 'Please' correctly match directed imperative pattern."""
    assert bool(DIRECTED_IMPERATIVE_PATTERN.search("Pl get me also on claude premium"))
    assert bool(DIRECTED_IMPERATIVE_PATTERN.search("Pls Pay 3500 on this"))
    assert bool(DIRECTED_IMPERATIVE_PATTERN.search("Please send the invoice"))
    assert bool(DIRECTED_IMPERATIVE_PATTERN.search("kindly update the spreadsheet"))


def test_future_tasks_complete_incorporate():
    """Future tasks with 'will complete' or 'will incorporate' evaluate as actionable."""
    context = MagicMock()
    context.get_chat.return_value = {"is_group": False, "participants": ["ME", "P13"]}
    context.chats = {"C10": {"is_group": False, "participants": ["ME", "P13"]}}
    chat = {"is_group": False, "participants": ["ME", "P13"]}

    cand1 = {
        "message_id": "M637",
        "chat_id": "C10",
        "sender_id": "ME",
        "category": "commitment",
        "matched_signals": ["will complete"],
    }
    msg1 = _make_msg("M637", "C10", "ME", 1000, "Accumulate over time: vayu, ShopEase. Short term: homeserve, .. will complete by sat.")

    is_act, cat, owner, conf = _evaluate_fallback_actionability(cand1, msg1, context, chat)
    assert is_act
    assert cat == "commitment"
    assert owner == "ME"

    cand2 = {
        "message_id": "M2592",
        "chat_id": "C10",
        "sender_id": "ME",
        "category": "commitment",
        "matched_signals": ["will incorporate"],
    }
    msg2 = _make_msg("M2592", "C10", "ME", 1000, "thanks., will incorporate as reminders. also lmk sandipa house tasks")

    is_act2, cat2, owner2, conf2 = _evaluate_fallback_actionability(cand2, msg2, context, chat)
    assert is_act2
    assert cat2 == "commitment"


def test_social_banter_and_birthday_wishes_excluded():
    """Birthday greetings and casual chatter must not be extracted as open loops."""
    context = MagicMock()
    context.chats = {"C50": {"is_group": True, "participants": ["ME", "P1", "P2", "P3"]}}
    chat = {"is_group": True, "participants": ["ME", "P1", "P2", "P3"]}

    birthday_msgs = [
        "Happy birthday Arjun! Have a great year ahead!!",
        "Wish you a very happy birthday bhai! Party kab de raha?",
        "Happy birthday! God bless you!",
    ]
    for text in birthday_msgs:
        cand = {"message_id": "M_bday", "chat_id": "C50", "sender_id": "P1", "category": "commitment"}
        msg = _make_msg("M_bday", "C50", "P1", 1000, text)
        is_act, _, _, _ = _evaluate_fallback_actionability(cand, msg, context, chat)
        assert not is_act


# --- Non-Currency Numbers & Transaction Amount Clash Safeguards (from Batch 11.1) (7 tests) ---

def test_non_currency_numbers_do_not_trigger_amount_clash():
    """Calendar years (2026), times (1230), and counts must not be treated as currency amounts."""
    # Test amount extractor directly
    assert _extract_transaction_amounts("Schedule meeting for 2026-09-02") == set()
    assert _extract_transaction_amounts("Call at 1230") == set()
    assert _extract_transaction_amounts("Review 3 documents") == set()
    assert _extract_transaction_amounts("Meet at 1100 tomorrow") == set()

    # Two matching tasks with dates/times should merge without amount clash
    ep = _make_loop("Schedule meeting 2026-09-02", "C10", "ME", "M10")
    cand = _make_loop("Schedule meeting 2026-09-03", "C10", "ME", "M20")

    context = MagicMock()
    context.dataset = {"messages_by_id": {}}
    context.get_chat.return_value = {"is_group": False}

    assert _are_compatible_task_episodes(ep, cand, context)


def test_contextual_amount_clash_prevents_merge():
    """Currency amounts and payment-grounded amounts (₹4,500 vs ₹3,500) prevent merging."""
    # Test amount extractor directly
    assert _extract_transaction_amounts("Pls Pay 3500 on this") == {3500}
    assert _extract_transaction_amounts("Transfer ₹4,500") == {4500}
    assert _extract_transaction_amounts("Pay 4500/-") == {4500}
    assert _extract_transaction_amounts("Rent 25000 per month") == {25000}
    assert _extract_transaction_amounts("Fee: 5000") == {5000}

    # Distinct amounts prevent merging
    ep = _make_loop("Pay ₹4,500", "C10", "ME", "M10")
    cand = _make_loop("Pay ₹3,500", "C10", "ME", "M20")

    context = MagicMock()
    context.dataset = {"messages_by_id": {}}
    context.get_chat.return_value = {"is_group": False}

    assert not _are_compatible_task_episodes(ep, cand, context)

    # Identical amounts allow merging
    cand_same = _make_loop("Pay ₹4,500", "C10", "ME", "M21")
    assert _are_compatible_task_episodes(ep, cand_same, context)


def test_closing_message_with_separate_obligation_not_subsumed():
    """A message that closed a prior task but contains an independent obligation must not be subsumed."""
    # Episode 1: Pay electricity bill, closed by M20
    ep = _make_loop("Pay electricity bill", "C10", "ME", "M10", status="closed", closed_by="M20")
    # Candidate M20 introduces a completely distinct task: Order 2 stand fans
    cand = _make_loop("Order 2 stand fans", "C10", "ME", "M20")

    context = MagicMock()
    context.dataset = {
        "messages_by_id": {
            "M10": _make_msg("M10", "C10", "P1", 1000, "Pls pay electricity bill"),
            "M20": _make_msg("M20", "C10", "ME", 2000, "Paid electricity bill, also order 2 stand fans"),
        }
    }
    context.get_chat.return_value = {"is_group": False}

    # Action clash (pay vs order) must prevent cand M20 from being subsumed into ep M10
    assert not _are_compatible_task_episodes(ep, cand, context)


def test_closing_message_with_continuation_subsumed():
    """A closure message that represents legitimate meeting/discussion coordination is subsumed."""
    ep = _make_loop("Discuss marketing strategy", "C10", "ME", "M10", status="closed", closed_by="M20")
    cand = _make_loop("Discuss meeting tomorrow at 4pm", "C10", "ME", "M20")

    context = MagicMock()
    context.dataset = {
        "messages_by_id": {
            "M10": _make_msg("M10", "C10", "P1", 1000, "Can we discuss marketing strategy?"),
            "M20": _make_msg("M20", "C10", "ME", 2000, "Sure, let's discuss meeting tomorrow at 4pm"),
        }
    }
    context.get_chat.return_value = {"is_group": False}

    assert _are_compatible_task_episodes(ep, cand, context)


def test_small_group_relevance_positive():
    """Directed action requests in a 2-person group chat are retained as relevant candidates."""
    context = MagicMock()
    context.chats = {
        "C2": {
            "id": "C2",
            "is_group": True,
            "participants": ["ME", "P1"],
        }
    }
    context.get_participant.side_effect = lambda pid: {"name": "Arjun"} if pid == "ME" else {"name": "Priya"}

    # 'Pls Pay 3500 on this'
    msg1 = _make_msg("M1", "C2", "P1", 1000, "Pls Pay 3500 on this")
    cand1 = {"category": "commitment", "matched_signals": ["pay"]}
    assert is_relevant_group_candidate(cand1, msg1, context)

    # 'Please send the invoice'
    msg2 = _make_msg("M2", "C2", "P1", 1000, "Please send the invoice copy today")
    cand2 = {"category": "commitment", "matched_signals": ["send"]}
    assert is_relevant_group_candidate(cand2, msg2, context)

    # 'Can you share the report?'
    msg3 = _make_msg("M3", "C2", "P1", 1000, "Can you share the report when ready?")
    cand3 = {"category": "question", "matched_signals": ["share"]}
    assert is_relevant_group_candidate(cand3, msg3, context)


def test_small_group_relevance_negative():
    """Generic announcements, ungrounded polite notices, and ambient chatter in 2-person groups are rejected."""
    context = MagicMock()
    context.chats = {
        "C2": {
            "id": "C2",
            "is_group": True,
            "participants": ["ME", "P1"],
        }
    }
    context.get_participant.side_effect = lambda pid: {"name": "Arjun"} if pid == "ME" else {"name": "Priya"}

    # Generic polite notice
    msg1 = _make_msg("M1", "C2", "P1", 1000, "Please note the office is closed tomorrow")
    cand1 = {"category": "commitment", "matched_signals": ["pls"]}
    assert not is_relevant_group_candidate(cand1, msg1, context)

    # Kindly be aware
    msg2 = _make_msg("M2", "C2", "P1", 1000, "Kindly be aware of the updated security guidelines")
    cand2 = {"category": "commitment", "matched_signals": ["kindly"]}
    assert not is_relevant_group_candidate(cand2, msg2, context)

    # Ungrounded phrase with 'pay'
    msg3 = _make_msg("M3", "C2", "P1", 1000, "We should pay heed to the advice")
    cand3 = {"category": "commitment", "matched_signals": ["pay"]}
    assert not is_relevant_group_candidate(cand3, msg3, context)


def test_in_flight_update_tightening_does_not_merge_payment_request():
    """'Pls Pay 3500 on this' must not trigger in-flight status update and merge into 'Pls call me'."""
    ep = _make_loop("Pls call me", "C10", "ME", "M3494", status="open")
    cand = _make_loop("Pls Pay 3500 on this", "C10", "ME", "M3780", status="open")

    context = MagicMock()
    context.dataset = {
        "messages_by_id": {
            "M3494": _make_msg("M3494", "C10", "P1", 1000, "Pls call me"),
            "M3780": _make_msg("M3780", "C10", "P1", 2000, "Pls Pay 3500 on this"),
        }
    }
    context.get_chat.return_value = {"is_group": False}

    # Action clash (call vs pay) must prevent merge despite "on this" in text
    assert not _are_compatible_task_episodes(ep, cand, context)


# --- Time String Exclusion & Decision Retention (from Batch 11.2) (6 tests) ---

def test_time_strings_and_years_excluded_from_amount_extraction():
    """Time expressions and years must not be extracted as transaction amounts."""
    time_cases = [
        "12 pm",
        "3 pm",
        "12:30 pm",
        "12:30",
        "2026",
        "meet at 12 pm tomorrow",
        "call at 3 pm",
        "catch up at 12:30 pm",
        "lunch at 12:30",
        "plan for 2026",
    ]
    for tc in time_cases:
        res = _extract_transaction_amounts(tc)
        assert res == set(), f"Expected empty set for '{tc}', got {res}"


def test_genuine_payment_amounts_extracted():
    """Explicit currencies and contextual payment amounts are reliably extracted."""
    amount_cases = {
        "Pls Pay 3500 on this": {3500},
        "Transfer ₹4,500": {4500},
        "Rent 25000/pm": {25000},
        "Rent 25000 per month": {25000},
        "Fees: 5000": {5000},
        "Pay 4500/-": {4500},
        "Send $500 advance": {500},
    }
    for ac, expected in amount_cases.items():
        res = _extract_transaction_amounts(ac)
        assert res == expected, f"Failed for '{ac}': got {res}, expected {expected}"


def test_different_times_do_not_create_transaction_clash():
    """Different meeting/call times (e.g. 12 pm vs 3 pm) do NOT trigger amount clash and allow merging."""
    ep = _make_loop("Call at 12 pm", "C10", "ME", "M10")
    cand = _make_loop("Call at 3 pm", "C10", "ME", "M20")

    context = MagicMock()
    context.dataset = {"messages_by_id": {}}
    context.get_chat.return_value = {"is_group": False}

    assert _are_compatible_task_episodes(ep, cand, context)

    # In contrast, different transaction amounts MUST clash
    ep_amt = _make_loop("Pay ₹4,500", "C10", "ME", "M10")
    cand_amt = _make_loop("Pay ₹3,500", "C10", "ME", "M20")
    assert not _are_compatible_task_episodes(ep_amt, cand_amt, context)


def test_actionable_decisions_retained_in_small_groups():
    """Decision queries with concrete action verbs or deadlines are retained."""
    context = MagicMock()
    context.chats = {
        "C04": {
            "chat_id": "C04",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
        }
    }
    context.get_participant.side_effect = lambda pid: {"name": f"Person_{pid}"}

    # 'Should we take it?'
    msg1 = _make_msg("M016", "C04", "P01", 1000, "Zest wants us at 60k/month instead of 80. Should we take it?")
    cand1 = {"category": "question", "matched_signals": ["should we"]}
    assert is_relevant_group_candidate(cand1, msg1, context)

    # 'They want an answer by Monday'
    msg2 = _make_msg("M020", "C04", "P01", 1000, "Any thoughts on Zest? They want an answer by Monday")
    cand2 = {"category": "question", "matched_signals": ["any thoughts"]}
    assert is_relevant_group_candidate(cand2, msg2, context)

    # 'Shall we sign the vendor contract?'
    msg3 = _make_msg("M030", "C04", "P01", 1000, "Contract draft looks good. Shall we sign it today?")
    cand3 = {"category": "question", "matched_signals": ["shall we"]}
    assert is_relevant_group_candidate(cand3, msg3, context)


def test_dinner_and_attendance_coordination_retained():
    """Direct dinner or attendance queries in small groups are retained."""
    context = MagicMock()
    context.chats = {
        "C73": {
            "chat_id": "C73",
            "is_group": True,
            "participants": ["ME", "P39", "P40"],
        }
    }
    context.get_participant.side_effect = lambda pid: {"name": f"Person_{pid}"}

    msg = _make_msg("M1874", "C73", "P39", 1000, "You both coming for dinner on 31st Aug / Monday right?")
    cand = {"category": "question", "matched_signals": ["coming for dinner"]}
    assert is_relevant_group_candidate(cand, msg, context)


def test_vague_small_group_inquiries_rejected():
    """Vague opinion requests and passive discussions without concrete actions are rejected."""
    context = MagicMock()
    context.chats = {
        "C04": {
            "chat_id": "C04",
            "is_group": True,
            "participants": ["ME", "P01", "P02"],
        }
    }
    context.get_participant.side_effect = lambda pid: {"name": f"Person_{pid}"}

    # Vague 'any thoughts?'
    msg1 = _make_msg("M101", "C04", "P01", 1000, "Any thoughts?")
    cand1 = {"category": "question", "matched_signals": ["any thoughts"]}
    assert not is_relevant_group_candidate(cand1, msg1, context)

    # Vague 'what do you think?'
    msg2 = _make_msg("M102", "C04", "P01", 1000, "Interesting market analysis. What do you think?")
    cand2 = {"category": "question", "matched_signals": ["what do you think"]}
    assert not is_relevant_group_candidate(cand2, msg2, context)

    # General 'should we wait and see?'
    msg3 = _make_msg("M103", "C04", "P01", 1000, "The competitors launched a feature. Should we wait and see?")
    cand3 = {"category": "question", "matched_signals": ["should we"]}
    assert not is_relevant_group_candidate(cand3, msg3, context)

    # Social chatter
    msg4 = _make_msg("M104", "C04", "P01", 1000, "Great work on the presentation everyone!")
    cand4 = {"category": "commitment", "matched_signals": []}
    assert not is_relevant_group_candidate(cand4, msg4, context)

    # General announcement
    msg5 = _make_msg("M105", "C04", "P01", 1000, "Please note the office is closed on Friday for maintenance.")
    cand5 = {"category": "commitment", "matched_signals": ["please"]}
    assert not is_relevant_group_candidate(cand5, msg5, context)



# =============================================================================
# Section: Consecutive-Turn Q&A Closure
# =============================================================================

# --- Consecutive-Turn Q&A Closure Recovery (from Batch 12.2) (8 tests) ---

def test_consecutive_turn_direct_qa_positive():
    """Direct 1-on-1 Q&A: substantive multi-token answer in the immediate next turn closes question."""
    src = _make_msg("M1", "C97", "P197", 1000, "Any plans for Bangalore? Or shall we Zoom?")
    ans = _make_msg("M2", "C97", "ME", 4723, "Likely zoom only over a Saturday . Bangalore this quarter.")
    messages = [src, ans]

    res = _evaluate_closing_message(
        msg=ans,
        source_msg=src,
        chat_id="C97",
        owner="ME",
        requester="P197",
        authenticated_delegates=set(),
        loop_type="question",
        title="Any plans for Bangalore? Or shall we Zoom",
        is_group=False,
        messages=messages,
    )
    assert res == ("done", "M2")


def test_consecutive_turn_counter_question_rejected():
    """An immediately following message that asks another question cannot close the question."""
    src = _make_msg("M1", "C97", "P197", 1000, "Any plans for Bangalore? Or shall we Zoom?")
    ans = _make_msg("M2", "C97", "ME", 1200, "What time are you free for a Zoom call?")
    messages = [src, ans]

    res = _evaluate_closing_message(
        msg=ans,
        source_msg=src,
        chat_id="C97",
        owner="ME",
        requester="P197",
        authenticated_delegates=set(),
        loop_type="question",
        title="Any plans for Bangalore? Or shall we Zoom",
        is_group=False,
        messages=messages,
    )
    assert res is None


def test_consecutive_turn_hedging_or_postponement_rejected():
    """An immediately following message with hedging or postponement cannot close the question."""
    src = _make_msg("M1", "C97", "P197", 1000, "Any plans for Bangalore? Or shall we Zoom?")
    ans = _make_msg("M2", "C97", "ME", 1200, "Not sure yet about Bangalore or Zoom, will let you know.")
    messages = [src, ans]

    res = _evaluate_closing_message(
        msg=ans,
        source_msg=src,
        chat_id="C97",
        owner="ME",
        requester="P197",
        authenticated_delegates=set(),
        loop_type="question",
        title="Any plans for Bangalore? Or shall we Zoom",
        is_group=False,
        messages=messages,
    )
    assert res is None


def test_consecutive_turn_bare_acknowledgment_rejected():
    """An immediately following message with bare acknowledgment (ok, cool) without tokens cannot close."""
    src = _make_msg("M1", "C97", "P197", 1000, "Any plans for Bangalore? Or shall we Zoom?")
    ans = _make_msg("M2", "C97", "ME", 1200, "Coolio, sounds good.")
    messages = [src, ans]

    res = _evaluate_closing_message(
        msg=ans,
        source_msg=src,
        chat_id="C97",
        owner="ME",
        requester="P197",
        authenticated_delegates=set(),
        loop_type="question",
        title="Any plans for Bangalore? Or shall we Zoom",
        is_group=False,
        messages=messages,
    )
    assert res is None


def test_consecutive_turn_in_group_chat_rejected():
    """Consecutive turn Q&A rule does NOT apply in a group chat."""
    src = _make_msg("M1", "C10", "P197", 1000, "Any plans for Bangalore? Or shall we Zoom?")
    ans = _make_msg("M2", "C10", "ME", 4723, "Likely zoom only over a Saturday . Bangalore this quarter.")
    messages = [src, ans]

    res = _evaluate_closing_message(
        msg=ans,
        source_msg=src,
        chat_id="C10",
        owner="ME",
        requester="P197",
        authenticated_delegates=set(),
        loop_type="question",
        title="Any plans for Bangalore? Or shall we Zoom",
        is_group=True,
        messages=messages,
    )
    assert res is None


def test_non_consecutive_turn_rejected():
    """If intervening messages exist, the consecutive-turn Q&A rule does not apply."""
    src = _make_msg("M1", "C97", "P197", 1000, "Any plans for Bangalore? Or shall we Zoom?")
    interv = _make_msg("M2", "C97", "P197", 1100, "Also checking on the report.")
    ans = _make_msg("M3", "C97", "ME", 4723, "Likely zoom only over a Saturday . Bangalore this quarter.")
    messages = [src, interv, ans]

    res = _evaluate_closing_message(
        msg=ans,
        source_msg=src,
        chat_id="C97",
        owner="ME",
        requester="P197",
        authenticated_delegates=set(),
        loop_type="question",
        title="Any plans for Bangalore? Or shall we Zoom",
        is_group=False,
        messages=messages,
    )
    assert res is None


def test_group_chat_name_mention_without_task_tokens_rejected():
    """In a group chat, mentioning a participant's name without deliverable tokens does not close task."""
    prev = _make_msg("M10", "C24", "P236", 900, "Calling, ha", sender_name="Reddy Nopany")
    src = _make_msg("M11", "C24", "P236", 1000, "Arjun, we need the PI Copy to raise the PO", sender_name="Reddy Nopany")
    interv = _make_msg("M12", "C24", "P236", 1050, "@Arjun")
    closer = _make_msg(
        "M13",
        "C24",
        "ME",
        1100,
        "shared in email @220138464732891 please reshare also if reddy not in email cc him.",
    )
    messages = [prev, src, interv, closer]

    res = _evaluate_closing_message(
        msg=closer,
        source_msg=src,
        chat_id="C24",
        owner="ME",
        requester="P236",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Arjun, we need the PI Copy to raise the PO",
        is_group=True,
        messages=messages,
    )
    assert res is None


def test_mixed_completion_and_followup_with_task_tokens_accepted():
    """A message that explicitly names the deliverable and confirms completion closes the task."""
    src = _make_msg("M20", "C24", "P236", 1000, "Arjun, we need the PI Copy to raise the PO")
    closer = _make_msg(
        "M21",
        "C24",
        "ME",
        1100,
        "Shared the PI copy on email. @Sakshi please also follow up on the PO.",
    )
    messages = [src, closer]

    res = _evaluate_closing_message(
        msg=closer,
        source_msg=src,
        chat_id="C24",
        owner="ME",
        requester="P236",
        authenticated_delegates=set(),
        loop_type="commitment",
        title="Arjun, we need the PI Copy to raise the PO",
        is_group=True,
        messages=messages,
    )
    assert res == ("done", "M21")



# =============================================================================
# Section: Appointment and Arrival Lifecycle
# =============================================================================

# --- In-Person Arrival Lifecycle & Referral Commitments (from Batch 14.1) (5 tests) ---

def test_phase1_referral_commitment_patterns():
    # Positive tests: genuine future commitments to connect or introduce
    positives = [
        "sure we'll connect with her. thanks for ref!",
        "we'll connect with him soon. thanks for ref",
        "i'll connect you both tomorrow",
        "will introduce you to her. thanks for ref!",
        "we'll reach out to her next week",
    ]
    for text in positives:
        assert not SHORT_VAGUE_ACK_PATTERN.search(text.strip()), f"Should not be blocked by ack pattern: {text}"
        assert FIRST_PERSON_COMMITMENT_PATTERN.search(text), f"Should match commitment pattern: {text}"

    # Negative tests: bare acknowledgements without pledge must be pruned
    negatives = [
        "thanks for ref!",
        "thanks for the ref",
        "thanks for ref",
        "sure, thanks for ref!",
        "ok thanks for the ref!",
    ]
    for text in negatives:
        assert SHORT_VAGUE_ACK_PATTERN.search(text.strip()), f"Should be blocked by ack pattern: {text}"


def test_phase2_scheduled_appointment_arrival():
    # Direct 1-on-1 chat context
    chat = {"chat_id": "C114", "is_group": False, "participants": ["ME", "P210"]}
    dataset = {
        "messages_by_chat": {
            "C114": [
                {"id": "M02900", "sender_id": "P210", "body": "Friday what time?", "ts": 100},
                {"id": "M02902", "sender_id": "ME", "body": "Anytime 3pm onwards works", "ts": 200},
                {"id": "M02903", "sender_id": "P210", "body": "3 pm ill reach", "ts": 300},
            ]
        }
    }
    context = MockContext(chats={"C114": chat}, dataset=dataset)

    cand = {"message_id": "M02903", "chat_id": "C114", "sender_id": "P210", "category": "commitment"}
    msg = dataset["messages_by_chat"]["C114"][2]

    is_act, cat, owner, conf = _evaluate_fallback_actionability(cand, msg, context, chat)
    assert is_act is True, "Appointment arrival commitment should be actionable"
    assert owner == "ME", f"Appointment arrival should resolve owner to ME, got {owner}"

    # Negative test: standalone transit chatter without scheduling query should be pruned
    cand_neg = {"message_id": "M99999", "chat_id": "C114", "sender_id": "P210", "category": "commitment"}
    msg_neg = {"id": "M99999", "chat_id": "C114", "sender_id": "P210", "body": "reaching now", "ts": 400}
    is_act_neg, _, _, _ = _evaluate_fallback_actionability(cand_neg, msg_neg, context, chat)
    assert is_act_neg is False, "Standalone transit chatter should be pruned"


def test_phase3_in_person_venue_arrival_closure():
    source_msg = {
        "id": "M00524",
        "chat_id": "C38",
        "sender_id": "ME",
        "ts": 1000,
        "body": "Sorry missed the call. Was pretty sick . Spice vc is also same stage as nath. We discuss this Saturday ?",
    }
    closing_msg = {
        "id": "M01575",
        "chat_id": "C38",
        "sender_id": "P01",
        "ts": 1000 + 3 * 86400 + 10,
        "body": "Flat 34",
    }
    messages = [
        source_msg,
        {"id": "M00526", "chat_id": "C38", "sender_id": "P01", "ts": 1010, "body": "Coolio"},
        {"id": "M01497", "chat_id": "C38", "sender_id": "ME", "ts": 1000 + 3 * 86400 - 3600, "body": "Just reconfirming today 11 am is on at ur office ,?"},
        {"id": "M01503", "chat_id": "C38", "sender_id": "P01", "ts": 1000 + 3 * 86400 - 3500, "body": "Yes"},
        {"id": "M01574", "chat_id": "C38", "sender_id": "ME", "ts": 1000 + 3 * 86400, "body": "Hi which floor."},
        closing_msg,
        {"id": "M01576", "chat_id": "C38", "sender_id": "P01", "ts": 1000 + 3 * 86400 + 20, "body": "3rd floor"},
    ]
    chat = {"chat_id": "C38", "is_group": False, "participants": ["ME", "P01"], "messages": messages}
    context = MockContext(chats={"C38": chat})

    res = _evaluate_closing_message(
        msg=closing_msg,
        source_msg=source_msg,
        chat_id="C38",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="question",
        title="Discuss Spice VC and nath with Nirant on Saturday",
        is_group=False,
        messages=messages,
    )
    assert res is not None, "In-person venue arrival should close the meeting obligation"
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M01575"


def test_phase4_dated_event_post_event_appreciation():
    source_msg = {
        "id": "M01874",
        "chat_id": "C73",
        "sender_id": "P39",
        "ts": 10000,
        "body": "You both coming for dinner on 31st Aug / Monday right? @249417682946514 @44444590182763",
    }
    closing_msg = {
        "id": "M04373",
        "chat_id": "C73",
        "sender_id": "P13",
        "ts": 10000 + 7 * 86400,
        "body": "So lovely!",
    }
    messages = [source_msg, closing_msg]
    chat = {"chat_id": "C73", "is_group": True, "participants": ["ME", "P39", "P13", "P51"], "messages": messages}
    context = MockContext(chats={"C73": chat})

    res = _evaluate_closing_message(
        msg=closing_msg,
        source_msg=source_msg,
        chat_id="C73",
        owner="ME",
        requester="P39",
        authenticated_delegates=set(),
        loop_type="question",
        title="Dinner on Monday 31st Aug",
        is_group=True,
        messages=messages,
    )
    # Under Batch 14.2 safety rules, generic "So lovely!" without explicit event linkage preserves open status
    assert res is None, "Generic post-event appreciation without explicit event linkage must not close obligation"


def test_duplicate_forward_safeguard():
    # If ME forwards an identical message to another chat within 10 minutes with the same beneficiary, secondary is pruned
    chat1 = {"chat_id": "C34", "is_group": True, "participants": ["ME", "P192"]}
    chat2 = {"chat_id": "C62", "is_group": False, "participants": ["ME", "P65"]}
    dataset = {
        "messages_by_chat": {
            "C34": [
                {"id": "M02306", "sender_id": "P192", "body": "Shared Lalitha Bhul contact for marketing", "ts": 990},
                {"id": "M02307", "sender_id": "ME", "body": "sure we'll connect with her. thanks for ref!", "ts": 1000},
            ],
            "C62": [
                {"id": "M02310", "sender_id": "ME", "body": "Shared Lalitha Bhul contact for marketing", "ts": 1130},
                {"id": "M02311", "sender_id": "ME", "body": "sure we'll connect with her. thanks for ref!", "ts": 1147},
            ],
        }
    }
    context = MockContext(chats={"C34": chat1, "C62": chat2}, dataset=dataset)

    # First occurrence in C34 must be actionable
    cand1 = {"message_id": "M02307", "chat_id": "C34", "sender_id": "ME", "category": "commitment"}
    msg1 = dataset["messages_by_chat"]["C34"][1]
    is_act1, _, _, _ = _evaluate_fallback_actionability(cand1, msg1, context, chat1)
    assert is_act1 is True, "Original commitment in C34 must be actionable"

    # Second occurrence in C62 must be pruned as duplicate forward of same beneficiary
    cand2 = {"message_id": "M02311", "chat_id": "C62", "sender_id": "ME", "category": "commitment"}
    msg2 = dataset["messages_by_chat"]["C62"][1]
    is_act2, _, _, _ = _evaluate_fallback_actionability(cand2, msg2, context, chat2)
    assert is_act2 is False, "Forwarded duplicate in C62 within 10 minutes must be pruned"


# --- In-Person Meeting Contextual Arrival Safety (from Batch 14.2) (12 tests) ---

def test_phase1_positive_m00524_contextual_arrival():
    source_msg = {
        "id": "M00524",
        "chat_id": "C38",
        "sender_id": "ME",
        "ts": 1787670073,
        "body": "Sorry missed the call. Was pretty sick . Spice vc is also same stage as nath. We discuss this Saturday ?",
    }
    messages = [
        source_msg,
        {"id": "M00526", "chat_id": "C38", "sender_id": "P01", "ts": 1787670100, "body": "Coolio"},
        {"id": "M01497", "chat_id": "C38", "sender_id": "ME", "ts": 1787970000, "body": "Just reconfirming today 11 am is on at ur office ,?"},
        {"id": "M01503", "chat_id": "C38", "sender_id": "P01", "ts": 1787970050, "body": "Yes"},
        {"id": "M01574", "chat_id": "C38", "sender_id": "ME", "ts": 1787973000, "body": "Hi which floor."},
        {"id": "M01575", "chat_id": "C38", "sender_id": "P01", "ts": 1787973010, "body": "Flat 34"},
        {"id": "M01576", "chat_id": "C38", "sender_id": "P01", "ts": 1787973020, "body": "3rd floor"},
    ]
    chat = {"chat_id": "C38", "is_group": False, "participants": ["ME", "P01"], "messages": messages}
    context = MockContext(chats={"C38": chat})

    # Candidate message M01575 answers arrival inquiry M01574
    res = _evaluate_closing_message(
        msg=messages[5],
        source_msg=source_msg,
        chat_id="C38",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="question",
        title="We discuss this Saturday ?",
        is_group=False,
        messages=messages,
    )
    assert res is not None, "Valid arrival coordination answer should close meeting"
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M01575"


def test_phase1_negative_unrelated_apartment_address():
    source_msg = {
        "id": "M00524",
        "chat_id": "C38",
        "sender_id": "ME",
        "ts": 1787670073,
        "body": "We discuss this Saturday ?",
    }
    messages = [
        source_msg,
        {"id": "M1", "chat_id": "C38", "sender_id": "P01", "ts": 1787973000, "body": "Here is the lease copy for Flat 34, 3rd floor"},
    ]
    res = _evaluate_closing_message(
        msg=messages[1],
        source_msg=source_msg,
        chat_id="C38",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="question",
        title="We discuss this Saturday ?",
        is_group=False,
        messages=messages,
    )
    assert res is None, "Unrelated apartment address must not close meeting"


def test_phase1_negative_different_meeting_outside_window():
    source_msg = {
        "id": "M00524",
        "chat_id": "C38",
        "sender_id": "ME",
        "ts": 1787670073 - 20 * 86400, # 20 days earlier
        "body": "We discuss this Saturday ?",
    }
    messages = [
        source_msg,
        {"id": "M01574", "chat_id": "C38", "sender_id": "ME", "ts": 1787973000, "body": "Hi which floor."},
        {"id": "M01575", "chat_id": "C38", "sender_id": "P01", "ts": 1787973010, "body": "Flat 34"},
    ]
    res = _evaluate_closing_message(
        msg=messages[2],
        source_msg=source_msg,
        chat_id="C38",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="question",
        title="We discuss this Saturday ?",
        is_group=False,
        messages=messages,
    )
    assert res is None, "Different meeting outside valid window must not close obligation"


def test_phase1_negative_no_preceding_arrival_question():
    source_msg = {
        "id": "M00524",
        "chat_id": "C38",
        "sender_id": "ME",
        "ts": 1787670073,
        "body": "We discuss this Saturday ?",
    }
    messages = [
        source_msg,
        {"id": "M1", "chat_id": "C38", "sender_id": "ME", "ts": 1787973000, "body": "Great see you then"},
        {"id": "M2", "chat_id": "C38", "sender_id": "P01", "ts": 1787973010, "body": "Flat 34"},
    ]
    res = _evaluate_closing_message(
        msg=messages[2],
        source_msg=source_msg,
        chat_id="C38",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="question",
        title="We discuss this Saturday ?",
        is_group=False,
        messages=messages,
    )
    assert res is None, "Venue message without preceding arrival query must not close meeting"


def test_phase1_negative_unauthorized_participant():
    source_msg = {
        "id": "M00524",
        "chat_id": "C38",
        "sender_id": "ME",
        "ts": 1787670073,
        "body": "We discuss this Saturday ?",
    }
    messages = [
        source_msg,
        {"id": "M1", "chat_id": "C38", "sender_id": "ME", "ts": 1787973000, "body": "Hi which floor."},
        {"id": "M2", "chat_id": "C38", "sender_id": "P99_STRANGER", "ts": 1787973010, "body": "Flat 34"},
    ]
    res = _evaluate_closing_message(
        msg=messages[2],
        source_msg=source_msg,
        chat_id="C38",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="question",
        title="We discuss this Saturday ?",
        is_group=False,
        messages=messages,
    )
    assert res is None, "Venue message from unauthorized participant must not close meeting"


def test_phase1_negative_outside_valid_episode_window():
    source_msg = {
        "id": "M00524",
        "chat_id": "C38",
        "sender_id": "ME",
        "ts": 1787670073,
        "body": "We discuss this Saturday ?",
    }
    messages = [
        source_msg,
        {"id": "M1", "chat_id": "C38", "sender_id": "ME", "ts": 1787670073 + 12 * 86400, "body": "Hi which floor."},
        {"id": "M2", "chat_id": "C38", "sender_id": "P01", "ts": 1787670073 + 12 * 86400 + 10, "body": "Flat 34"},
    ]
    res = _evaluate_closing_message(
        msg=messages[2],
        source_msg=source_msg,
        chat_id="C38",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="question",
        title="We discuss this Saturday ?",
        is_group=False,
        messages=messages,
    )
    assert res is None, "Venue coordination outside valid 7-day episode window must not close meeting"


def test_phase2_generic_so_lovely_rejected():
    source_msg = {
        "id": "M01874",
        "chat_id": "C73",
        "sender_id": "P39",
        "ts": 1788018286,
        "body": "You both coming for dinner on 31st Aug / Monday right? @249417682946514 @44444590182763",
    }
    closing_msg = {
        "id": "M04373",
        "chat_id": "C73",
        "sender_id": "P13",
        "ts": 1788634882,
        "body": "So lovely!",
    }
    res = _evaluate_closing_message(
        msg=closing_msg,
        source_msg=source_msg,
        chat_id="C73",
        owner="ME",
        requester="P39",
        authenticated_delegates=set(),
        loop_type="question",
        title="Dinner on Monday 31st Aug",
        is_group=True,
        messages=[source_msg, closing_msg],
    )
    assert res is None, "Generic 'So lovely!' without explicit event linkage must not close obligation"


def test_phase2_appreciation_for_another_event():
    source_msg = {
        "id": "M1",
        "chat_id": "C50",
        "sender_id": "P20",
        "ts": 1000,
        "body": "Dinner on Saturday?",
    }
    closing_msg = {
        "id": "M2",
        "chat_id": "C50",
        "sender_id": "P20",
        "ts": 1000 + 4 * 86400,
        "body": "Thanks for the book you sent!",
    }
    res = _evaluate_closing_message(
        msg=closing_msg,
        source_msg=source_msg,
        chat_id="C50",
        owner="ME",
        requester="P20",
        authenticated_delegates=set(),
        loop_type="question",
        title="Dinner on Saturday?",
        is_group=True,
        messages=[source_msg, closing_msg],
    )
    assert res is None, "Appreciation for unrelated matter must not close dinner event"


def test_phase2_event_date_not_passed():
    source_msg = {
        "id": "M1",
        "chat_id": "C50",
        "sender_id": "P20",
        "ts": 1000,
        "body": "Dinner on 31st Aug?",
    }
    closing_msg = {
        "id": "M2",
        "chat_id": "C50",
        "sender_id": "P20",
        "ts": 1050, # Before the dinner occurred
        "body": "So lovely of you to invite us!",
    }
    res = _evaluate_closing_message(
        msg=closing_msg,
        source_msg=source_msg,
        chat_id="C50",
        owner="ME",
        requester="P20",
        authenticated_delegates=set(),
        loop_type="question",
        title="Dinner on 31st Aug?",
        is_group=True,
        messages=[source_msg, closing_msg],
    )
    assert res is None, "Appreciation before event date must not close obligation"


def test_phase2_unrelated_group_participant():
    source_msg = {
        "id": "M1",
        "chat_id": "C50",
        "sender_id": "P20",
        "ts": 1000,
        "body": "Dinner on Monday?",
    }
    closing_msg = {
        "id": "M2",
        "chat_id": "C50",
        "sender_id": "P99_OUTSIDER",
        "ts": 1000 + 5 * 86400,
        "body": "Great pictures!",
    }
    res = _evaluate_closing_message(
        msg=closing_msg,
        source_msg=source_msg,
        chat_id="C50",
        owner="ME",
        requester="P20",
        authenticated_delegates=set(),
        loop_type="question",
        title="Dinner on Monday?",
        is_group=True,
        messages=[source_msg, closing_msg],
    )
    assert res is None, "Message from non-participant outsider must not close event"


def test_phase3_separate_referrals_different_beneficiaries():
    # Two identical referral messages in different chats must represent SEPARATE obligations
    # when their surrounding contexts identify different beneficiaries
    chatA = {"chat_id": "C_Alice", "is_group": True, "participants": ["ME", "P1"]}
    chatB = {"chat_id": "C_Brenda", "is_group": True, "participants": ["ME", "P2"]}
    dataset = {
        "messages_by_chat": {
            "C_Alice": [
                {"id": "M_A1", "sender_id": "P1", "body": "Shared Alice Johnson contact for performance marketing", "ts": 1000},
                {"id": "M_A2", "sender_id": "ME", "body": "sure we'll connect with her. thanks for ref!", "ts": 1020},
            ],
            "C_Brenda": [
                {"id": "M_B1", "sender_id": "P2", "body": "Shared Brenda Vance contact for packaging design", "ts": 1100},
                {"id": "M_B2", "sender_id": "ME", "body": "sure we'll connect with her. thanks for ref!", "ts": 1120},
            ],
        }
    }
    context = MockContext(chats={"C_Alice": chatA, "C_Brenda": chatB}, dataset=dataset)

    candA = {"message_id": "M_A2", "chat_id": "C_Alice", "sender_id": "ME", "category": "commitment"}
    msgA = dataset["messages_by_chat"]["C_Alice"][1]
    is_actA, _, ownerA, _ = _evaluate_fallback_actionability(candA, msgA, context, chatA)
    assert is_actA is True, "Referral commitment for Alice in Chat A must be actionable"
    assert ownerA == "ME"

    candB = {"message_id": "M_B2", "chat_id": "C_Brenda", "sender_id": "ME", "category": "commitment"}
    msgB = dataset["messages_by_chat"]["C_Brenda"][1]
    is_actB, _, ownerB, _ = _evaluate_fallback_actionability(candB, msgB, context, chatB)
    assert is_actB is True, "Referral commitment for Brenda in Chat B must be actionable (separate beneficiary)"
    assert ownerB == "ME"


def test_phase3_unrelated_earlier_scheduling_language_negative():
    # Unrelated earlier scheduling words must NOT make an isolated transit message actionable
    chat = {"chat_id": "C100", "is_group": False, "participants": ["ME", "P10"]}

    # Case A: Earlier message contained "works" in a non-scheduling context
    datasetA = {
        "messages_by_chat": {
            "C100": [
                {"id": "M1", "sender_id": "P10", "body": "That new monitor works great", "ts": 1000},
                {"id": "M2", "sender_id": "P10", "body": "3 pm ill reach", "ts": 1050},
            ]
        }
    }
    contextA = MockContext(chats={"C100": chat}, dataset=datasetA)
    candA = {"message_id": "M2", "chat_id": "C100", "sender_id": "P10", "category": "commitment"}
    msgA = datasetA["messages_by_chat"]["C100"][1]
    is_actA, _, _, _ = _evaluate_fallback_actionability(candA, msgA, contextA, chat)
    assert is_actA is False, "Incidental 'works' in non-scheduling context must not make transit message actionable"

    # Case B: Scheduling proposal from days earlier outside conversation session (> 2 hours)
    datasetB = {
        "messages_by_chat": {
            "C100": [
                {"id": "M3", "sender_id": "ME", "body": "When can we meet?", "ts": 1000},
                {"id": "M4", "sender_id": "P10", "body": "3 pm ill reach", "ts": 1000 + 3 * 86400}, # 3 days later
            ]
        }
    }
    contextB = MockContext(chats={"C100": chat}, dataset=datasetB)
    candB = {"message_id": "M4", "chat_id": "C100", "sender_id": "P10", "category": "commitment"}
    msgB = datasetB["messages_by_chat"]["C100"][1]
    is_actB, _, _, _ = _evaluate_fallback_actionability(candB, msgB, contextB, chat)
    assert is_actB is False, "Scheduling message from 3 days earlier must not make transit message actionable"

    # Case C: Genuine immediate scheduling proposal (M02903) remains actionable
    datasetC = {
        "messages_by_chat": {
            "C100": [
                {"id": "M5", "sender_id": "ME", "body": "Anytime 3pm onwards works", "ts": 1000},
                {"id": "M6", "sender_id": "P10", "body": "3 pm ill reach", "ts": 1029}, # 29 seconds later
            ]
        }
    }
    contextC = MockContext(chats={"C100": chat}, dataset=datasetC)
    candC = {"message_id": "M6", "chat_id": "C100", "sender_id": "P10", "category": "commitment"}
    msgC = datasetC["messages_by_chat"]["C100"][1]
    is_actC, _, ownerC, _ = _evaluate_fallback_actionability(candC, msgC, contextC, chat)
    assert is_actC is True, "Immediate response to scheduling proposal must be actionable"
    assert ownerC == "ME"


# --- Narrow Appointment Arrival Closure (from Batch 14.3) (9 tests) ---

def test_1_m02903_style_scheduled_arrival_followed_by_eta_inquiry_and_arrival_response():
    """1. M02903-style scheduled arrival followed by an explicit ETA inquiry and valid arrival response."""
    source_msg = {
        "id": "M02903",
        "chat_id": "C114",
        "sender_id": "P210",
        "ts": 1788324729,
        "body": "3 pm ill reach",
    }
    messages = [
        {"id": "M02899", "chat_id": "C114", "sender_id": "ME", "ts": 1788324708, "body": "Office ? Friday 2nd half karte hain ."},
        {"id": "M02900", "chat_id": "C114", "sender_id": "P210", "ts": 1788324710, "body": "Friday what time?"},
        {"id": "M02902", "chat_id": "C114", "sender_id": "ME", "ts": 1788324723, "body": "Anytime 3pm onwards works"},
        source_msg, # M02903
        {"id": "M03995", "chat_id": "C114", "sender_id": "ME", "ts": 1788529711, "body": "8th floor."},
        {"id": "M03996", "chat_id": "C114", "sender_id": "ME", "ts": 1788529716, "body": "eta ?"},
        {"id": "M03998", "chat_id": "C114", "sender_id": "P210", "ts": 1788529789, "body": "Elevator"},
    ]
    chat = {"chat_id": "C114", "is_group": False, "participants": ["ME", "P210"], "messages": messages}
    context = MockContext(chats={"C114": chat})

    res = _evaluate_closing_message(
        msg=messages[6], # M03998
        source_msg=source_msg,
        chat_id="C114",
        owner="P210",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="event",
        title="3 pm ill reach",
        is_group=False,
        messages=messages,
    )
    assert res is not None, "M03998 ('Elevator') answering M03996 ('eta ?') must close appointment"
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M03998"


def test_2_arrival_response_without_preceding_inquiry_remains_open():
    """2. Arrival response without a preceding inquiry — must remain open."""
    source_msg = {
        "id": "M02903",
        "chat_id": "C114",
        "sender_id": "P210",
        "ts": 1788324729,
        "body": "3 pm ill reach",
    }
    messages = [
        source_msg,
        {"id": "M1", "chat_id": "C114", "sender_id": "ME", "ts": 1788529710, "body": "Sounds good"},
        {"id": "M2", "chat_id": "C114", "sender_id": "P210", "ts": 1788529789, "body": "Elevator"},
    ]
    res = _evaluate_closing_message(
        msg=messages[2],
        source_msg=source_msg,
        chat_id="C114",
        owner="P210",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="event",
        title="3 pm ill reach",
        is_group=False,
        messages=messages,
    )
    assert res is None, "Elevator response without preceding arrival inquiry must not close obligation"


def test_3_generic_elevator_or_outside_message_unrelated_to_appointment():
    """3. Generic elevator or outside message unrelated to an appointment — must remain open."""
    source_msg = {
        "id": "M_TASK",
        "chat_id": "C114",
        "sender_id": "ME",
        "ts": 1788324729,
        "body": "Please review the updated financial projections sheet",
    }
    messages = [
        source_msg,
        {"id": "M_Q", "chat_id": "C114", "sender_id": "ME", "ts": 1788529716, "body": "eta ?"},
        {"id": "M_A", "chat_id": "C114", "sender_id": "P210", "ts": 1788529789, "body": "Elevator"},
    ]
    res = _evaluate_closing_message(
        msg=messages[2],
        source_msg=source_msg,
        chat_id="C114",
        owner="P210",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="task",
        title="Review financial projections sheet",
        is_group=False,
        messages=messages,
    )
    assert res is None, "Generic elevator message for a document task must never close it"


def test_4_response_from_unauthorised_or_unrelated_participant():
    """4. Response from an unauthorised or unrelated participant — must remain open."""
    source_msg = {
        "id": "M02903",
        "chat_id": "C114",
        "sender_id": "P210",
        "ts": 1788324729,
        "body": "3 pm ill reach",
    }
    messages = [
        source_msg,
        {"id": "M_Q", "chat_id": "C114", "sender_id": "ME", "ts": 1788529716, "body": "eta ?"},
        {"id": "M_A", "chat_id": "C114", "sender_id": "P999", "ts": 1788529789, "body": "Elevator"}, # outsider
    ]
    res = _evaluate_closing_message(
        msg=messages[2],
        source_msg=source_msg,
        chat_id="C114",
        owner="P210",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="event",
        title="3 pm ill reach",
        is_group=False,
        messages=messages,
    )
    assert res is None, "Arrival message from unauthorized non-obligor must not close appointment"


def test_5_arrival_response_outside_permitted_time_window():
    """5. Arrival response outside the permitted time window — must remain open."""
    source_msg = {
        "id": "M02903",
        "chat_id": "C114",
        "sender_id": "P210",
        "ts": 1788324729,
        "body": "3 pm ill reach",
    }
    # Case A: Response > 30 minutes after inquiry (e.g. 45 mins)
    messages_a = [
        source_msg,
        {"id": "M_Q", "chat_id": "C114", "sender_id": "ME", "ts": 1788529716, "body": "eta ?"},
        {"id": "M_A", "chat_id": "C114", "sender_id": "P210", "ts": 1788529716 + 2700, "body": "Elevator"}, # 45 mins later
    ]
    res_a = _evaluate_closing_message(
        msg=messages_a[2],
        source_msg=source_msg,
        chat_id="C114",
        owner="P210",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="event",
        title="3 pm ill reach",
        is_group=False,
        messages=messages_a,
    )
    assert res_a is None, "Arrival message > 30 minutes after inquiry must not close obligation"

    # Case B: Response > 7 days after appointment commitment (e.g. 10 days later)
    messages_b = [
        source_msg,
        {"id": "M_Q2", "chat_id": "C114", "sender_id": "ME", "ts": 1788324729 + 10 * 86400, "body": "eta ?"},
        {"id": "M_A2", "chat_id": "C114", "sender_id": "P210", "ts": 1788324729 + 10 * 86400 + 60, "body": "Elevator"},
    ]
    res_b = _evaluate_closing_message(
        msg=messages_b[2],
        source_msg=source_msg,
        chat_id="C114",
        owner="P210",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="event",
        title="3 pm ill reach",
        is_group=False,
        messages=messages_b,
    )
    assert res_b is None, "Arrival message > 7 days after commitment must not close obligation"


def test_6_group_chat_arrival_chatter_rejected():
    """6. Group-chat arrival chatter — must not use this new direct-chat closure path."""
    source_msg = {
        "id": "M_GRP",
        "chat_id": "C_GRP",
        "sender_id": "P1",
        "ts": 1788324729,
        "body": "3 pm ill reach",
    }
    messages = [
        source_msg,
        {"id": "M_Q", "chat_id": "C_GRP", "sender_id": "ME", "ts": 1788529716, "body": "eta ?"},
        {"id": "M_A", "chat_id": "C_GRP", "sender_id": "P1", "ts": 1788529789, "body": "Elevator"},
    ]
    res = _evaluate_closing_message(
        msg=messages[2],
        source_msg=source_msg,
        chat_id="C_GRP",
        owner="P1",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="event",
        title="3 pm ill reach",
        is_group=True, # Group chat!
        messages=messages,
    )
    assert res is None, "Group-chat arrival message must not close under direct-chat Case D"


def test_7_existing_m00524_case_d_remains_valid():
    """7. Existing M00524 Case D behaviour remains valid."""
    source_msg = {
        "id": "M00524",
        "chat_id": "C38",
        "sender_id": "ME",
        "ts": 1787670073,
        "body": "Sorry missed the call. Was pretty sick . Spice vc is also same stage as nath. We discuss this Saturday ?",
    }
    messages = [
        source_msg,
        {"id": "M01574", "chat_id": "C38", "sender_id": "ME", "ts": 1787973000, "body": "Hi which floor."},
        {"id": "M01575", "chat_id": "C38", "sender_id": "P01", "ts": 1787973010, "body": "Flat 34"},
        {"id": "M01576", "chat_id": "C38", "sender_id": "P01", "ts": 1787973020, "body": "3rd floor"},
    ]
    chat = {"chat_id": "C38", "is_group": False, "participants": ["ME", "P01"], "messages": messages}
    context = MockContext(chats={"C38": chat})

    res = _evaluate_closing_message(
        msg=messages[2], # M01575
        source_msg=source_msg,
        chat_id="C38",
        owner="P01",
        requester="ME",
        authenticated_delegates=set(),
        loop_type="question",
        title="We discuss this Saturday ?",
        is_group=False,
        messages=messages,
    )
    assert res is not None, "M00524 arrival coordination ('Flat 34') must remain valid"
    status, closed_by = res
    assert status == "done"
    assert closed_by == "M01575"


def test_8_m01874_remains_open():
    """8. M01874 remains open (Case E is NOT restored)."""
    source_msg = {
        "id": "M01874",
        "chat_id": "C73",
        "sender_id": "P39",
        "ts": 1788018286,
        "body": "You both coming for dinner on 31st Aug / Monday right? @249417682946514 @44444590182763",
    }
    messages = [
        source_msg,
        {"id": "M04373", "chat_id": "C73", "sender_id": "P13", "ts": 1788548482, "body": "So lovely!"},
    ]
    chat = {"chat_id": "C73", "is_group": True, "participants": ["ME", "P13", "P39", "P51"], "messages": messages}
    context = MockContext(chats={"C73": chat})

    status, closed_by = verify_full_chat_closure(
        source_msg=source_msg,
        chat_id="C73",
        owner="ME",
        loop_type="event",
        title="You both coming for dinner on 31st Aug / Monday right?",
        context=context,
    )
    assert status == "open", "M01874 must remain open; Case E must not be restored"
    assert closed_by is None


def test_9_other_arrival_responses_with_inquiry():
    """9. Verify lift, lobby, outside, at the gate, reached when answering arrival inquiry."""
    source_msg = {
        "id": "M_SCHED",
        "chat_id": "C114",
        "sender_id": "P210",
        "ts": 1788324729,
        "body": "3 pm ill reach",
    }
    arrival_tokens = ["lift", "lobby", "outside", "at the gate", "downstairs", "reached", "just reached"]
    for token in arrival_tokens:
        messages = [
            source_msg,
            {"id": "M_Q", "chat_id": "C114", "sender_id": "ME", "ts": 1788529716, "body": "Where are you?"},
            {"id": "M_ANS", "chat_id": "C114", "sender_id": "P210", "ts": 1788529750, "body": token},
        ]
        res = _evaluate_closing_message(
            msg=messages[2],
            source_msg=source_msg,
            chat_id="C114",
            owner="P210",
            requester="ME",
            authenticated_delegates=set(),
            loop_type="event",
            title="3 pm ill reach",
            is_group=False,
            messages=messages,
        )
        assert res is not None, f"Token '{token}' answering 'Where are you?' must close appointment"
        status, closed_by = res
        assert status == "done"
        assert closed_by == "M_ANS"

