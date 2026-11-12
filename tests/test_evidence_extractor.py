"""Unit and regression tests for CandidateEvidenceExtractor."""

import copy
from pathlib import Path
import pytest

from src.extraction.evidence_extractor import (
    ALL_CATEGORIES,
    CandidateEvidenceExtractor,
    extract_candidate_evidence,
    extract_message_evidence,
    is_relevant_group_candidate,
    is_syndicated_broadcast,
    mask_url_query_delimiters,
)
from src.ingestion.loader import load_dataset
from src.preprocessing.context_builder import build_conversation_context


@pytest.fixture(scope="module")
def example_context():
    """Build ConversationContext for example dataset once for read-only tests."""
    dataset = load_dataset(Path("example"))
    return build_conversation_context(dataset)


@pytest.fixture(scope="module")
def dev_context():
    """Build ConversationContext for development dataset once for read-only tests."""
    dataset = load_dataset(Path("data/dev"))
    return build_conversation_context(dataset)


def test_commitment_indicators():
    """Verify detection of English and Hinglish commitment indicators."""
    extractor = CandidateEvidenceExtractor()

    # English commitments
    msg1 = {"id": "M1", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "I'll book the homestay, promise"}
    evidence1 = extractor.extract_message(msg1)
    cats1 = {e["category"] for e in evidence1}
    assert "commitment" in cats1
    commit_record = next(e for e in evidence1 if e["category"] == "commitment")
    assert any("i'll" in sig.lower() for sig in commit_record["matched_signals"])
    assert any("promise" in sig.lower() for sig in commit_record["matched_signals"])

    # "I'm on it"
    msg2 = {"id": "M2", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "Halcyon deck is due Friday, I'm on it"}
    evidence2 = extractor.extract_message(msg2)
    assert any(e["category"] == "commitment" for e in evidence2)

    # Hinglish commitments
    msg3 = {"id": "M3", "chat_id": "C1", "sender_id": "P1", "ts": 102, "body": "Madam kal shaam tak bhej dunga"}
    evidence3 = extractor.extract_message(msg3)
    assert any(e["category"] == "commitment" for e in evidence3)
    c3 = next(e for e in evidence3 if e["category"] == "commitment")
    assert any("bhej dunga" in sig.lower() for sig in c3["matched_signals"])


def test_question_indicators():
    """Verify detection of questions via question marks and question openers."""
    extractor = CandidateEvidenceExtractor()

    # Explicit question mark
    msg1 = {"id": "M1", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "Did you look at Coorg?"}
    ev1 = extractor.extract_message(msg1)
    assert any(e["category"] == "question" for e in ev1)

    # Short query with question mark
    msg2 = {"id": "M2", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "Coorg?"}
    ev2 = extractor.extract_message(msg2)
    assert any(e["category"] == "question" for e in ev2)

    # Hinglish question markers
    msg3 = {"id": "M3", "chat_id": "C1", "sender_id": "P1", "ts": 102, "body": "Das ji, proofs kab tak"}
    ev3 = extractor.extract_message(msg3)
    assert any(e["category"] == "question" for e in ev3)


def test_followup_indicators():
    """Verify detection of follow-up indicators."""
    extractor = CandidateEvidenceExtractor()

    msg1 = {"id": "M1", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "Gentle reminder to send the October invoice"}
    ev1 = extractor.extract_message(msg1)
    assert any(e["category"] == "followup" for e in ev1)

    msg2 = {"id": "M2", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "Let me know your thoughts on this"}
    ev2 = extractor.extract_message(msg2)
    assert any(e["category"] == "followup" for e in ev2)

    msg3 = {"id": "M3", "chat_id": "C1", "sender_id": "P1", "ts": 102, "body": "Following up on the vendor proposal"}
    ev3 = extractor.extract_message(msg3)
    assert any(e["category"] == "followup" for e in ev3)


def test_deadline_indicators():
    """Verify detection of deadline and time-limit signals."""
    extractor = CandidateEvidenceExtractor()

    # Due day
    msg1 = {"id": "M1", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "Pitch deck is due Friday"}
    ev1 = extractor.extract_message(msg1)
    assert any(e["category"] == "deadline" for e in ev1)

    # By time
    msg2 = {"id": "M2", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "Arrange car at jodhpur park by 4pm"}
    ev2 = extractor.extract_message(msg2)
    assert any(e["category"] == "deadline" for e in ev2)

    # By month / end of week
    msg3 = {"id": "M3", "chat_id": "C1", "sender_id": "P1", "ts": 102, "body": "Target to close 10 clients by March"}
    ev3 = extractor.extract_message(msg3)
    assert any(e["category"] == "deadline" for e in ev3)

    # Hinglish deadline
    msg4 = {"id": "M4", "chat_id": "C1", "sender_id": "P1", "ts": 103, "body": "Madam kal shaam tak bhej dunga"}
    ev4 = extractor.extract_message(msg4)
    assert any(e["category"] == "deadline" for e in ev4)


def test_goal_indicators():
    """Verify detection of goals, milestones, and planning statements."""
    extractor = CandidateEvidenceExtractor()

    msg1 = {"id": "M1", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "10 retainer clients by March is the target"}
    ev1 = extractor.extract_message(msg1)
    assert any(e["category"] == "goal" for e in ev1)

    msg2 = {"id": "M2", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "I am training for a half marathon"}
    ev2 = extractor.extract_message(msg2)
    assert any(e["category"] == "goal" for e in ev2)

    # "I'll send you my training plan for the marathon" is a commitment, NOT automatically a goal
    msg2_plan = {"id": "M2_PLAN", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "I'll send you my training plan for the marathon"}
    ev2_plan = extractor.extract_message(msg2_plan)
    assert not any(e["category"] == "goal" for e in ev2_plan)
    assert any(e["category"] == "commitment" for e in ev2_plan)

    msg3 = {"id": "M3", "chat_id": "C1", "sender_id": "P1", "ts": 102, "body": "Our goal is to reach 100k ARR"}
    ev3 = extractor.extract_message(msg3)
    assert any(e["category"] == "goal" for e in ev3)


def test_person_and_organisation_mentions(example_context):
    """Verify context-aware person and organisation mention identification."""
    extractor = CandidateEvidenceExtractor()

    # M018 in example has both person (Neha) and org (Halcyon, Zest)
    msg = {
        "id": "M018", "chat_id": "C04", "sender_id": "ME", "ts": 100,
        "body": "Pitch deck for Halcyon is due Friday, I'm on it. Neha let me think about Zest"
    }
    ev = extractor.extract_message(msg, context=example_context)
    cats = {e["category"] for e in ev}

    assert "person_mention" in cats
    person_rec = next(e for e in ev if e["category"] == "person_mention")
    assert any("neha" in sig.lower() for sig in person_rec["matched_signals"])

    assert "organisation_mention" in cats
    org_rec = next(e for e in ev if e["category"] == "organisation_mention")
    assert any("zest" in sig.lower() or "halcyon" in sig.lower() for sig in org_rec["matched_signals"])


def test_multiple_categories_for_one_message():
    """Verify that a multi-intent message produces one record per matching category."""
    msg = {
        "id": "M100", "chat_id": "C1", "sender_id": "P1", "sender_name": "Arjun",
        "ts": 1787535000,
        "body": "Can you check the pitch deck by Friday? I'll send it to Dr. Roy."
    }
    extractor = CandidateEvidenceExtractor()
    evidence = extractor.extract_message(msg)

    categories = [e["category"] for e in evidence]
    # Expect: question ("Can you", "?"), deadline ("by Friday"), commitment ("I'll send"), person_mention ("Dr. Roy")
    assert "question" in categories
    assert "deadline" in categories
    assert "commitment" in categories
    assert "person_mention" in categories

    # All records share the same message metadata
    for e in evidence:
        assert e["message_id"] == "M100"
        assert e["chat_id"] == "C1"
        assert e["sender_id"] == "P1"
        assert e["timestamp"] == 1787535000
        assert e["original_text"] == msg["body"]


def test_duplicate_prevention_same_category():
    """Verify that multiple matching signals in the same category do not create duplicate records."""
    msg = {
        "id": "M200", "chat_id": "C1", "sender_id": "P1", "ts": 100,
        # Multiple commitment signals in one message
        "body": "I will do it and I'll send the file, promise! Leave it to me."
    }
    extractor = CandidateEvidenceExtractor()
    evidence = extractor.extract_message(msg)

    # Exactly ONE record for commitment
    commit_records = [e for e in evidence if e["category"] == "commitment"]
    assert len(commit_records) == 1

    # But matched_signals contains multiple distinct signals
    signals = commit_records[0]["matched_signals"]
    assert len(signals) >= 2


def test_unicode_and_hinglish_preservation():
    """Verify that diverse scripts, emojis, and multiline texts are preserved verbatim."""
    body_text = (
        "Line 1: Haan bhaiya, kal shaam tak bhej dunga 🙏\n"
        "Line 2: वन्दे मातरम्\n"
        "Line 3: Kaomoji (⁠˃͈⁠ ⁠દ⁠ ⁠˂͈⁠ ⁠༶⁠ ⁠) and *bold* text."
    )
    msg = {"id": "M_UNI", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": body_text}
    extractor = CandidateEvidenceExtractor()
    evidence = extractor.extract_message(msg)

    assert len(evidence) > 0
    for e in evidence:
        assert e["original_text"] == body_text


def test_empty_dataset_and_chats():
    """Verify that empty contexts, chats with zero messages, and empty messages return empty results."""
    extractor = CandidateEvidenceExtractor()

    # Empty message body
    assert extractor.extract_message({"id": "M_EMPTY", "body": ""}) == []
    assert extractor.extract_message({"id": "M_NONE", "body": None}) == []

    # None context
    assert extractor.extract(None) == []

    # Empty context
    empty_context = build_conversation_context({
        "chats": [], "messages": [], "seed": {}, "chats_by_id": {},
        "messages_by_id": {}, "messages_by_chat": {}, "participants": {}
    })
    assert extractor.extract(empty_context) == []


def test_deterministic_ordering(example_context):
    """Verify that repeated extraction runs produce strictly identical output order."""
    extractor = CandidateEvidenceExtractor()

    run1 = extractor.extract(example_context)
    run2 = extractor.extract(example_context)

    assert len(run1) > 0
    assert len(run1) == len(run2)

    for r1, r2 in zip(run1, run2):
        assert r1 == r2

    # Verify chronological sorting
    for i in range(len(run1) - 1):
        ts1 = run1[i]["timestamp"] or -1
        ts2 = run1[i + 1]["timestamp"] or -1
        assert ts1 <= ts2


def test_no_mutation_of_source_messages(example_context):
    """Verify that source messages and context are not mutated during extraction."""
    context_copy = copy.deepcopy(example_context)
    first_msg_before = copy.deepcopy(example_context.dataset["messages"][0])

    extractor = CandidateEvidenceExtractor()
    _ = extractor.extract(example_context)

    first_msg_after = example_context.dataset["messages"][0]
    assert first_msg_before == first_msg_after
    assert example_context.summary == context_copy.summary


def test_convenience_functions(example_context):
    """Verify extract_candidate_evidence and extract_message_evidence convenience functions."""
    all_evidence = extract_candidate_evidence(example_context)
    assert len(all_evidence) > 0

    first_msg = example_context.dataset["messages"][0]
    single_evidence = extract_message_evidence(first_msg, context=example_context)
    assert isinstance(single_evidence, list)


def test_negative_person_directive_call_me(example_context):
    """Verify that directive phrases like 'call me' or 'ask them' do not produce person mentions."""
    extractor = CandidateEvidenceExtractor()

    msg1 = {"id": "M_NEG_1", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "Please call me tomorrow"}
    ev1 = extractor.extract_message(msg1, context=example_context)
    assert not any(e["category"] == "person_mention" for e in ev1)

    msg2 = {"id": "M_NEG_2", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "Ask them about the updates"}
    ev2 = extractor.extract_message(msg2, context=example_context)
    assert not any(e["category"] == "person_mention" for e in ev2)


def test_negative_organisation_generic_words(example_context):
    """Verify generic words and sub-tokens do not trigger false organisation mentions."""
    extractor = CandidateEvidenceExtractor()

    # Generic word 'school'
    msg_school = {"id": "M_SCH", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "I went to school today"}
    ev_school = extractor.extract_message(msg_school, context=example_context)
    assert not any(e["category"] == "organisation_mention" for e in ev_school)

    # Word 'Design' from seeded 'Brightside Design' must not trigger on 'The design looks good'
    msg_design = {"id": "M_DES", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "The design looks good"}
    ev_design = extractor.extract_message(msg_design, context=example_context)
    assert not any(e["category"] == "organisation_mention" for e in ev_design)


def test_full_name_matching_vs_unrelated_partial_occurrence():
    """Verify full participant name matches and partial name occurrence does not match."""
    # Extractor with a known participant named 'Neha Sharma'
    extractor = CandidateEvidenceExtractor(extra_person_names=["Neha Sharma"])

    # Full name in message matches
    msg_full = {"id": "M_FULL", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "Neha Sharma sent the draft"}
    ev_full = extractor.extract_message(msg_full)
    assert any(e["category"] == "person_mention" and "Neha Sharma" in e["matched_signals"] for e in ev_full)

    # Unrelated occurrence of only 'Sharma' must not match 'Neha Sharma'
    msg_partial = {"id": "M_PART", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "Sharma went to the market today"}
    ev_partial = extractor.extract_message(msg_partial)
    assert not any(e["category"] == "person_mention" for e in ev_partial)


def test_negative_trivial_desire_vs_strong_goal():
    """Verify 'I want to buy lunch' is excluded while stronger goal signals match."""
    extractor = CandidateEvidenceExtractor()

    # Trivial immediate action excluded
    msg_lunch = {"id": "M_LUNCH", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "I want to buy lunch"}
    ev_lunch = extractor.extract_message(msg_lunch)
    assert not any(e["category"] == "goal" for e in ev_lunch)

    # Stronger goal signals match
    msg_goal = {"id": "M_STRONG", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "My goal is to buy lunch"}
    ev_goal = extractor.extract_message(msg_goal)
    assert any(e["category"] == "goal" for e in ev_goal)

    msg_achieve = {"id": "M_ACH", "chat_id": "C1", "sender_id": "P1", "ts": 102, "body": "I want to achieve 100k ARR"}
    ev_achieve = extractor.extract_message(msg_achieve)
    assert any(e["category"] == "goal" for e in ev_achieve)


def test_goal_precision_marathon_positive():
    """Verify genuine goal declarations regarding marathons and training are captured."""
    extractor = CandidateEvidenceExtractor()

    # 'My goal is to run a marathon'
    msg1 = {"id": "M_G1", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "My goal is to run a marathon"}
    ev1 = extractor.extract_message(msg1)
    assert any(e["category"] == "goal" for e in ev1)

    # 'I am training for a half marathon'
    msg2 = {"id": "M_G2", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "I am training for a half marathon"}
    ev2 = extractor.extract_message(msg2)
    assert any(e["category"] == "goal" for e in ev2)

    # 'I'm training for a marathon'
    msg3 = {"id": "M_G3", "chat_id": "C1", "sender_id": "P1", "ts": 102, "body": "I'm training for a marathon"}
    ev3 = extractor.extract_message(msg3)
    assert any(e["category"] == "goal" for e in ev3)


def test_goal_precision_marathon_negative():
    """Verify that incidental mentions of training plans or marathons do not trigger goals."""
    extractor = CandidateEvidenceExtractor()

    # 'I'll send you my training plan for the marathon' -> commitment, not goal
    msg1 = {"id": "M_NG1", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "I'll send you my training plan for the marathon"}
    ev1 = extractor.extract_message(msg1)
    assert not any(e["category"] == "goal" for e in ev1)
    assert any(e["category"] == "commitment" for e in ev1)

    # General incidental mentions
    msg2 = {"id": "M_NG2", "chat_id": "C1", "sender_id": "P1", "ts": 101, "body": "The marathon route is closed today"}
    ev2 = extractor.extract_message(msg2)
    assert not any(e["category"] == "goal" for e in ev2)

    msg3 = {"id": "M_NG3", "chat_id": "C1", "sender_id": "P1", "ts": 102, "body": "Did you receive the training plan document?"}
    ev3 = extractor.extract_message(msg3)
    assert not any(e["category"] == "goal" for e in ev3)


def test_goal_travel_and_experience_ambitions_positive():
    """Verify approved travel and experience phrases are extracted as goal candidates."""
    extractor = CandidateEvidenceExtractor()
    positive_phrases = [
        "Family trip to Coorg",
        "Trip to Paris",
        "We want to visit Coorg",
        "We want to travel to Japan",
        "Planning a trip to Goa",
        "Family trip to New York",
    ]
    for idx, phrase in enumerate(positive_phrases):
        msg = {"id": f"M_TRV_POS_{idx}", "chat_id": "C1", "sender_id": "P1", "ts": 100 + idx, "body": phrase}
        ev = extractor.extract_message(msg)
        assert any(e["category"] == "goal" for e in ev), f"Failed to match positive travel goal: {phrase!r}"


def test_goal_travel_and_experience_ambitions_negative():
    """Verify generic travel words, workplace terms, contacts, and local errands do not trigger goal candidates."""
    extractor = CandidateEvidenceExtractor()
    negative_phrases = [
        "Family trip to office",
        "Trip to grocery store",
        "We want to visit a client",
        "We want to visit you",
        "We want to visit work",
        "Planning a trip to meet you",
        "I want to travel to the office",
        "I had a great trip",
        "Please visit our website",
        "I travel for business",
        "Trip to the grocery store",
        "We want to visit the doctor",
    ]
    for idx, phrase in enumerate(negative_phrases):
        msg = {"id": f"M_TRV_NEG_{idx}", "chat_id": "C1", "sender_id": "P1", "ts": 200 + idx, "body": phrase}
        ev = extractor.extract_message(msg)
        assert not any(e["category"] == "goal" for e in ev), f"False positive travel goal match on: {phrase!r}"


def test_goal_habit_adoption_and_maintenance_positive():
    """Verify approved habit adoption and maintenance phrases are extracted as goal candidates."""
    extractor = CandidateEvidenceExtractor()
    positive_phrases = [
        "Start running four times a week",
        "Begin daily meditation",
        "Build a running routine",
        "Maintain four runs a week",
        "Continue my running routine",
        "Keep up my meditation routine",
    ]
    for idx, phrase in enumerate(positive_phrases):
        msg = {"id": f"M_HBT_POS_{idx}", "chat_id": "C1", "sender_id": "P1", "ts": 300 + idx, "body": phrase}
        ev = extractor.extract_message(msg)
        assert any(e["category"] == "goal" for e in ev), f"Failed to match positive habit goal: {phrase!r}"


def test_goal_habit_adoption_and_maintenance_negative():
    """Verify descriptive routines and generic activities without adoption/maintenance intent are excluded."""
    extractor = CandidateEvidenceExtractor()
    negative_phrases = [
        "I run four times a week",
        "I meditate every morning",
        "I have a meditation routine",
        "My running routine is four days a week",
        "Just part of my daily routine",
        "Routine checkup scheduled",
        "Running late for meeting",
    ]
    for idx, phrase in enumerate(negative_phrases):
        msg = {"id": f"M_HBT_NEG_{idx}", "chat_id": "C1", "sender_id": "P1", "ts": 400 + idx, "body": phrase}
        ev = extractor.extract_message(msg)
        assert not any(e["category"] == "goal" for e in ev), f"False positive habit goal match on: {phrase!r}"


def test_goal_property_and_financial_milestones_positive():
    """Verify approved property milestones and major savings targets are extracted as goal candidates."""
    extractor = CandidateEvidenceExtractor()
    positive_phrases = [
        "Saving 20 lakhs for down payment",
        "Save for a house",
        "Buy a 3BHK",
        "Purchase a flat",
    ]
    for idx, phrase in enumerate(positive_phrases):
        msg = {"id": f"M_PROP_POS_{idx}", "chat_id": "C1", "sender_id": "P1", "ts": 500 + idx, "body": phrase}
        ev = extractor.extract_message(msg)
        assert any(e["category"] == "goal" for e in ev), f"Failed to match positive property goal: {phrase!r}"


def test_goal_property_and_financial_milestones_negative():
    """Verify everyday shopping, digital files, and household errands do not trigger property goals."""
    extractor = CandidateEvidenceExtractor()
    negative_phrases = [
        "Buy milk",
        "Purchase batteries",
        "Buy lunch",
        "Ordered keyboard",
        "Save the document",
    ]
    for idx, phrase in enumerate(negative_phrases):
        msg = {"id": f"M_PROP_NEG_{idx}", "chat_id": "C1", "sender_id": "P1", "ts": 600 + idx, "body": phrase}
        ev = extractor.extract_message(msg)
        assert not any(e["category"] == "goal" for e in ev), f"False positive property goal match on: {phrase!r}"


def test_goal_athletic_preparation_extended():
    """Verify athletic events and preparation are recognized while generic running statements are excluded."""
    extractor = CandidateEvidenceExtractor()

    # Positive athletic statements
    pos_msgs = [
        "Run the half marathon in January",
        "Doing four runs a week to train for the marathon",
    ]
    for idx, phrase in enumerate(pos_msgs):
        msg = {"id": f"M_ATH_POS_{idx}", "chat_id": "C1", "sender_id": "P1", "ts": 700 + idx, "body": phrase}
        ev = extractor.extract_message(msg)
        assert any(e["category"] == "goal" for e in ev), f"Failed athletic positive: {phrase!r}"

    # Negative non-athletic running statements
    neg_msgs = [
        "Run tests",
        "Run daily",
    ]
    for idx, phrase in enumerate(neg_msgs):
        msg = {"id": f"M_ATH_NEG_{idx}", "chat_id": "C1", "sender_id": "P1", "ts": 800 + idx, "body": phrase}
        ev = extractor.extract_message(msg)
        assert not any(e["category"] == "goal" for e in ev), f"False positive athletic match on: {phrase!r}"


def test_context_derived_full_participant_names():
    """Verify that participant names indexed in ConversationContext are matched as full names."""
    # Build a minimal ConversationContext containing a participant named 'Neha Sharma'
    test_dataset = {
        "chats": [{"id": "C10", "name": "Work Chat", "is_group": False, "participants": ["ME", "P_NEHA"]}],
        "messages": [
            {"id": "M_FULL", "chat_id": "C10", "sender_id": "P_NEHA", "sender_name": "Neha Sharma", "ts": 100, "body": "Neha Sharma sent the draft"},
            {"id": "M_SURNAME", "chat_id": "C10", "sender_id": "P_NEHA", "sender_name": "Neha Sharma", "ts": 101, "body": "Sharma went to the market"},
            {"id": "M_FIRST", "chat_id": "C10", "sender_id": "P_NEHA", "sender_name": "Neha Sharma", "ts": 102, "body": "Neha went to the market"},
        ],
        "seed": {},
        "chats_by_id": {"C10": {"id": "C10", "name": "Work Chat", "is_group": False, "participants": ["ME", "P_NEHA"]}},
        "messages_by_id": {},
        "messages_by_chat": {"C10": []},
        "participants": {
            "P_NEHA": {
                "participant_id": "P_NEHA",
                "sender_names": ["Neha Sharma"],
                "direct_chat_names": ["Neha Sharma"],
                "chats": ["C10"],
                "message_count": 3,
            }
        },
    }
    context = build_conversation_context(test_dataset)
    extractor = CandidateEvidenceExtractor()  # Note: No extra_person_names passed!

    # 1. Full name matches
    msg_full = test_dataset["messages"][0]
    ev_full = extractor.extract_message(msg_full, context=context)
    person_records_full = [e for e in ev_full if e["category"] == "person_mention"]
    assert len(person_records_full) >= 1
    assert any("Neha Sharma" in e["matched_signals"] for e in person_records_full)

    # 2. 'Sharma' alone does not match
    msg_surname = test_dataset["messages"][1]
    ev_surname = extractor.extract_message(msg_surname, context=context)
    assert not any(e["category"] == "person_mention" for e in ev_surname)

    # 3. 'Neha' alone does not match
    msg_first = test_dataset["messages"][2]
    ev_first = extractor.extract_message(msg_first, context=context)
    assert not any(e["category"] == "person_mention" for e in ev_first)

    # 4. Across full context extraction
    all_ev = extractor.extract(context)
    all_persons = [e for e in all_ev if e["category"] == "person_mention"]
    assert len(all_persons) == 1
    assert all_persons[0]["message_id"] == "M_FULL"
    assert "Neha Sharma" in all_persons[0]["matched_signals"]


def test_custom_patterns_and_extra_keywords():
    """Verify that CandidateEvidenceExtractor respects custom patterns and extra keywords."""
    msg = {"id": "M1", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": "custom_trigger activated"}

    # Default extractor won't match "custom_trigger"
    default_extractor = CandidateEvidenceExtractor()
    assert default_extractor.extract_message(msg) == []

    # Custom extractor with extra keyword
    custom_extractor = CandidateEvidenceExtractor(
        extra_keywords={"commitment": ["custom_trigger"]}
    )
    ev = custom_extractor.extract_message(msg)
    assert len(ev) == 1
    assert ev[0]["category"] == "commitment"
    assert "custom_trigger" in ev[0]["matched_signals"]


def test_extraction_on_dev_dataset(dev_context):
    """Verify extraction runs successfully on the full 4502-message development dataset."""
    extractor = CandidateEvidenceExtractor()
    evidence = extractor.extract(dev_context)

    assert len(evidence) > 0
    cats = {e["category"] for e in evidence}
    # All 7 categories should be represented across 4502 messages
    for cat in ALL_CATEGORIES:
        assert cat in cats, f"Category '{cat}' had zero candidate extractions in dev dataset"


def test_rhetorical_and_non_actionable_questions_filtered():
    """Verify that pure rhetorical, pleasantry, and confirmation questions are filtered,
    while legitimate actionable questions are retained."""
    extractor = CandidateEvidenceExtractor()

    # Rhetorical / pleasantry questions that must NOT produce a question candidate
    rhetorical_samples = [
        "How are you?",
        "How are things?",
        "How's it going?",
        "What's up?",
        "How have you been?",
        "Are you free?",
        "You there?",
        "Kya haal hai?",
        "Sab theek?",
        "Sab badhiya?",
        "Right?",
        "Correct?",
        "Hai na?",
        "Haina?",
        "Sahi hai na?",
    ]
    for text in rhetorical_samples:
        msg = {"id": "M_RHET", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": text}
        evidence = extractor.extract_message(msg)
        assert not any(e["category"] == "question" for e in evidence), f"Expected no question candidate for '{text}'"

    # Actionable questions that MUST produce a question candidate
    actionable_samples = [
        "Did you book the tickets?",
        "Can you share the report?",
        "Have you reviewed the PR?",
        "Could you send over the invoice?",
        "Proofs kab tak milenge?",
        "Kab tak bhejoge?",
        "Kya status hai project ka?",
        "How are you? Did you send the invoice?",  # Mixed: pleasantry followed by actionable question
    ]
    for text in actionable_samples:
        msg = {"id": "M_ACT", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": text}
        evidence = extractor.extract_message(msg)
        assert any(e["category"] == "question" for e in evidence), f"Expected question candidate for '{text}'"


def test_open_loop_pattern_enhancements_english_and_hinglish():
    """Verify enhanced English and Hinglish patterns for commitments, questions, followups, deadlines."""
    extractor = CandidateEvidenceExtractor()

    # Commitments
    commit_samples = [
        "I'll make sure the deck is ready",
        "I will ensure everything is done",
        "Leave it to me, I got this",
        "You can count on me",
        "Main dekh lunga tension mat lo",
        "Kal tak bhej dunga",
        "Bhai ho jayega",
    ]
    for text in commit_samples:
        msg = {"id": "M_COM", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": text}
        evidence = extractor.extract_message(msg)
        assert any(e["category"] == "commitment" for e in evidence), f"Expected commitment for '{text}'"

    # Followups
    followup_samples = [
        "Any update on the proposal?",
        "Any progress on this task?",
        "Checking in on the status",
        "Following up on our discussion",
        "Kya status hai iska?",
        "Kuch hua iska?",
    ]
    for text in followup_samples:
        msg = {"id": "M_FOL", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": text}
        evidence = extractor.extract_message(msg)
        assert any(e["category"] == "followup" for e in evidence), f"Expected followup for '{text}'"

    # Deadlines
    deadline_samples = [
        "Submit latest by tomorrow",
        "Need this within 3 days",
        "Deliver by end of day",
        "Is hafte tak submit karna hai",
        "Kal shaam tak finalize karte hain",
    ]
    for text in deadline_samples:
        msg = {"id": "M_DED", "chat_id": "C1", "sender_id": "P1", "ts": 100, "body": text}
        evidence = extractor.extract_message(msg)
        assert any(e["category"] == "deadline" for e in evidence), f"Expected deadline for '{text}'"


def test_group_chat_relevance_filtering():
    """Verify grounded group chat relevance filtering via is_relevant_group_candidate."""
    class MockContext:
        chats = {
            "C_DIRECT": {
                "id": "C_DIRECT",
                "is_group": False,
                "messages": [],
            },
            "C_GROUP": {
                "id": "C_GROUP",
                "is_group": True,
                "participants": ["ME", "P2", "P3"],
                "messages": [
                    {"id": "M_ME_OLD", "chat_id": "C_GROUP", "sender_id": "ME", "ts": 100, "body": "Who has the key?"},
                    {"id": "M_OTHER_OLD", "chat_id": "C_GROUP", "sender_id": "P2", "ts": 101, "body": "Not me"},
                ],
            },
        }
        participants = {
            "ME": {"participant_id": "ME", "sender_names": ["Aditi"]},
            "P2": {"participant_id": "P2", "sender_names": ["Rahul"]},
            "P3": {"participant_id": "P3", "sender_names": ["Pooja"]},
        }

    ctx = MockContext()

    # 1. Existing direct-chat behavior remains unchanged (1:1 chats are always relevant)
    cand_direct = {"category": "question", "matched_signals": ["did you"]}
    msg_direct = {"id": "M_DIR", "chat_id": "C_DIRECT", "sender_id": "P2", "body": "Did you check?"}
    assert is_relevant_group_candidate(cand_direct, msg_direct, context=ctx) is True

    # 2. "Bhai please send the file" does not automatically establish ME as recipient
    cand_bhai = {"category": "question", "matched_signals": ["please send"]}
    msg_bhai = {"id": "M_BHAI", "chat_id": "C_GROUP", "sender_id": "P2", "body": "Bhai please send the file"}
    assert is_relevant_group_candidate(cand_bhai, msg_bhai, context=ctx) is False

    # 3. "Didi please call the vendor" does not automatically establish ME as recipient
    cand_didi = {"category": "question", "matched_signals": ["please call"]}
    msg_didi = {"id": "M_DIDI", "chat_id": "C_GROUP", "sender_id": "P2", "body": "Didi please call the vendor"}
    assert is_relevant_group_candidate(cand_didi, msg_didi, context=ctx) is False

    # 4. A real participant explicitly named in a group message is recognized only when grounded in participant metadata
    # Grounded: "Rahul" is established in ctx.participants for P2
    cand_named = {"category": "question", "matched_signals": ["can you"]}
    msg_grounded = {"id": "M_GROUND", "chat_id": "C_GROUP", "sender_id": "P3", "body": "Rahul please send the report"}
    assert is_relevant_group_candidate(cand_named, msg_grounded, context=ctx) is True

    # Grounded with @mention: @Pooja
    msg_mention = {"id": "M_MENTION", "chat_id": "C_GROUP", "sender_id": "P2", "body": "@Pooja please check status on this"}
    assert is_relevant_group_candidate(cand_named, msg_mention, context=ctx) is True

    # Ungrounded: "Vikram" is not in ctx.participants
    msg_ungrounded = {"id": "M_UNGROUND", "chat_id": "C_GROUP", "sender_id": "P3", "body": "Vikram please send the report"}
    assert is_relevant_group_candidate(cand_named, msg_ungrounded, context=ctx) is False

    # 5. A request explicitly directed to ME using a name genuinely established by supplied metadata is retained
    # "Aditi" is established for ME in ctx.participants
    cand_me_req = {"category": "question", "matched_signals": ["can you"]}
    msg_me_req = {"id": "M_TO_ME", "chat_id": "C_GROUP", "sender_id": "P2", "body": "Aditi can you review the draft?"}
    assert is_relevant_group_candidate(cand_me_req, msg_me_req, context=ctx) is True

    # Directed to ME via @ME
    msg_me_at = {"id": "M_TO_ME_AT", "chat_id": "C_GROUP", "sender_id": "P2", "body": "@ME please review"}
    assert is_relevant_group_candidate(cand_me_req, msg_me_at, context=ctx) is True

    # 6. A message quoting ME's actual message remains eligible
    cand_quote = {"category": "question", "matched_signals": ["what about"]}
    msg_quote = {"id": "M_QUOTE", "chat_id": "C_GROUP", "sender_id": "P2", "quoted_id": "M_ME_OLD", "body": "What about the key?"}
    assert is_relevant_group_candidate(cand_quote, msg_quote, context=ctx) is True

    # Quoting another participant (not ME) without other criteria is false:
    msg_quote_other = {"id": "M_QUOTE_OTHER", "chat_id": "C_GROUP", "sender_id": "P3", "quoted_id": "M_OTHER_OLD", "body": "Okay"}
    assert is_relevant_group_candidate(cand_quote, msg_quote_other, context=ctx) is False

    # 7. A first-person commitment from another participant remains eligible
    cand_commit_en = {"category": "commitment", "matched_signals": ["i'll"]}
    msg_commit_en = {"id": "M_COM_EN", "chat_id": "C_GROUP", "sender_id": "P2", "body": "I'll submit the documents tomorrow"}
    assert is_relevant_group_candidate(cand_commit_en, msg_commit_en, context=ctx) is True

    cand_commit_hi = {"category": "commitment", "matched_signals": ["kar dunga"]}
    msg_commit_hi = {"id": "M_COM_HI", "chat_id": "C_GROUP", "sender_id": "P2", "body": "Shaam tak kar dunga"}
    assert is_relevant_group_candidate(cand_commit_hi, msg_commit_hi, context=ctx) is True

    # 8. An ambiguous group-wide broadcast remains filtered
    cand_bcast = {"category": "question", "matched_signals": ["anyone know"]}
    msg_bcast = {"id": "M_BCAST", "chat_id": "C_GROUP", "sender_id": "P2", "body": "Anyone know where the presentation is?"}
    assert is_relevant_group_candidate(cand_bcast, msg_bcast, context=ctx) is False

    msg_bcast_me = {"id": "M_BCAST_ME", "chat_id": "C_GROUP", "sender_id": "ME", "body": "Does anyone have the link?"}
    assert is_relevant_group_candidate(cand_bcast, msg_bcast_me, context=ctx) is False

    # 9. A candidate sent by ME without actionable relevance is not automatically retained
    # Non-open-loop candidate category (person_mention)
    cand_me_ambient = {"category": "person_mention", "matched_signals": ["Rahul"]}
    msg_me_ambient = {"id": "M_ME_AMB", "chat_id": "C_GROUP", "sender_id": "ME", "body": "Saw Rahul at the cafeteria"}
    assert is_relevant_group_candidate(cand_me_ambient, msg_me_ambient, context=ctx) is False

    # Ambient chatter / unaddressed question sent by ME
    cand_me_chatter = {"category": "followup", "matched_signals": ["update"]}
    msg_me_chatter = {"id": "M_ME_CHAT", "chat_id": "C_GROUP", "sender_id": "ME", "body": "Random message in group"}
    assert is_relevant_group_candidate(cand_me_chatter, msg_me_chatter, context=ctx) is False

    # In contrast, an actionable commitment or deadline by ME is retained
    cand_me_commit = {"category": "commitment", "matched_signals": ["i will"]}
    msg_me_commit = {"id": "M_ME_COM", "chat_id": "C_GROUP", "sender_id": "ME", "body": "I will handle this"}
    assert is_relevant_group_candidate(cand_me_commit, msg_me_commit, context=ctx) is True

    # Informational deadline from ME is filtered (e.g. "Halcyon deck is due Friday", "The presentation deadline is tomorrow", "The deck is due Friday")
    cand_me_deadline_info = {"category": "deadline", "matched_signals": ["due friday"]}
    msg_me_deadline_info1 = {"id": "M_ME_DED_INFO1", "chat_id": "C_GROUP", "sender_id": "ME", "body": "Halcyon deck is due Friday"}
    assert is_relevant_group_candidate(cand_me_deadline_info, msg_me_deadline_info1, context=ctx) is False

    msg_me_deadline_info2 = {"id": "M_ME_DED_INFO2", "chat_id": "C_GROUP", "sender_id": "ME", "body": "The presentation deadline is tomorrow"}
    assert is_relevant_group_candidate(cand_me_deadline_info, msg_me_deadline_info2, context=ctx) is False

    msg_me_deadline_info3 = {"id": "M_ME_DED_INFO3", "chat_id": "C_GROUP", "sender_id": "ME", "body": "The deck is due Friday."}
    assert is_relevant_group_candidate(cand_me_deadline_info, msg_me_deadline_info3, context=ctx) is False

    # Inquiries about deadlines or personal availability statements from ME are filtered
    msg_me_inquiry1 = {"id": "M_ME_INQ1", "chat_id": "C_GROUP", "sender_id": "ME", "body": "I'll ask Rahul whether the deck is due Friday."}
    assert is_relevant_group_candidate(cand_me_deadline_info, msg_me_inquiry1, context=ctx) is False

    msg_me_inquiry2 = {"id": "M_ME_INQ2", "chat_id": "C_GROUP", "sender_id": "ME", "body": "I need to know if the deck is due Friday."}
    assert is_relevant_group_candidate(cand_me_deadline_info, msg_me_inquiry2, context=ctx) is False

    msg_me_leave = {"id": "M_ME_LEAVE", "chat_id": "C_GROUP", "sender_id": "ME", "body": "I will be on leave by Friday."}
    assert is_relevant_group_candidate(cand_me_deadline_info, msg_me_leave, context=ctx) is False

    # Explicitly self-owned deadline from ME is retained
    cand_me_deadline_owned = {"category": "deadline", "matched_signals": ["by friday"]}
    msg_me_deadline_owned1 = {"id": "M_ME_DED_OWN1", "chat_id": "C_GROUP", "sender_id": "ME", "body": "I need to submit the report by Friday."}
    assert is_relevant_group_candidate(cand_me_deadline_owned, msg_me_deadline_owned1, context=ctx) is True

    msg_me_deadline_owned2 = {"id": "M_ME_DED_OWN2", "chat_id": "C_GROUP", "sender_id": "ME", "body": "I'll finish this by tomorrow."}
    assert is_relevant_group_candidate(cand_me_deadline_owned, msg_me_deadline_owned2, context=ctx) is True

    msg_me_deadline_owned3 = {"id": "M_ME_DED_OWN3", "chat_id": "C_GROUP", "sender_id": "ME", "body": "I am responsible for sending the documents by Monday."}
    assert is_relevant_group_candidate(cand_me_deadline_owned, msg_me_deadline_owned3, context=ctx) is True

    msg_me_deadline_owned4 = {"id": "M_ME_DED_OWN4", "chat_id": "C_GROUP", "sender_id": "ME", "body": "I have to complete the report by Friday."}
    assert is_relevant_group_candidate(cand_me_deadline_owned, msg_me_deadline_owned4, context=ctx) is True

    msg_me_deadline_owned5 = {"id": "M_ME_DED_OWN5", "chat_id": "C_GROUP", "sender_id": "ME", "body": "My task is to submit the report by Friday."}
    assert is_relevant_group_candidate(cand_me_deadline_owned, msg_me_deadline_owned5, context=ctx) is True

    msg_me_deadline_owned6 = {"id": "M_ME_DED_OWN6", "chat_id": "C_GROUP", "sender_id": "ME", "body": "Mujhe report submit karna hai Friday tak."}
    assert is_relevant_group_candidate(cand_me_deadline_owned, msg_me_deadline_owned6, context=ctx) is True

    # Group-wide deadline / broadcast from ME is filtered
    msg_me_deadline_bcast = {"id": "M_ME_DED_BCAST", "chat_id": "C_GROUP", "sender_id": "ME", "body": "Everyone must submit by Monday."}
    assert is_relevant_group_candidate(cand_me_deadline_owned, msg_me_deadline_bcast, context=ctx) is False

    # Explicitly grounded assignment remains retained
    cand_assign_deadline = {"category": "deadline", "matched_signals": ["by friday"]}
    msg_assign_deadline = {"id": "M_ASS_DED", "chat_id": "C_GROUP", "sender_id": "ME", "body": "Rahul please submit the deck by Friday"}
    assert is_relevant_group_candidate(cand_assign_deadline, msg_assign_deadline, context=ctx) is True

    # 10. No hardcoded personal name or generic alias is used as identity proof in prompt or filter
    # "Arjun" is NOT in metadata for ME or any participant
    cand_arjun = {"category": "question", "matched_signals": ["can you"]}
    msg_arjun = {"id": "M_ARJ", "chat_id": "C_GROUP", "sender_id": "P2", "body": "Arjun can you review the draft?"}
    assert is_relevant_group_candidate(cand_arjun, msg_arjun, context=ctx) is False

    for generic_term in ["Bhai", "Bhaiya", "Didi", "Sir", "Madam", "Bro", "Dude", "Boss"]:
        msg_gen = {"id": f"M_{generic_term}", "chat_id": "C_GROUP", "sender_id": "P2", "body": f"{generic_term} please check this"}
        assert is_relevant_group_candidate(cand_arjun, msg_gen, context=ctx) is False, f"Expected {generic_term} not to establish recipient"


def test_exact_message_id_and_deterministic_ordering():
    """Verify that candidate extraction preserves exact message ID and emits deterministic category ordering."""
    extractor = CandidateEvidenceExtractor()

    raw_id = "M_EXACT_123_#@!"
    msg = {
        "id": raw_id,
        "chat_id": "C01",
        "sender_id": "ME",
        "ts": 1787535000,
        "body": "I'll finish the deck by Friday. Can you review it?",
    }
    evidence = extractor.extract_message(msg)
    assert len(evidence) >= 2

    # Verify exact message ID preservation without any alteration
    for ev in evidence:
        assert ev["message_id"] == raw_id

    # Verify deterministic ordering: categories are sorted deterministically
    cats = [e["category"] for e in evidence]
    assert cats == sorted(cats), f"Categories not in deterministic order: {cats}"

    # Verify repeated extraction produces identical ordering
    evidence2 = extractor.extract_message(msg)
    assert evidence == evidence2


def test_group_chat_identity_grounding_edge_cases():
    """Verify edge cases for group-chat recipient resolution and identity grounding."""
    class MockContextEdge:
        chats = {
            "C_GRP": {
                "id": "C_GRP",
                "is_group": True,
                "participants": ["ME", "P01", "P02"],
                "messages": [
                    {"id": "M_P01", "chat_id": "C_GRP", "sender_id": "P01", "sender_name": "Karan Malhotra", "ts": 100, "body": "Hey"},
                ],
            },
        }
        participants = {
            "ME": {"participant_id": "ME", "sender_names": ["Aditi Rao"]},
            "P01": {"participant_id": "P01", "sender_names": ["Karan Malhotra"]},
            "P02": {"participant_id": "P02", "sender_names": ["Sneha"]},
        }

    ctx = MockContextEdge()
    cand = {"category": "question", "matched_signals": ["can you"]}

    # 1. First name extracted from full name in metadata (e.g. 'Karan' from 'Karan Malhotra')
    msg1 = {"id": "M1", "chat_id": "C_GRP", "sender_id": "P02", "body": "Karan please update the status"}
    assert is_relevant_group_candidate(cand, msg1, context=ctx) is True

    # 2. Tag address syntax at the end of sentence: "Can you send the slides, Sneha?"
    msg2 = {"id": "M2", "chat_id": "C_GRP", "sender_id": "P01", "body": "Can you send the slides, Sneha?"}
    assert is_relevant_group_candidate(cand, msg2, context=ctx) is True

    # 3. Directed to ME via first name from full name ('Aditi' from 'Aditi Rao')
    msg3 = {"id": "M3", "chat_id": "C_GRP", "sender_id": "P01", "body": "Aditi please check the budget"}
    assert is_relevant_group_candidate(cand, msg3, context=ctx) is True

    # 4. Context is None: cannot resolve grounded participants -> returns False
    msg_no_ctx = {"id": "M4", "chat_id": "C_GRP", "sender_id": "P01", "body": "Karan please update the status"}
    assert is_relevant_group_candidate(cand, msg_no_ctx, context=None) is False

    # 5. Empty or whitespace-only message body -> returns False
    msg_empty = {"id": "M5", "chat_id": "C_GRP", "sender_id": "P01", "body": "   "}
    assert is_relevant_group_candidate(cand, msg_empty, context=ctx) is False


def test_first_person_deadline_ownership_precision():
    """Verify precision of first-person deadline ownership in group chats.

    Positive cases (explicit, action-grounded task ownership by ME) must be retained.
    Negative cases (inquiries about deadlines, personal availability/scheduling, informational, broadcast) must be filtered.
    """
    class MockContext:
        chats = {
            "C_GRP": {
                "id": "C_GRP",
                "is_group": True,
                "participants": ["ME", "P01", "P02"],
                "messages": [],
            },
        }
        participants = {
            "ME": {"participant_id": "ME", "sender_names": ["Aditi"]},
            "P01": {"participant_id": "P01", "sender_names": ["Rahul"]},
            "P02": {"participant_id": "P02", "sender_names": ["Sneha"]},
        }

    ctx = MockContext()
    cand_deadline = {"category": "deadline", "matched_signals": ["by friday"]}

    # Positive cases — must be retained
    positive_cases = [
        "I need to submit the report by Friday.",
        "I'll finish this by tomorrow.",
        "I am responsible for sending the documents by Monday.",
        "I have to complete the report by Friday.",
        "My task is to submit the report by Friday.",
        "Mujhe report submit karna hai Friday tak.",
    ]
    for idx, text in enumerate(positive_cases, start=1):
        msg = {
            "id": f"M_POS_{idx}",
            "chat_id": "C_GRP",
            "sender_id": "ME",
            "body": text,
        }
        assert is_relevant_group_candidate(cand_deadline, msg, context=ctx) is True, f"Expected positive retention for: {text}"

    # Negative cases — must be filtered
    negative_cases = [
        "I'll ask Rahul whether the deck is due Friday.",
        "I will be on leave by Friday.",
        "I need to know if the deck is due Friday.",
        "The deck is due Friday.",
        "Everyone must submit by Monday.",
    ]
    for idx, text in enumerate(negative_cases, start=1):
        msg = {
            "id": f"M_NEG_{idx}",
            "chat_id": "C_GRP",
            "sender_id": "ME",
            "body": text,
        }
        assert is_relevant_group_candidate(cand_deadline, msg, context=ctx) is False, f"Expected negative filtering for: {text}"


def test_url_query_parameters_do_not_trigger_question_candidate():
    """Verify M00247-style URL query parameters produce no question candidate from URL delimiters."""
    extractor = CandidateEvidenceExtractor()

    # M00247-style marketing URL containing ?, &, =
    msg_m00247 = {
        "id": "M00247",
        "chat_id": "C31",
        "sender_id": "ME",
        "sender_name": "Arjun Khetan",
        "ts": 1787590379,
        "body": "https://www.vertaflow.com/?utm_source=x&utm_medium=paid_social&utm_campaign=Vertaflow_X_Sales_IND&utm_content=Ad2_OutcomeLed&twclid=3g8b2rmjt5plwcne4kvyz91mhd",
    }
    candidates = extractor.extract_message(msg_m00247)
    question_candidates = [c for c in candidates if c["category"] == "question"]
    assert not question_candidates, f"Expected no question candidate for M00247 URL query params, got: {question_candidates}"

    # Another URL with multiple query params and path delimiters
    msg_url_params = {
        "id": "M_URL_1",
        "chat_id": "C1",
        "sender_id": "P01",
        "body": "Check this dashboard: https://metrics.corp.internal/view?project=alpha&tab=overview&filter=active",
    }
    candidates2 = extractor.extract_message(msg_url_params)
    assert not any(c["category"] == "question" for c in candidates2)


def test_genuine_question_outside_url_preserved():
    """Verify a genuine question mark outside a URL still produces a question candidate."""
    extractor = CandidateEvidenceExtractor()

    # Question mark with trailing space outside URL
    msg1 = {
        "id": "M_Q1",
        "chat_id": "C1",
        "sender_id": "P01",
        "body": "Have you seen this link https://example.com/?utm_source=test&id=123 ?",
    }
    cands1 = extractor.extract_message(msg1)
    q_cand1 = next((c for c in cands1 if c["category"] == "question"), None)
    assert q_cand1 is not None, "Expected question candidate from outer question mark"
    assert "?" in q_cand1["matched_signals"]
    assert q_cand1["original_text"] == msg1["body"]

    # Trailing question mark immediately adjacent to URL
    msg2 = {
        "id": "M_Q2",
        "chat_id": "C1",
        "sender_id": "P01",
        "body": "Did you see https://example.com/?q=hello&lang=en?",
    }
    cands2 = extractor.extract_message(msg2)
    q_cand2 = next((c for c in cands2 if c["category"] == "question"), None)
    assert q_cand2 is not None, "Expected question candidate from trailing question mark"
    assert "?" in q_cand2["matched_signals"]
    assert q_cand2["original_text"] == msg2["body"]


def test_multiple_urls_and_adjacent_punctuation():
    """Verify safe delimiter masking across multiple URLs with diverse surrounding punctuation."""
    extractor = CandidateEvidenceExtractor()

    body = (
        "Please review (https://service1.com/?ref=abc&v=1) and "
        "<https://service2.com/?ref=xyz=2>! Can we ship by tomorrow?"
    )
    msg = {"id": "M_MULTI", "chat_id": "C1", "sender_id": "P01", "body": body}
    cands = extractor.extract_message(msg)

    # Question from "Can we ship by tomorrow?"
    q_cand = next((c for c in cands if c["category"] == "question"), None)
    assert q_cand is not None
    assert "?" in q_cand["matched_signals"]

    # Deadline from "by tomorrow"
    d_cand = next((c for c in cands if c["category"] == "deadline"), None)
    assert d_cand is not None

    # Stored original text is completely unchanged
    assert q_cand["original_text"] == body
    assert d_cand["original_text"] == body


def test_structural_syndicated_broadcast_filtered():
    """Verify M00043-style syndicated news broadcasts are filtered at Stage A when all structural conditions are met."""
    extractor = CandidateEvidenceExtractor()

    # M00043-style message with >= 3 paragraphs, syndication anchor, and repeated headline formatting
    broadcast_body = (
        "*BW Top Headlines*\n\n"
        "*Onam 2026: A preview of the year's digital festive marketing playbook?*\n\n"
        "Read More: https://www.brandwire.in/festive-season-news/onam-2026-905899.html\n\n"
        "*From ₹1.2 lakh budget to the big screen: How MSMEs are driving CTV boom*\n\n"
        "Read More: https://www.brandwire.in/digital-news/msmes-ctv-boom-689062.html\n\n"
        "*Socho Aur Jeeto: What MCQ's new line really signals*\n\n"
        "Read More: https://www.brandwire.in/media-tv-news/socho-aur-jeeto-463906.html"
    )
    msg = {
        "id": "M00043_SYNTH",
        "chat_id": "C05",
        "sender_id": "P07",
        "sender_name": "",
        "body": broadcast_body,
    }
    cands = extractor.extract_message(msg)
    assert cands == [], f"Expected syndicated broadcast to be filtered at Stage A, got: {cands}"
    assert is_syndicated_broadcast(broadcast_body) is True


def test_multi_paragraph_message_without_syndication_anchor_retained():
    """Verify a multi-paragraph message without syndication anchors is retained for candidate extraction."""
    extractor = CandidateEvidenceExtractor()

    body = (
        "Hey Nirant, how are things going?\n\n"
        "I'll review the pull request by Friday.\n\n"
        "Here are the two documentation links: https://internal.dev/docs and https://internal.dev/api"
    )
    msg = {"id": "M_PARA", "chat_id": "C1", "sender_id": "P01", "body": body}
    cands = extractor.extract_message(msg)

    # Must NOT be filtered by broadcast detection
    assert is_syndicated_broadcast(body) is False
    categories = {c["category"] for c in cands}
    assert "commitment" in categories
    assert "deadline" in categories


def test_multiple_anchors_and_urls_without_repeated_headlines_retained():
    """Verify multiple 'Read More:' anchors and URLs without repeated formatted headlines are retained."""
    extractor = CandidateEvidenceExtractor()

    # 3 paragraphs, multiple Read More anchors and URLs, but plain text (no bold or bullet headlines)
    body = (
        "Weekly industry update from regional teams.\n\n"
        "First market segment update is available now. Read More: https://example.com/segment1\n\n"
        "Second market segment update is also published. Read More: https://example.com/segment2"
    )
    # Must NOT be classified as a broadcast because repeated formatted headlines are absent
    assert is_syndicated_broadcast(body) is False
    cands = extractor.extract_message({"id": "M_TEST_PLAIN", "chat_id": "C1", "body": body})
    assert isinstance(cands, list)


def test_three_paragraphs_with_only_one_headline_retained():
    """Verify three or more paragraphs with only one formatted headline are retained."""
    extractor = CandidateEvidenceExtractor()

    # 3 paragraphs, syndication anchor, multiple URLs, but only ONE bold headline
    body = (
        "*Single Spotlight Article*\n\n"
        "Discussion on initial findings. Read More: https://example.com/article1\n\n"
        "Concluding remarks and secondary resource at https://example.com/article2"
    )
    # Incomplete headline formatting (only 1 headline) must not trigger broadcast filtering
    assert is_syndicated_broadcast(body) is False
    cands = extractor.extract_message({"id": "M_TEST_SINGLE_HL", "chat_id": "C1", "body": body})
    assert isinstance(cands, list)


def test_two_formatted_headlines_each_with_url_filtered():
    """Verify two formatted headlines, each followed by its own external URL, are filtered when other conditions hold."""
    extractor = CandidateEvidenceExtractor()

    # >= 3 paragraphs, syndication anchor, and 2 distinct formatted headlines each followed by its own URL
    body = (
        "*BW Top Headlines*\n\n"
        "*Market rally continues as indices surge*\n\n"
        "Read More: https://brandwire.in/markets/rally-9812\n\n"
        "*Tech firms expand regional hiring*\n\n"
        "Read More: https://brandwire.in/tech/hiring-4512"
    )
    assert is_syndicated_broadcast(body) is True
    cands = extractor.extract_message({"id": "M_TEST_TWO_HL", "chat_id": "C1", "body": body})
    assert cands == [], f"Expected two formatted headline-URL pairs to be filtered, got: {cands}"


def test_two_formatted_headlines_with_same_url_retained():
    """Verify that two formatted headlines associated with the same URL are not classified as a syndicated broadcast."""
    extractor = CandidateEvidenceExtractor()

    # >= 3 paragraphs, syndication anchor, and 2 formatted headlines, but both associated with the same URL
    body = (
        "*BW Top Headlines*\n\n"
        "*Market rally continues as indices surge*\n\n"
        "Read More: https://brandwire.in/markets/shared-link\n\n"
        "*Tech firms expand regional hiring*\n\n"
        "Read More: https://brandwire.in/markets/shared-link"
    )
    assert is_syndicated_broadcast(body) is False
    cands = extractor.extract_message({"id": "M_TEST_SAME_URL", "chat_id": "C1", "body": body})
    assert isinstance(cands, list)

    # Also test when headlines and same URL are in the same paragraphs
    body_inline = (
        "*BW Top Headlines*\n\n"
        "*Market rally continues as indices surge* Read More: https://brandwire.in/markets/shared-link\n\n"
        "*Tech firms expand regional hiring* Read More: https://brandwire.in/markets/shared-link"
    )
    assert is_syndicated_broadcast(body_inline) is False


def test_single_link_sharing_m00008_retained():
    """Verify ordinary single-link sharing messages (like M00008) are never classified as broadcasts."""
    extractor = CandidateEvidenceExtractor()

    m00008_body = (
        "Concluded our first AI Summit successfully at Gurgaon on Friday!\n\n"
        "https://www.linkedin.com/feed/update/urn:li:activity:7512834095127362541"
    )
    msg = {
        "id": "M00008",
        "chat_id": "C04",
        "sender_id": "P06",
        "sender_name": "Amish Choraria",
        "body": m00008_body,
    }
    assert is_syndicated_broadcast(m00008_body) is False

    # Link sharing that includes an explicit question
    question_link_body = (
        "Has everyone seen this update?\n\n"
        "https://www.linkedin.com/feed/update/urn:li:activity:123456789"
    )
    msg_q = {"id": "M_LINK_Q", "chat_id": "C04", "sender_id": "P06", "body": question_link_body}
    assert is_syndicated_broadcast(question_link_body) is False
    cands_q = extractor.extract_message(msg_q)
    assert any(c["category"] == "question" for c in cands_q)


def test_missing_sender_name_does_not_cause_rejection():
    """Verify that a missing or empty sender_name alone never causes candidate rejection."""
    extractor = CandidateEvidenceExtractor()

    msg = {
        "id": "M_NO_SENDER",
        "chat_id": "C1",
        "sender_id": "P99",
        "sender_name": "",  # Unsaved contact / missing name
        "body": "I'll send the updated invoice tomorrow by 3pm.",
    }
    assert is_syndicated_broadcast(msg["body"]) is False
    cands = extractor.extract_message(msg)
    categories = {c["category"] for c in cands}
    assert "commitment" in categories
    assert "deadline" in categories


def test_original_candidate_body_unmutated_after_normalization():
    """Verify original message dictionary and candidate original_text remain unmutated."""
    extractor = CandidateEvidenceExtractor()

    msg = {
        "id": "M_MUT",
        "chat_id": "C1",
        "sender_id": "ME",
        "body": "Can you review https://example.com/?ref=param&mode=edit?",
    }
    msg_snapshot = copy.deepcopy(msg)
    cands = extractor.extract_message(msg)

    # Original message dictionary was not modified
    assert msg == msg_snapshot

    # Resulting candidate evidence retains original text with delimiters
    assert len(cands) >= 1
    for cand in cands:
        assert cand["original_text"] == msg_snapshot["body"]
        assert "?ref=param&mode=edit?" in cand["original_text"]


def test_confirmed_gold_messages_candidate_recall(dev_context):
    """Verify M00350 and M00270 question candidate recall, and verify non-broadcast status for M00074 and M00391."""
    extractor = CandidateEvidenceExtractor()
    messages_by_id = {m["id"]: m for m in dev_context.dataset["messages"]}

    # 1. M00247 (URL with query string): no question candidate
    assert "M00247" in messages_by_id
    m00247_cands = extractor.extract_message(messages_by_id["M00247"], context=dev_context)
    assert not any(c["category"] == "question" for c in m00247_cands), f"M00247 should not have question candidates, got {m00247_cands}"

    # 2. M00043 (BW Top Headlines): filtered as syndicated broadcast at Stage A
    assert "M00043" in messages_by_id
    m00043_cands = extractor.extract_message(messages_by_id["M00043"], context=dev_context)
    assert m00043_cands == [], f"M00043 should produce 0 candidates, got {m00043_cands}"

    # 3. M00008 (Single link sharing): retained, not filtered as broadcast
    assert "M00008" in messages_by_id
    assert is_syndicated_broadcast(messages_by_id["M00008"]["body"]) is False

    # 4. Confirmed gold open loop messages: verify question candidate recall
    # M00350: "Possible to get a car at 4pm in Bhowanipore?" -> Question candidate verified
    assert "M00350" in messages_by_id
    m00350_cands = extractor.extract_message(messages_by_id["M00350"], context=dev_context)
    assert any(c["category"] == "question" for c in m00350_cands)

    # M00270: "Can we pls have a meeting tomorrow anytime convenient to you between 11-2 pls. Thanks :)" -> Question candidate verified
    assert "M00270" in messages_by_id
    m00270_cands = extractor.extract_message(messages_by_id["M00270"], context=dev_context)
    assert any(c["category"] == "question" for c in m00270_cands)

    # 5. Non-broadcast verification: verify M00074 and M00391 are not rejected by the broadcast detector
    # M00074: "FBC coffee. please ask." -> Not a syndicated broadcast
    assert "M00074" in messages_by_id
    assert is_syndicated_broadcast(messages_by_id["M00074"]["body"]) is False

    # M00391: "Pls pay 2800" -> Not a syndicated broadcast
    assert "M00391" in messages_by_id
    assert is_syndicated_broadcast(messages_by_id["M00391"]["body"]) is False


def test_financial_candidate_detection_positive_and_negative():
    """Verify financial and payment requests are captured while ordinary financial statements are excluded."""
    extractor = CandidateEvidenceExtractor(enabled_categories=["commitment", "deadline", "question", "followup"])

    # Positive cases: direct payment/transfer requests and pledges
    positive_examples = [
        {"id": "F01", "chat_id": "C01", "sender_id": "P01", "body": "Pls pay 2800", "ts": 100, "kind": "text"},
        {"id": "F02", "chat_id": "C01", "sender_id": "P01", "body": "Please transfer ₹3,500 to Nayna", "ts": 101, "kind": "text"},
        {"id": "F03", "chat_id": "C01", "sender_id": "P01", "body": "Kindly settle the invoice by tomorrow", "ts": 102, "kind": "text"},
        {"id": "F04", "chat_id": "C01", "sender_id": "P01", "body": "Pay 1500 to driver", "ts": 103, "kind": "text"},
        {"id": "F05", "chat_id": "C01", "sender_id": "P01", "body": "Please reimburse ₹1200 for groceries", "ts": 104, "kind": "text"},
        {"id": "F06", "chat_id": "C01", "sender_id": "P01", "body": "Paise transfer kar dena", "ts": 105, "kind": "text"},
    ]
    for msg in positive_examples:
        cands = extractor.extract_message(msg)
        assert any(c["category"] == "commitment" for c in cands), f"Failed to extract commitment candidate for: {msg['body']}"

    # Negative cases: purely informational financial discussions or statements and non-financial uses of 'pay'
    negative_examples = [
        {"id": "NF01", "chat_id": "C01", "sender_id": "P01", "body": "The total cost was 2800 yesterday.", "ts": 200, "kind": "text"},
        {"id": "NF02", "chat_id": "C01", "sender_id": "P01", "body": "Revenue grew by 15% this quarter.", "ts": 201, "kind": "text"},
        {"id": "NF03", "chat_id": "C01", "sender_id": "P01", "body": "We spent 2800 on dinner with friends.", "ts": 202, "kind": "text"},
        {"id": "NF04", "chat_id": "C01", "sender_id": "P01", "body": "The stock is trading at 2800 rupees.", "ts": 203, "kind": "text"},
        {"id": "NF05", "chat_id": "C01", "sender_id": "P01", "body": "I paid 2800 last month.", "ts": 204, "kind": "text"},
        {"id": "NF06", "chat_id": "C01", "sender_id": "P01", "body": "Please pay attention to this.", "ts": 205, "kind": "text"},
        {"id": "NF07", "chat_id": "C01", "sender_id": "P01", "body": "Kindly pay attention to the project details.", "ts": 206, "kind": "text"},
        {"id": "NF08", "chat_id": "C01", "sender_id": "P01", "body": "Please pay heed to the warnings.", "ts": 207, "kind": "text"},
        {"id": "NF09", "chat_id": "C01", "sender_id": "P01", "body": "Please pay a visit to the campus.", "ts": 208, "kind": "text"},
        {"id": "NF10", "chat_id": "C01", "sender_id": "P01", "body": "Please pay respects to the family.", "ts": 209, "kind": "text"},
    ]
    for msg in negative_examples:
        cands = extractor.extract_message(msg)
        assert not any(c["category"] == "commitment" for c in cands), f"Erroneously extracted commitment candidate for informational statement: {msg['body']}"






