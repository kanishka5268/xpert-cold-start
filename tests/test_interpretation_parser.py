"""Unit tests for the LLM response parsing and validation module."""

import json
import pytest

from src.interpretation.batcher import EvidenceBatch
from src.interpretation.parser import (
    InterpretationResponseError,
    ResponseParsingError,
    ResponseStructureError,
    _is_valid_timestamp,
    extract_json_payload,
    is_name_grounded_in_text,
    parse_chat_interpretation_response,
    parse_participant_interpretation_response,
)
from src.interpretation.records import (
    InterpretedGoal,
    InterpretedOpenLoop,
    InterpretedOrganisation,
    InterpretedPerson,
)


@pytest.fixture
def sample_batch():
    """A realistic EvidenceBatch for chat-level testing."""
    messages = [
        {"id": "M015", "chat_id": "C04", "sender_id": "P04", "sender_name": "Neha", "ts": 1787534800, "body": "Are we taking the Zest client?"},
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535000, "body": "Pitch deck for Halcyon is due Friday. Neha let me think about Zest"},
        {"id": "M017", "chat_id": "C04", "sender_id": "P04", "sender_name": "Neha", "ts": 1787535200, "body": "Okay, let me know by tomorrow"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535400, "body": "Will do. Sent the invoice to Brightside"},
    ]
    candidates = [
        {
            "message_id": "M016",
            "chat_id": "C04",
            "sender_id": "ME",
            "sender_name": "ME",
            "timestamp": 1787535000,
            "category": "question",
            "matched_signals": ["let me think"],
            "original_text": "Pitch deck for Halcyon is due Friday. Neha let me think about Zest",
        },
        {
            "message_id": "M018",
            "chat_id": "C04",
            "sender_id": "ME",
            "sender_name": "ME",
            "timestamp": 1787535400,
            "category": "commitment",
            "matched_signals": ["will do"],
            "original_text": "Will do. Sent the invoice to Brightside",
        },
    ]
    return EvidenceBatch(
        chat_id="C04",
        messages=messages,
        candidates=candidates,
        candidate_message_ids=["M016", "M018"],
        categories=["commitment", "question"],
    )


@pytest.fixture
def sample_dossiers():
    """Realistic participant dossiers for participant-level testing."""
    return [
        {
            "person_id": "P03",
            "names": ["Raju", "Raju driver"],
            "chats": [{"chat_id": "C03", "name": "Driver", "is_group": False}],
            "messages": [
                {"id": "M010", "chat_id": "C03", "sender_id": "ME", "sender_name": "ME", "ts": 1787530000, "body": "Raju please pick up the kids at 2pm"},
                {"id": "M011", "chat_id": "C03", "sender_id": "P03", "sender_name": "Raju driver", "ts": 1787530100, "body": "Haan bhaiya, pahuch jaunga"},
            ],
        },
        {
            "person_id": "P04",
            "names": ["Neha", "Neha Brightside"],
            "chats": [{"chat_id": "C04", "name": "Brightside Work", "is_group": False}],
            "messages": [
                {"id": "M016", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535000, "body": "Pitch deck for Halcyon is due Friday"},
                {"id": "M017", "chat_id": "C04", "sender_id": "P04", "sender_name": "Neha", "ts": 1787535200, "body": "I visited Das Printers today for sample checks"},
            ],
        },
    ]


# ==============================================================================
# 1. Conservative JSON Extraction Tests
# ==============================================================================

def test_extract_direct_json():
    raw = '{"open_loops": [], "goals": []}'
    data = extract_json_payload(raw)
    assert data == {"open_loops": [], "goals": []}


def test_extract_single_fenced_json():
    raw = "```json\n{\n  \"open_loops\": [],\n  \"goals\": []\n}\n```"
    data = extract_json_payload(raw)
    assert data == {"open_loops": [], "goals": []}


def test_extract_single_fenced_json_no_lang():
    raw = "```\n{\"people\": [], \"organisations\": []}\n```"
    data = extract_json_payload(raw)
    assert data == {"people": [], "organisations": []}


def test_extract_enclosed_json_with_commentary():
    raw = "Here is your analysis:\n{\"open_loops\": [], \"goals\": []}\nHope this helps!"
    data = extract_json_payload(raw)
    assert data == {"open_loops": [], "goals": []}


def test_extract_unclosed_fence_raises_error():
    raw = "```json\n{\"open_loops\": []}"
    with pytest.raises(ResponseParsingError, match="Ambiguous or truncated code fences"):
        extract_json_payload(raw)


def test_extract_multiple_fences_raises_error():
    raw = "```json\n{\"a\": 1}\n```\nand\n```json\n{\"b\": 2}\n```"
    with pytest.raises(ResponseParsingError, match="Ambiguous or truncated code fences"):
        extract_json_payload(raw)


def test_extract_trailing_comma_raises_error():
    # Conservative parser: no regex patching of trailing commas
    raw = '{"open_loops": [], "goals": [],}'
    with pytest.raises(ResponseParsingError, match="Malformed, duplicate, or non-standard JSON"):
        extract_json_payload(raw)


def test_extract_empty_string_raises_error():
    with pytest.raises(ResponseParsingError, match="Empty or non-string"):
        extract_json_payload("")

    with pytest.raises(ResponseParsingError, match="Empty or non-string"):
        extract_json_payload("   \n\t  ")


def test_extract_non_dict_root_raises_structure_error():
    raw = '[{"item": 1}]'
    with pytest.raises(ResponseStructureError, match="Expected top-level JSON object"):
        extract_json_payload(raw)


def test_extract_primitive_root_raises_structure_error():
    raw = '"just a string"'
    with pytest.raises(ResponseStructureError, match="Expected top-level JSON object"):
        extract_json_payload(raw)


# ==============================================================================
# 2. Top-Level Collection Keys Validation Tests
# ==============================================================================

def test_chat_missing_open_loops_raises_structure_error(sample_batch):
    raw = json.dumps({"goals": []})
    with pytest.raises(ResponseStructureError, match="missing required 'open_loops' list"):
        parse_chat_interpretation_response(raw, sample_batch)


def test_chat_missing_goals_raises_structure_error(sample_batch):
    raw = json.dumps({"open_loops": []})
    with pytest.raises(ResponseStructureError, match="missing required 'goals' list"):
        parse_chat_interpretation_response(raw, sample_batch)


def test_chat_wrong_container_type_raises_structure_error(sample_batch):
    raw = json.dumps({"open_loops": "none", "goals": []})
    with pytest.raises(ResponseStructureError, match="missing required 'open_loops' list"):
        parse_chat_interpretation_response(raw, sample_batch)


def test_chat_explicitly_empty_lists_valid(sample_batch):
    raw = json.dumps({"open_loops": [], "goals": []})
    loops, goals = parse_chat_interpretation_response(raw, sample_batch)
    assert loops == []
    assert goals == []


def test_participant_missing_people_raises_structure_error(sample_dossiers):
    raw = json.dumps({"organisations": []})
    with pytest.raises(ResponseStructureError, match="missing required 'people' list"):
        parse_participant_interpretation_response(raw, sample_dossiers)


def test_participant_missing_organisations_raises_structure_error(sample_dossiers):
    raw = json.dumps({"people": []})
    with pytest.raises(ResponseStructureError, match="missing required 'organisations' list"):
        parse_participant_interpretation_response(raw, sample_dossiers)


def test_participant_wrong_container_type_raises_structure_error(sample_dossiers):
    raw = json.dumps({"people": {}, "organisations": []})
    with pytest.raises(ResponseStructureError, match="missing required 'people' list"):
        parse_participant_interpretation_response(raw, sample_dossiers)


def test_participant_explicitly_empty_lists_valid(sample_dossiers):
    raw = json.dumps({"people": [], "organisations": []})
    people, orgs = parse_participant_interpretation_response(raw, sample_dossiers)
    assert people == []
    assert orgs == []


# ==============================================================================
# 3. Open Loop Grounding & Referential Integrity Tests
# ==============================================================================

def test_open_loop_source_msg_must_be_in_candidate_ids(sample_batch):
    # M015 is in batch.messages, but NOT in candidate_message_ids (only M016, M018 are candidates)
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M015",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "ME",
                "target_person": None,
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 0  # Rejected because M015 is not a candidate


def test_open_loop_chat_id_mismatch_rejected(sample_batch):
    # Candidate M016 is valid, but chat_id does not match batch.chat_id (C04)
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C99",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "ME",
                "target_person": None,
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 0  # Rejected (not silently overwritten!)


def test_open_loop_owner_unsupplied_in_prompt_becomes_none(sample_batch):
    # P99 was never in batch.messages or candidates
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "P99",
                "target_person": None,
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].owner is None
    assert loops[0].validation_flags == ["unsupplied_owner:P99"]


def test_open_loop_owner_display_name_becomes_none(sample_batch):
    # Neha is display name, not participant ID (P04)
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "Neha",
                "target_person": None,
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].owner is None
    assert loops[0].validation_flags == ["unsupplied_owner:Neha"]


def test_open_loop_target_person_unsupplied_becomes_none(sample_batch):
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "ME",
                "target_person": "P99",
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].target_person is None
    assert loops[0].validation_flags == ["unsupplied_target_person:P99"]


def test_open_loop_closed_by_unsupplied_in_batch_messages_becomes_none(sample_batch):
    # M999 was never supplied in batch.messages
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "ME",
                "target_person": "P04",
                "status": "done",
                "closed_by": "M999",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].closed_by is None
    assert loops[0].validation_flags == ["unsupplied_closed_by:M999"]


def test_open_loop_closed_by_inverted_timestamp(sample_batch):
    # M015 ts=1787534800, M016 ts=1787535000. M015 is earlier in time than M016!
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "ME",
                "target_person": "P04",
                "status": "done",
                "closed_by": "M015",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].closed_by is None
    assert loops[0].validation_flags == ["invalid_closed_by_timestamp:M015"]


def test_open_loop_closed_by_valid(sample_batch):
    # M017 ts=1787535200 > M016 ts=1787535000 and M017 is in batch.messages
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "ME",
                "target_person": "P04",
                "status": "done",
                "closed_by": "M017",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].closed_by == "M017"
    assert loops[0].validation_flags is None


# ==============================================================================
# 4. Open Loop Semantic & Field Validation Tests
# ==============================================================================

def test_open_loop_invalid_type_enum_rejected(sample_batch):
    # No fallback substitution to "commitment"
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "task",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 0


def test_open_loop_invalid_status_enum_rejected(sample_batch):
    # No fallback substitution to "open"
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send invoice",
                "owner": "ME",
                "target_person": "P04",
                "status": "pending",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 0


def test_open_loop_is_genuine_false_discarded(sample_batch):
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": False,
                "title": "Rhetorical question",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 0  # Discarded cleanly


def test_open_loop_is_genuine_non_boolean_rejected(sample_batch):
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": "true",
                "title": "Question with string boolean",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 0  # Rejected due to non-boolean


def test_open_loop_confidence_out_of_bounds_no_clamp(sample_batch):
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
                "confidence": 1.5,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].confidence is None  # Never clamped to 1.0!
    assert loops[0].validation_flags == ["invalid_confidence:1.5"]


def test_open_loop_due_date_invalid_calendar_date(sample_batch):
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "question",
                "is_genuine": True,
                "title": "Decide on client",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": "2026-02-31",
                "suggested_action": "remind",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].due_date is None
    assert loops[0].validation_flags == ["invalid_date_format:2026-02-31"]
    assert loops[0].suggested_action == "remind"


# ==============================================================================
# 5. Goal Grounding & Validation Tests
# ==============================================================================

def test_goal_invalid_area_enum_rejected(sample_batch):
    # No fallback substitution to "self"
    raw = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Get fit",
                "area": "fitness",
                "type": "habit",
                "horizon": "1yr",
                "evidence_msgs": ["M016"],
                "is_genuine": True,
            }
        ],
    })
    _, goals = parse_chat_interpretation_response(raw, sample_batch)
    assert len(goals) == 0


def test_goal_invalid_type_enum_rejected(sample_batch):
    # No fallback substitution to "milestone"
    raw = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Get fit",
                "area": "health",
                "type": "aspiration",
                "horizon": "1yr",
                "evidence_msgs": ["M016"],
                "is_genuine": True,
            }
        ],
    })
    _, goals = parse_chat_interpretation_response(raw, sample_batch)
    assert len(goals) == 0


def test_goal_invalid_horizon_enum_rejected(sample_batch):
    # No fallback substitution to "1yr"
    raw = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Get fit",
                "area": "health",
                "type": "habit",
                "horizon": "6mo",
                "evidence_msgs": ["M016"],
                "is_genuine": True,
            }
        ],
    })
    _, goals = parse_chat_interpretation_response(raw, sample_batch)
    assert len(goals) == 0


def test_goal_evidence_msgs_restricted_to_batch(sample_batch):
    # M016 is in batch.messages; M999 is NOT in batch.messages
    raw = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Run half marathon",
                "area": "health",
                "type": "milestone",
                "horizon": "3mo",
                "evidence_msgs": ["M016", "M999"],
                "is_genuine": True,
            }
        ],
    })
    _, goals = parse_chat_interpretation_response(raw, sample_batch)
    assert len(goals) == 1
    assert goals[0].evidence_msgs == ["M016"]  # M999 filtered out


def test_goal_zero_valid_evidence_rejected(sample_batch):
    # All evidence messages unsupplied in the batch
    raw = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Run half marathon",
                "area": "health",
                "type": "milestone",
                "horizon": "3mo",
                "evidence_msgs": ["M999"],
                "is_genuine": True,
            }
        ],
    })
    _, goals = parse_chat_interpretation_response(raw, sample_batch)
    assert len(goals) == 0  # Ungrounded goal rejected


def test_goal_is_genuine_false_discarded(sample_batch):
    raw = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Eat pizza for dinner",
                "area": "health",
                "type": "experience",
                "horizon": "3mo",
                "evidence_msgs": ["M016"],
                "is_genuine": False,
            }
        ],
    })
    _, goals = parse_chat_interpretation_response(raw, sample_batch)
    assert len(goals) == 0


def test_goal_no_validation_flags_supported(sample_batch):
    raw = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Run marathon",
                "area": "health",
                "type": "milestone",
                "horizon": "1yr",
                "evidence_msgs": ["M016"],
                "is_genuine": True,
                "confidence": 0.95,
            }
        ],
    })
    _, goals = parse_chat_interpretation_response(raw, sample_batch)
    assert len(goals) == 1
    assert not hasattr(goals[0], "validation_flags")


# ==============================================================================
# 6. Participant Grounding & Validation Tests
# ==============================================================================

def test_person_id_not_in_dossiers_rejected(sample_dossiers):
    # P99 was not in sample_dossiers (only P03 and P04 are)
    raw = json.dumps({
        "people": [
            {
                "person_id": "P99",
                "circle": "friend",
                "looks_after": [],
                "associated_org_names": [],
                "org_link_type": None,
                "evidence_msgs": [],
            }
        ],
        "organisations": [],
    })
    people, _ = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(people) == 0


def test_person_id_me_rejected(sample_dossiers):
    raw = json.dumps({
        "people": [
            {
                "person_id": "ME",
                "circle": "friend",
                "looks_after": [],
                "associated_org_names": [],
                "org_link_type": None,
                "evidence_msgs": [],
            }
        ],
        "organisations": [],
    })
    people, _ = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(people) == 0


def test_person_looks_after_strictly_bounded_to_dossiers(sample_dossiers):
    # P04 is in sample_dossiers; P99 is not in sample_dossiers
    raw = json.dumps({
        "people": [
            {
                "person_id": "P03",
                "circle": "staff",
                "looks_after": ["P04", "P99", "kids"],
                "associated_org_names": [],
                "org_link_type": None,
                "evidence_msgs": ["M010"],
            }
        ],
        "organisations": [],
    })
    people, _ = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(people) == 1
    assert people[0].looks_after == ["P04"]  # P99 and display name 'kids' filtered out


def test_person_associated_org_names_grounded_phrase_match(sample_dossiers):
    # In sample_dossiers: P04 has chat name 'Brightside Work', and M017 mentions 'Das Printers'
    raw = json.dumps({
        "people": [
            {
                "person_id": "P04",
                "circle": "colleague",
                "looks_after": [],
                "associated_org_names": ["Brightside Work", "Das Printers", "Fabricated Org"],
                "org_link_type": "runs",
                "evidence_msgs": ["M016"],
            }
        ],
        "organisations": [],
    })
    people, _ = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(people) == 1
    assert people[0].associated_org_names == ["Brightside Work", "Das Printers"]
    assert "Fabricated Org" not in people[0].associated_org_names


def test_person_associated_org_names_accidental_substring_filtered():
    # If text has 'pineapples', org candidate 'Apple' must NOT match!
    dossier = [
        {
            "person_id": "P01",
            "names": ["Bob"],
            "chats": [],
            "messages": [{"id": "M01", "body": "I love eating pineapples"}],
        }
    ]
    raw = json.dumps({
        "people": [
            {
                "person_id": "P01",
                "circle": "friend",
                "looks_after": [],
                "associated_org_names": ["Apple"],
                "org_link_type": None,
                "evidence_msgs": ["M01"],
            }
        ],
        "organisations": [],
    })
    people, _ = parse_participant_interpretation_response(raw, dossier)
    assert len(people) == 1
    assert people[0].associated_org_names == []  # Apple rejected as accidental substring!


def test_person_evidence_restricted_to_dossier_messages(sample_dossiers):
    # M010 is in P03's messages; M999 is unsupplied
    raw = json.dumps({
        "people": [
            {
                "person_id": "P03",
                "circle": "staff",
                "looks_after": [],
                "associated_org_names": [],
                "org_link_type": None,
                "evidence_msgs": ["M010", "M999"],
            }
        ],
        "organisations": [],
    })
    people, _ = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(people) == 1
    assert people[0].evidence_msgs == ["M010"]


def test_person_circle_enum_and_case_insensitive(sample_dossiers):
    raw = json.dumps({
        "people": [
            {
                "person_id": "P03",
                "circle": "STAFF",
                "looks_after": [],
                "associated_org_names": [],
                "org_link_type": "supports",
                "evidence_msgs": ["M010"],
            },
            {
                "person_id": "P04",
                "circle": "acquaintance",  # Invalid circle
                "looks_after": [],
                "associated_org_names": [],
                "org_link_type": "leads",  # Invalid link type
                "evidence_msgs": ["M016"],
            },
        ],
        "organisations": [],
    })
    people, _ = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(people) == 2
    assert people[0].circle == "staff"
    assert people[0].org_link_type == "supports"
    assert people[1].circle is None
    assert people[1].org_link_type is None


def test_person_no_validation_flags_supported(sample_dossiers):
    raw = json.dumps({
        "people": [
            {
                "person_id": "P03",
                "circle": "staff",
                "looks_after": [],
                "associated_org_names": [],
                "org_link_type": None,
                "evidence_msgs": ["M010"],
            }
        ],
        "organisations": [],
    })
    people, _ = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(people) == 1
    assert not hasattr(people[0], "validation_flags")


# ==============================================================================
# 7. Organisation Grounding & Validation Tests
# ==============================================================================

def test_org_name_ungrounded_rejected(sample_dossiers):
    # 'Acme Corp' appears nowhere in sample_dossiers
    raw = json.dumps({
        "people": [],
        "organisations": [
            {
                "name": "Acme Corp",
                "org_type": "company",
                "side": "work",
                "role": "owner",
                "evidence_msgs": ["M016"],
                "associated_chats": ["C04"],
            }
        ],
    })
    _, orgs = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(orgs) == 0  # Ungrounded org rejected


def test_org_name_grounded_accepted(sample_dossiers):
    # 'Das Printers' appears in message M017 in sample_dossiers
    raw = json.dumps({
        "people": [],
        "organisations": [
            {
                "name": "Das Printers",
                "org_type": "company",
                "side": "work",
                "role": "partner",
                "evidence_msgs": ["M017", "M999"],
                "associated_chats": ["C04", "C99"],
            }
        ],
    })
    _, orgs = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(orgs) == 1
    assert orgs[0].name == "Das Printers"
    assert orgs[0].org_type == "company"
    assert orgs[0].side == "work"
    assert orgs[0].role == "partner"
    assert orgs[0].evidence_msgs == ["M017"]  # M999 unsupplied, filtered out
    assert orgs[0].associated_chats == ["C04"]  # C99 unsupplied, filtered out


def test_org_name_accidental_substring_rejected():
    dossier = [
        {
            "person_id": "P01",
            "names": ["Bob"],
            "chats": [],
            "messages": [{"id": "M01", "body": "I visited the bookstore"}],
        }
    ]
    raw = json.dumps({
        "people": [],
        "organisations": [
            {
                "name": "book",  # 'book' is inside 'bookstore' but not a standalone phrase
                "org_type": "company",
                "side": "work",
                "role": "client",
                "evidence_msgs": ["M01"],
                "associated_chats": [],
            }
        ],
    })
    _, orgs = parse_participant_interpretation_response(raw, dossier)
    assert len(orgs) == 0  # Rejected because 'book' is an accidental substring match inside 'bookstore'


def test_org_associated_chats_strictly_bounded_to_dossiers(sample_dossiers):
    # C04 and C03 are in sample_dossiers; C99 is not
    raw = json.dumps({
        "people": [],
        "organisations": [
            {
                "name": "Das Printers",
                "org_type": "company",
                "side": "work",
                "role": "owner",
                "evidence_msgs": ["M017"],
                "associated_chats": ["C04", "C99"],
            }
        ],
    })
    _, orgs = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(orgs) == 1
    assert orgs[0].associated_chats == ["C04"]


def test_org_optional_enums(sample_dossiers):
    raw = json.dumps({
        "people": [],
        "organisations": [
            {
                "name": "Das Printers",
                "org_type": "invalid_type",
                "side": "invalid_side",
                "role": "invalid_role",
                "evidence_msgs": ["M017"],
                "associated_chats": ["C04"],
            }
        ],
    })
    _, orgs = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(orgs) == 1
    assert orgs[0].org_type is None
    assert orgs[0].side is None
    assert orgs[0].role is None


def test_org_no_validation_flags_supported(sample_dossiers):
    raw = json.dumps({
        "people": [],
        "organisations": [
            {
                "name": "Das Printers",
                "org_type": "company",
                "side": "work",
                "role": "owner",
                "evidence_msgs": ["M017"],
                "associated_chats": ["C04"],
            }
        ],
    })
    _, orgs = parse_participant_interpretation_response(raw, sample_dossiers)
    assert len(orgs) == 1
    assert not hasattr(orgs[0], "validation_flags")


# ==============================================================================
# 8. Fault Isolation & Input Validation Tests
# ==============================================================================

def test_single_corrupted_item_does_not_fail_batch(sample_batch):
    raw = json.dumps({
        "open_loops": [
            # Corrupted item: invalid type enum
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "corrupted_type",
                "is_genuine": True,
                "title": "Bad loop",
                "owner": "ME",
                "target_person": None,
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            },
            # Valid item
            {
                "source_msg": "M018",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send invoice to Brightside",
                "owner": "ME",
                "target_person": None,
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": "remind",
            },
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].source_msg == "M018"
    assert loops[0].title == "Send invoice to Brightside"


def test_chat_batch_input_validation():
    with pytest.raises(ValueError, match="Expected EvidenceBatch"):
        parse_chat_interpretation_response("{}", None)

    with pytest.raises(ValueError, match="Expected EvidenceBatch"):
        parse_chat_interpretation_response("{}", {"chat_id": "C1"})


def test_participant_dossier_input_validation():
    with pytest.raises(ValueError, match="Dossiers cannot be None"):
        parse_participant_interpretation_response("{}", None)

    with pytest.raises(ValueError, match="Dossiers list cannot be empty"):
        parse_participant_interpretation_response("{}", [])


# ==============================================================================
# 9. Strict JSON Parsing & Timestamp Regression Tests
# ==============================================================================

def test_extract_duplicate_keys_top_level_raises_parsing_error():
    raw = '{"open_loops": [], "goals": [], "open_loops": []}'
    with pytest.raises(ResponseParsingError, match="Duplicate key"):
        extract_json_payload(raw)


def test_extract_duplicate_keys_nested_level_raises_parsing_error():
    raw = '{"open_loops": [{"title": "A", "title": "B"}], "goals": []}'
    with pytest.raises(ResponseParsingError, match="Duplicate key"):
        extract_json_payload(raw)


def test_extract_nan_constant_raises_parsing_error():
    raw = '{"open_loops": [], "goals": [], "confidence": NaN}'
    with pytest.raises(ResponseParsingError, match="Non-standard JSON constant"):
        extract_json_payload(raw)


def test_extract_infinity_constant_raises_parsing_error():
    raw = '{"open_loops": [], "goals": [], "confidence": Infinity}'
    with pytest.raises(ResponseParsingError, match="Non-standard JSON constant"):
        extract_json_payload(raw)


def test_extract_negative_infinity_constant_raises_parsing_error():
    raw = '{"open_loops": [], "goals": [], "confidence": -Infinity}'
    with pytest.raises(ResponseParsingError, match="Non-standard JSON constant"):
        extract_json_payload(raw)


def test_open_loop_source_msg_must_exist_in_batch_messages(sample_batch):
    # Candidate message ID M999 is in candidate_message_ids but NOT in batch.messages
    batch_missing_msg = EvidenceBatch(
        chat_id="C04",
        messages=sample_batch.messages,
        candidates=sample_batch.candidates,
        candidate_message_ids=["M999"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M999",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Action",
                "owner": "ME",
                "target_person": None,
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch_missing_msg)
    assert len(loops) == 0


def test_open_loop_closed_by_missing_source_timestamp_discards_closed_by():
    # Source message has ts=None
    msgs = [
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "ts": None, "body": "Need report"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "ts": 1787535400, "body": "Sent it"},
    ]
    batch = EvidenceBatch(
        chat_id="C04",
        messages=msgs,
        candidates=[{"message_id": "M016", "chat_id": "C04", "sender_id": "ME", "category": "commitment"}],
        candidate_message_ids=["M016"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send report",
                "owner": "ME",
                "target_person": None,
                "status": "done",
                "closed_by": "M018",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch)
    assert len(loops) == 1
    assert loops[0].closed_by is None
    assert loops[0].validation_flags == ["invalid_timestamp_comparison:M018"]





def test_open_loop_closed_by_missing_closing_timestamp_discards_closed_by():
    # Closing message has ts=None
    msgs = [
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "ts": 1787535000, "body": "Need report"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "ts": None, "body": "Sent it"},
    ]
    batch = EvidenceBatch(
        chat_id="C04",
        messages=msgs,
        candidates=[{"message_id": "M016", "chat_id": "C04", "sender_id": "ME", "category": "commitment"}],
        candidate_message_ids=["M016"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send report",
                "owner": "ME",
                "target_person": None,
                "status": "done",
                "closed_by": "M018",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch)
    assert len(loops) == 1
    assert loops[0].closed_by is None
    assert loops[0].validation_flags == ["invalid_timestamp_comparison:M018"]





def test_open_loop_closed_by_invalid_string_timestamp_discards_closed_by():
    # Timestamps with non-numeric strings
    msgs = [
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "ts": "invalid_ts", "body": "Need report"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "ts": "invalid_ts_2", "body": "Sent it"},
    ]
    batch = EvidenceBatch(
        chat_id="C04",
        messages=msgs,
        candidates=[{"message_id": "M016", "chat_id": "C04", "sender_id": "ME", "category": "commitment"}],
        candidate_message_ids=["M016"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send report",
                "owner": "ME",
                "target_person": None,
                "status": "done",
                "closed_by": "M018",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch)
    assert len(loops) == 1
    assert loops[0].closed_by is None
    assert loops[0].validation_flags == ["invalid_timestamp_comparison:M018"]





def test_is_valid_timestamp_numeric_and_finite():
    assert _is_valid_timestamp(1787535000) is True
    assert _is_valid_timestamp(0) is True
    assert _is_valid_timestamp(-100) is True
    assert _is_valid_timestamp(1787535000.5) is True
    assert _is_valid_timestamp(0.0) is True
    assert _is_valid_timestamp(True) is False
    assert _is_valid_timestamp(False) is False
    assert _is_valid_timestamp(None) is False
    assert _is_valid_timestamp("1787535000") is False
    assert _is_valid_timestamp(float("nan")) is False
    assert _is_valid_timestamp(float("inf")) is False
    assert _is_valid_timestamp(float("-inf")) is False


def test_open_loop_closed_by_source_timestamp_nan_discards_closed_by():
    msgs = [
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "ts": float("nan"), "body": "Need report"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "ts": 1787535400, "body": "Sent it"},
    ]
    batch = EvidenceBatch(
        chat_id="C04",
        messages=msgs,
        candidates=[{"message_id": "M016", "chat_id": "C04", "sender_id": "ME", "category": "commitment"}],
        candidate_message_ids=["M016"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send report",
                "owner": "ME",
                "target_person": None,
                "status": "done",
                "closed_by": "M018",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch)
    assert len(loops) == 1
    assert loops[0].closed_by is None
    assert loops[0].validation_flags == ["invalid_timestamp_comparison:M018"]





def test_open_loop_closed_by_closing_timestamp_nan_discards_closed_by():
    msgs = [
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "ts": 1787535000, "body": "Need report"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "ts": float("nan"), "body": "Sent it"},
    ]
    batch = EvidenceBatch(
        chat_id="C04",
        messages=msgs,
        candidates=[{"message_id": "M016", "chat_id": "C04", "sender_id": "ME", "category": "commitment"}],
        candidate_message_ids=["M016"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send report",
                "owner": "ME",
                "target_person": None,
                "status": "done",
                "closed_by": "M018",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch)
    assert len(loops) == 1
    assert loops[0].closed_by is None
    assert loops[0].validation_flags == ["invalid_timestamp_comparison:M018"]





def test_open_loop_closed_by_source_timestamp_pos_inf_discards_closed_by():
    msgs = [
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "ts": float("inf"), "body": "Need report"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "ts": 1787535400, "body": "Sent it"},
    ]
    batch = EvidenceBatch(
        chat_id="C04",
        messages=msgs,
        candidates=[{"message_id": "M016", "chat_id": "C04", "sender_id": "ME", "category": "commitment"}],
        candidate_message_ids=["M016"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send report",
                "owner": "ME",
                "target_person": None,
                "status": "done",
                "closed_by": "M018",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch)
    assert len(loops) == 1
    assert loops[0].closed_by is None
    assert loops[0].validation_flags == ["invalid_timestamp_comparison:M018"]





def test_open_loop_closed_by_closing_timestamp_neg_inf_discards_closed_by():
    msgs = [
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "ts": 1787535000, "body": "Need report"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "ts": float("-inf"), "body": "Sent it"},
    ]
    batch = EvidenceBatch(
        chat_id="C04",
        messages=msgs,
        candidates=[{"message_id": "M016", "chat_id": "C04", "sender_id": "ME", "category": "commitment"}],
        candidate_message_ids=["M016"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send report",
                "owner": "ME",
                "target_person": None,
                "status": "done",
                "closed_by": "M018",
                "due_date": None,
                "suggested_action": None,
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch)
    assert len(loops) == 1
    assert loops[0].closed_by is None
    assert loops[0].validation_flags == ["invalid_timestamp_comparison:M018"]


def test_open_loop_valid_due_anchor_msg(sample_batch):
    """Verify that a valid due_date with a valid due_anchor_msg is preserved."""
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Halcyon pitch deck",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": "2026-05-15",
                "due_anchor_msg": "M017",
                "suggested_action": "remind",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].due_date == "2026-05-15"
    assert loops[0].due_anchor_msg == "M017"
    assert loops[0].validation_flags is None


def test_open_loop_due_date_with_source_as_anchor(sample_batch):
    """Verify that using source_msg itself as due_anchor_msg is valid and preserved."""
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Halcyon pitch deck",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": "2026-05-15",
                "due_anchor_msg": "M016",
                "suggested_action": "remind",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].due_date == "2026-05-15"
    assert loops[0].due_anchor_msg == "M016"
    assert loops[0].validation_flags is None


def test_open_loop_missing_anchor_with_non_null_due_date(sample_batch):
    """Verify that missing/null due_anchor_msg with non-null due_date invalidates due_date."""
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Halcyon pitch deck",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": "2026-05-15",
                "due_anchor_msg": None,
                "suggested_action": "remind",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].due_date is None
    assert loops[0].due_anchor_msg is None
    assert "ungrounded_due_date:2026-05-15" in (loops[0].validation_flags or [])


def test_open_loop_unsupplied_anchor_not_in_batch(sample_batch):
    """Verify that due_anchor_msg not present in batch invalidates due_date."""
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Halcyon pitch deck",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": "2026-05-15",
                "due_anchor_msg": "M999",
                "suggested_action": "remind",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].due_date is None
    assert loops[0].due_anchor_msg is None
    flags = loops[0].validation_flags or []
    assert "unsupplied_due_anchor_msg:M999" in flags
    assert "ungrounded_due_date:2026-05-15" in flags


def test_open_loop_cross_chat_anchor():
    """Verify that due_anchor_msg from a different chat_id invalidates due_date."""
    msgs = [
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "ts": 1787535000, "body": "Pitch deck due Friday"},
        {"id": "M020", "chat_id": "C99", "sender_id": "P05", "ts": 1787535200, "body": "Deadline note"},
    ]
    batch = EvidenceBatch(
        chat_id="C04",
        messages=msgs,
        candidates=[{"message_id": "M016", "chat_id": "C04", "sender_id": "ME", "category": "commitment"}],
        candidate_message_ids=["M016"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Pitch deck",
                "owner": "ME",
                "status": "open",
                "due_date": "2026-05-15",
                "due_anchor_msg": "M020",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch)
    assert len(loops) == 1
    assert loops[0].due_date is None
    assert loops[0].due_anchor_msg is None
    flags = loops[0].validation_flags or []
    assert "cross_chat_due_anchor_msg:M020" in flags
    assert "ungrounded_due_date:2026-05-15" in flags


def test_open_loop_anchor_timestamp_before_source(sample_batch):
    """Verify that due_anchor_msg with timestamp prior to source_msg invalidates due_date."""
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Halcyon pitch deck",
                "owner": "ME",
                "status": "open",
                "due_date": "2026-05-15",
                "due_anchor_msg": "M015",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].due_date is None
    assert loops[0].due_anchor_msg is None
    flags = loops[0].validation_flags or []
    assert "invalid_due_anchor_timestamp:M015" in flags
    assert "ungrounded_due_date:2026-05-15" in flags


def test_open_loop_anchor_with_invalid_timestamp():
    """Verify that due_anchor_msg with non-numeric timestamp invalidates due_date."""
    msgs = [
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "ts": 1787535000, "body": "Deck due Friday"},
        {"id": "M017", "chat_id": "C04", "sender_id": "P04", "ts": "invalid_ts", "body": "Tomorrow"},
    ]
    batch = EvidenceBatch(
        chat_id="C04",
        messages=msgs,
        candidates=[{"message_id": "M016", "chat_id": "C04", "sender_id": "ME", "category": "commitment"}],
        candidate_message_ids=["M016"],
        categories=["commitment"],
    )
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Pitch deck",
                "owner": "ME",
                "status": "open",
                "due_date": "2026-05-15",
                "due_anchor_msg": "M017",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, batch)
    assert len(loops) == 1
    assert loops[0].due_date is None
    assert loops[0].due_anchor_msg is None
    flags = loops[0].validation_flags or []
    assert "invalid_timestamp_comparison:M017" in flags
    assert "ungrounded_due_date:2026-05-15" in flags


def test_open_loop_anchor_with_null_due_date(sample_batch):
    """Verify that when due_date is None, due_anchor_msg is strictly forced to None."""
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Halcyon pitch deck",
                "owner": "ME",
                "status": "open",
                "due_date": None,
                "due_anchor_msg": "M017",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].due_date is None
    assert loops[0].due_anchor_msg is None


def test_open_loop_anchor_padded_whitespace_rejected_no_strip(sample_batch):
    """Verify that due_anchor_msg with whitespace is rejected without .strip() normalization."""
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Halcyon pitch deck",
                "owner": "ME",
                "status": "open",
                "due_date": "2026-05-15",
                "due_anchor_msg": " M017 ",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].due_date is None
    assert loops[0].due_anchor_msg is None
    flags = loops[0].validation_flags or []
    assert "invalid_due_anchor_msg: M017 " in flags
    assert "ungrounded_due_date:2026-05-15" in flags


def test_open_loop_existing_valid_response_without_due_date(sample_batch):
    """Verify that an existing valid open loop without due date preserves due_date=None and due_anchor_msg=None."""
    raw = json.dumps({
        "open_loops": [
            {
                "source_msg": "M016",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Halcyon pitch deck",
                "owner": "ME",
                "target_person": "P04",
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": "assist",
            }
        ],
        "goals": [],
    })
    loops, _ = parse_chat_interpretation_response(raw, sample_batch)
    assert len(loops) == 1
    assert loops[0].due_date is None
    assert loops[0].due_anchor_msg is None
    assert loops[0].validation_flags is None





