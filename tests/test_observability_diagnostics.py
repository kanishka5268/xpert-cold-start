"""Focused regression tests for WhatsApp interpretation pipeline observability and diagnostics."""

import json
from typing import Any, Dict, List
import pytest

from src.ingestion.loader import load_dataset
from src.interpretation.batcher import (
    CandidateBatchTrace,
    EvidenceBatch,
    EvidenceBatcher,
    trace_candidates_in_batches,
)
from src.interpretation.interpreter import (
    BatchFailureDiagnostic,
    InterpretationDiagnostics,
    InterpretationResult,
    LLMEvidenceInterpreter,
)
from src.interpretation.parser import (
    ChatParserDiagnostics,
    parse_chat_interpretation_response,
)
from src.interpretation.prompts import PARTICIPANT_SYSTEM_PROMPT
from src.interpretation.records import InterpretedGoal, InterpretedOpenLoop
from src.preprocessing.context_builder import build_conversation_context


@pytest.fixture
def sample_context(tmp_path):
    """Create a realistic ConversationContext for diagnostics testing."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    chats = [
        {"chat_id": "C01", "name": "Direct Neha", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C04", "name": "Work Group", "is_group": True, "participants": ["ME", "P01", "P04"]},
    ]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "ME", "sender_name": "ME", "ts": 1787530000, "kind": "text", "body": "Neha can you send the report?"},
        {"id": "M015", "chat_id": "C04", "sender_id": "P04", "sender_name": "Neha", "ts": 1787534800, "kind": "text", "body": "Are we taking the Zest client?"},
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535000, "kind": "text", "body": "Pitch deck for Halcyon is due Friday. Neha let me think about Zest"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535400, "kind": "text", "body": "Will do. Sent the invoice to Brightside"},
        {"id": "M019", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535500, "kind": "text", "body": "10 retainer clients by March is our target"},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")

    return build_conversation_context(load_dataset(tmp_path))


@pytest.fixture
def sample_chat_batch():
    messages = [
        {"id": "M015", "chat_id": "C04", "sender_id": "P04", "sender_name": "Neha", "ts": 1787534800, "body": "Are we taking the Zest client?"},
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535000, "body": "Pitch deck for Halcyon is due Friday. Neha let me think about Zest"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535400, "body": "Will do. Sent the invoice to Brightside"},
        {"id": "M019", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535500, "body": "10 retainer clients by March is our target"},
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
        {
            "message_id": "M019",
            "chat_id": "C04",
            "sender_id": "ME",
            "sender_name": "ME",
            "timestamp": 1787535500,
            "category": "goal",
            "matched_signals": ["target"],
            "original_text": "10 retainer clients by March is our target",
        },
    ]
    return EvidenceBatch(
        chat_id="C04",
        messages=messages,
        candidates=candidates,
        candidate_message_ids=["M016", "M018", "M019"],
        categories=["commitment", "question", "goal"],
    )


# ==============================================================================
# 1. Goal acceptance and rejection tests
# ==============================================================================

def test_goal_accepted_with_valid_evidence_and_enums(sample_chat_batch, sample_context):
    raw_payload = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Reach 10 clients by March",
                "area": "work",
                "type": "milestone",
                "horizon": "1yr",
                "evidence_msgs": ["M019"],
                "target_date": "2026-03-31",
                "is_genuine": True,
            }
        ]
    })
    diag = ChatParserDiagnostics()
    loops, goals = parse_chat_interpretation_response(raw_payload, sample_chat_batch, sample_context, diagnostics=diag)

    assert len(goals) == 1
    assert goals[0].title == "Reach 10 clients by March"
    assert goals[0].area == "work"
    assert goals[0].type == "milestone"
    assert goals[0].horizon == "1yr"
    assert diag.goals_total_raw == 1
    assert diag.goals_accepted == 1
    assert diag.goals_invalid_enums == 0
    assert diag.goals_invalid_or_ungrounded_evidence == 0
    assert diag.goals_rejected_not_genuine == 0


@pytest.mark.parametrize("area,goal_type,horizon,expected_reason", [
    ("invalid_area", "milestone", "1yr", "invalid_area_enum"),
    ("work", "invalid_type", "1yr", "invalid_type_enum"),
    ("work", "milestone", "invalid_horizon", "invalid_horizon_enum"),
])
def test_goal_rejected_for_invalid_area_type_or_horizon(sample_chat_batch, sample_context, area, goal_type, horizon, expected_reason):
    raw_payload = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Reach 10 clients by March",
                "area": area,
                "type": goal_type,
                "horizon": horizon,
                "evidence_msgs": ["M019"],
                "is_genuine": True,
            }
        ]
    })
    diag = ChatParserDiagnostics()
    loops, goals = parse_chat_interpretation_response(raw_payload, sample_chat_batch, sample_context, diagnostics=diag)

    assert len(goals) == 0
    assert diag.goals_total_raw == 1
    assert diag.goals_accepted == 0
    assert diag.goals_invalid_enums == 1
    assert len(diag.rejection_details) == 1
    assert diag.rejection_details[0]["reason"] == expected_reason


@pytest.mark.parametrize("evidence_ids", [[], ["M999"], ["NOT_EXIST"]])
def test_goal_rejected_for_empty_or_ungrounded_evidence(sample_chat_batch, sample_context, evidence_ids):
    raw_payload = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Reach 10 clients by March",
                "area": "work",
                "type": "milestone",
                "horizon": "1yr",
                "evidence_msgs": evidence_ids,
                "is_genuine": True,
            }
        ]
    })
    diag = ChatParserDiagnostics()
    loops, goals = parse_chat_interpretation_response(raw_payload, sample_chat_batch, sample_context, diagnostics=diag)

    assert len(goals) == 0
    assert diag.goals_total_raw == 1
    assert diag.goals_accepted == 0
    assert diag.goals_invalid_or_ungrounded_evidence == 1
    assert len(diag.rejection_details) == 1
    assert diag.rejection_details[0]["reason"] == "empty_or_ungrounded_evidence"


def test_goal_rejected_when_is_genuine_is_false(sample_chat_batch, sample_context):
    raw_payload = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Reach 10 clients by March",
                "area": "work",
                "type": "milestone",
                "horizon": "1yr",
                "evidence_msgs": ["M019"],
                "is_genuine": False,
            }
        ]
    })
    diag = ChatParserDiagnostics()
    loops, goals = parse_chat_interpretation_response(raw_payload, sample_chat_batch, sample_context, diagnostics=diag)

    assert len(goals) == 0
    assert diag.goals_total_raw == 1
    assert diag.goals_accepted == 0
    assert diag.goals_rejected_not_genuine == 1
    assert len(diag.rejection_details) == 1
    assert diag.rejection_details[0]["reason"] == "rejected_not_genuine"


def test_goal_rejected_when_is_genuine_is_missing_or_invalid(sample_chat_batch, sample_context):
    raw_payload = json.dumps({
        "open_loops": [],
        "goals": [
            {
                "title": "Reach 10 clients by March",
                "area": "work",
                "type": "milestone",
                "horizon": "1yr",
                "evidence_msgs": ["M019"],
                # missing is_genuine
            },
            {
                "title": "Reach 10 clients by March 2",
                "area": "work",
                "type": "milestone",
                "horizon": "1yr",
                "evidence_msgs": ["M019"],
                "is_genuine": "yes",  # not a boolean
            }
        ]
    })
    diag = ChatParserDiagnostics()
    loops, goals = parse_chat_interpretation_response(raw_payload, sample_chat_batch, sample_context, diagnostics=diag)

    assert len(goals) == 0
    assert diag.goals_total_raw == 2
    assert diag.goals_accepted == 0
    assert diag.goals_invalid_or_missing_is_genuine == 2


# ==============================================================================
# 2. Open loop rejection tests
# ==============================================================================

def test_open_loop_rejected_for_invalid_candidate_source_message(sample_chat_batch, sample_context):
    raw_payload = json.dumps({
        "open_loops": [
            {
                "source_msg": "M999",  # not in candidate_message_ids
                "chat_id": "C04",
                "title": "Do something invalid",
                "type": "obligation",
                "status": "open",
                "owner": "Alice",
                "is_genuine": True,
            }
        ],
        "goals": []
    })
    diag = ChatParserDiagnostics()
    loops, goals = parse_chat_interpretation_response(raw_payload, sample_chat_batch, sample_context, diagnostics=diag)

    assert len(loops) == 0
    assert diag.open_loops_total_raw == 1
    assert diag.open_loops_accepted == 0
    assert diag.open_loops_invalid_or_ungrounded_source == 1
    assert len(diag.rejection_details) == 1
    assert diag.rejection_details[0]["reason"] == "invalid_or_ungrounded_source"


@pytest.mark.parametrize("status,loop_type,expected_reason", [
    ("invalid_status", "commitment", "invalid_status_enum"),
    ("open", "invalid_type", "invalid_type_enum"),
])
def test_open_loop_rejected_for_invalid_status_or_type(sample_chat_batch, sample_context, status, loop_type, expected_reason):
    raw_payload = json.dumps({
        "open_loops": [
            {
                "source_msg": "M018",
                "chat_id": "C04",
                "title": "Send invoice",
                "type": loop_type,
                "status": status,
                "owner": "ME",
                "is_genuine": True,
            }
        ],
        "goals": []
    })
    diag = ChatParserDiagnostics()
    loops, goals = parse_chat_interpretation_response(raw_payload, sample_chat_batch, sample_context, diagnostics=diag)

    assert len(loops) == 0
    assert diag.open_loops_total_raw == 1
    assert diag.open_loops_accepted == 0
    assert diag.open_loops_invalid_enums == 1
    assert len(diag.rejection_details) == 1
    assert diag.rejection_details[0]["reason"] == expected_reason


# ==============================================================================
# 3. Contract compatibility and diagnostics behavior tests
# ==============================================================================

def test_existing_parser_callers_retain_return_contract(sample_chat_batch, sample_context):
    raw_payload = json.dumps({
        "open_loops": [
            {
                "source_msg": "M018",
                "chat_id": "C04",
                "title": "Send invoice to Brightside",
                "type": "commitment",
                "status": "open",
                "owner": "ME",
                "is_genuine": True,
            }
        ],
        "goals": [
            {
                "title": "Reach 10 clients by March",
                "area": "work",
                "type": "milestone",
                "horizon": "1yr",
                "evidence_msgs": ["M019"],
                "is_genuine": True,
            }
        ]
    })
    # Calling without diagnostics parameter
    result = parse_chat_interpretation_response(raw_payload, sample_chat_batch, sample_context)

    assert isinstance(result, tuple)
    assert len(result) == 2
    loops, goals = result
    assert isinstance(loops, list)
    assert isinstance(goals, list)
    assert len(loops) == 1
    assert isinstance(loops[0], InterpretedOpenLoop)
    assert len(goals) == 1
    assert isinstance(goals[0], InterpretedGoal)


def test_diagnostics_do_not_change_output(sample_chat_batch, sample_context):
    raw_payload = json.dumps({
        "open_loops": [
            {
                "source_msg": "M018",
                "chat_id": "C04",
                "title": "Send invoice to Brightside",
                "type": "commitment",
                "status": "open",
                "owner": "ME",
                "is_genuine": True,
            }
        ],
        "goals": [
            {
                "title": "Reach 10 clients by March",
                "area": "work",
                "type": "milestone",
                "horizon": "1yr",
                "evidence_msgs": ["M019"],
                "is_genuine": True,
            }
        ]
    })
    diag = ChatParserDiagnostics()
    loops_with_diag, goals_with_diag = parse_chat_interpretation_response(
        raw_payload, sample_chat_batch, sample_context, diagnostics=diag
    )
    loops_without_diag, goals_without_diag = parse_chat_interpretation_response(
        raw_payload, sample_chat_batch, sample_context
    )

    assert len(loops_with_diag) == len(loops_without_diag) == 1
    assert loops_with_diag[0].source_msg == loops_without_diag[0].source_msg
    assert len(goals_with_diag) == len(goals_without_diag) == 1
    assert goals_with_diag[0].title == goals_without_diag[0].title


# ==============================================================================
# 4. Candidate-to-batch tracing tests
# ==============================================================================

def test_candidate_to_batch_tracing(sample_chat_batch):
    candidates = [
        {"message_id": "M016", "chat_id": "C04", "category": "question"},
        {"message_id": "M018", "chat_id": "C04", "category": "commitment"},
        {"message_id": "M99", "chat_id": "C02", "category": "unbatched"},
    ]
    traces = trace_candidates_in_batches([sample_chat_batch], candidates)

    assert len(traces) == 3
    t1 = next(t for t in traces if t.message_id == "M016")
    assert t1.entered_batch is True
    assert t1.batch_id == sample_chat_batch.batch_id
    assert t1.included_in_prompt is True
    assert t1.category == "question"

    t2 = next(t for t in traces if t.message_id == "M018")
    assert t2.entered_batch is True
    assert t2.batch_id == sample_chat_batch.batch_id
    assert t2.included_in_prompt is True

    t3 = next(t for t in traces if t.message_id == "M99")
    assert t3.entered_batch is False
    assert t3.batch_id is None
    assert t3.included_in_prompt is False


def test_evidence_batcher_trace_candidates(sample_context):
    candidates = [
        {"message_id": "M016", "chat_id": "C04", "category": "question"},
        {"message_id": "M018", "chat_id": "C04", "category": "commitment"},
    ]
    batcher = EvidenceBatcher()
    batches = batcher.batch(sample_context, candidates)
    traces = batcher.trace_candidates(candidates)

    assert len(traces) == 2
    assert all(t.entered_batch for t in traces)
    assert all(t.included_in_prompt for t in traces)
    assert batcher.last_traces == traces


# ==============================================================================
# 5. Interpretation diagnostics aggregate counts tests
# ==============================================================================

def test_interpretation_diagnostics_aggregate_counts(sample_context, sample_chat_batch):
    def fake_chat(messages, **kwargs):
        if messages and messages[0].get("content") == PARTICIPANT_SYSTEM_PROMPT:
            return json.dumps({"people": [], "organisations": []})

        return json.dumps({
            "open_loops": [
                {
                    "source_msg": "M018",
                    "chat_id": "C04",
                    "title": "Send invoice to Brightside",
                    "type": "commitment",
                    "status": "open",
                    "owner": "ME",
                    "is_genuine": True,
                },
                {
                    "source_msg": "M999",  # Rejected: ungrounded
                    "chat_id": "C04",
                    "title": "Invalid action",
                    "type": "obligation",
                    "status": "open",
                    "owner": "Alice",
                    "is_genuine": True,
                }
            ],
            "goals": [
                {
                    "title": "Reach 10 clients by March",
                    "area": "work",
                    "type": "milestone",
                    "horizon": "1yr",
                    "evidence_msgs": ["M019"],
                    "is_genuine": True,
                },
                {
                    "title": "Invalid Goal",
                    "area": "bad_area",  # Rejected: invalid enum
                    "type": "milestone",
                    "horizon": "1yr",
                    "evidence_msgs": ["M019"],
                    "is_genuine": True,
                }
            ]
        })

    interpreter = LLMEvidenceInterpreter(
        chat_fn=fake_chat,
        model="test-model",
        cache_enabled=False,
    )
    candidates = [
        {"message_id": "M016", "chat_id": "C04", "category": "question"},
        {"message_id": "M018", "chat_id": "C04", "category": "commitment"},
        {"message_id": "M019", "chat_id": "C04", "category": "goal"},
    ]

    result = interpreter.interpret(sample_context, candidates)
    diag = result.diagnostics

    assert diag.chat_batches_created == 1
    assert diag.live_llm_calls >= 1
    assert diag.cache_hits == 0
    assert diag.offline_cache_misses == 0
    assert diag.successful_parsed_batches >= 1
    assert diag.failed_batches == 0
    assert diag.raw_goals_returned == 2
    assert diag.goals_accepted == 1
    assert diag.raw_open_loops_returned == 2
    assert diag.open_loops_accepted == 1

    d_dict = diag.to_dict()
    assert d_dict["chat_batches_created"] == 1
    assert d_dict["raw_goals_returned"] == 2
    assert d_dict["goals_accepted"] == 1
    assert d_dict["raw_open_loops_returned"] == 2
    assert d_dict["open_loops_accepted"] == 1

    # Ensure no sensitive tokens or strings leaked
    for key, val in d_dict.items():
        assert not isinstance(val, str) or ("api" not in val.lower() and "bearer" not in val.lower())
