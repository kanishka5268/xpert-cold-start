"""Unit and regression tests for goal data models, candidate conversion, and early evidence validation."""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.goals import (
    CandidateGoal,
    GoalExtractionConfig,
    GoalExtractionDiagnostics,
    GoalExtractionResult,
    build_timestamp_index,
    convert_interpreted_goals_to_candidates,
    extract_goals,
    extract_seed_goals,
    interpret_goal_evidence,
)
from src.interpretation.interpreter import BatchFailureDiagnostic, LLMConfigurationError
from src.interpretation.records import InterpretedGoal
from src.preprocessing.context_builder import ConversationContext, build_conversation_context


# ==============================================================================
# 1. Diagnostics & Result Model Tests
# ==============================================================================


def test_goal_extraction_diagnostics_defaults():
    """Verify default diagnostic counters are initialized to zero without leaks."""
    diag = GoalExtractionDiagnostics()
    assert diag.goal_candidates_received == 0
    assert diag.batches_created == 0
    assert diag.batches_succeeded == 0
    assert diag.batches_failed == 0
    assert diag.batches_unattempted == 0
    assert diag.invalid_records_discarded == 0
    assert diag.interpreted_goals_received == 0
    assert diag.genuine_goals_received == 0
    assert diag.candidates_created == 0
    assert diag.candidates_rejected == 0
    assert diag.candidates_validated == 0
    assert diag.invalid_evidence_ids_discarded == 0
    assert diag.ungrounded_candidates_discarded == 0
    assert diag.deduplicated_candidates == 0
    assert diag.seed_goals_matched == 0
    assert diag.final_goals_emitted == 0


def test_goal_extraction_diagnostics_no_raw_text():
    """Verify diagnostic fields contain strictly numeric counters and no personal text fields."""
    diag = GoalExtractionDiagnostics()
    for field_name, val in diag.__dict__.items():
        assert isinstance(val, int), f"Field '{field_name}' must be an integer counter, got {type(val)}"


def test_goal_extraction_result_defaults_and_goal_count():
    """Verify GoalExtractionResult defaults, goal_count property, and mutable field isolation."""
    res1 = GoalExtractionResult()
    res2 = GoalExtractionResult()

    assert res1.goal_count == 0
    assert res1.goals == []
    assert isinstance(res1.diagnostics, GoalExtractionDiagnostics)

    # Verify mutable list isolation between instances
    res1.goals.append({"title": "Test Goal", "area": "health", "type": "habit", "horizon": "3mo", "evidence": ["M1"]})
    assert res1.goal_count == 1
    assert res2.goal_count == 0
    assert len(res2.goals) == 0

    # Verify mutable diagnostics isolation
    res1.diagnostics.candidates_validated += 5
    assert res2.diagnostics.candidates_validated == 0


# ==============================================================================
# 2. Candidate Conversion - Positive & Valid Cases
# ==============================================================================


def test_convert_valid_genuine_goal():
    """Verify converting a fully valid genuine InterpretedGoal to CandidateGoal."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": "C01", "sender_id": "P01", "ts": 1000},
        "M02": {"id": "M02", "chat_id": "C01", "sender_id": "P01", "ts": 1010},
    }
    goal = InterpretedGoal(
        title="Run the half marathon in January",
        area="health",
        type="milestone",
        horizon="1yr",
        evidence_msgs=["M01", "M02"],
        is_genuine=True,
        confidence=0.85,
    )
    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages, diagnostics=diag)

    assert len(candidates) == 1
    cand = candidates[0]
    assert isinstance(cand, CandidateGoal)
    assert cand.title == "Run the half marathon in January"
    assert cand.area == "health"
    assert cand.type == "milestone"
    assert cand.horizon == "1yr"
    assert cand.evidence == ["M01", "M02"]
    assert cand.confidence == 0.85
    assert cand.source_chats == ["C01"]
    assert cand.is_seed is False
    assert cand.evidence_count == 2
    assert cand.chat_count == 1

    # Diagnostics verification
    assert diag.interpreted_goals_received == 1
    assert diag.genuine_goals_received == 1
    assert diag.candidates_created == 1
    assert diag.candidates_validated == 1
    assert diag.candidates_rejected == 0
    assert diag.invalid_evidence_ids_discarded == 0
    assert diag.ungrounded_candidates_discarded == 0


def test_convert_evidence_spanning_multiple_chats_and_deterministic_order():
    """Verify cross-chat evidence derives sorted unique chat IDs and preserves evidence order."""
    dataset_messages = {
        "M10": {"id": "M10", "chat_id": "C_WORK", "ts": 200},
        "M20": {"id": "M20", "chat_id": "C_FAMILY", "ts": 100},
        "M30": {"id": "M30", "chat_id": "C_WORK", "ts": 300},
    }
    goal = InterpretedGoal(
        title="Family trip to Coorg",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence_msgs=["M10", "M20", "M30"],
        is_genuine=True,
        confidence=0.9,
    )
    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages, diagnostics=diag)

    assert len(candidates) == 1
    cand = candidates[0]
    # Evidence insertion order strictly preserved
    assert cand.evidence == ["M10", "M20", "M30"]
    # Source chats are sorted uniquely
    assert cand.source_chats == ["C_FAMILY", "C_WORK"]
    assert cand.chat_count == 2
    assert cand.evidence_count == 3


# ==============================================================================
# 3. Candidate Conversion - Rejection & Negative Cases
# ==============================================================================


def test_reject_non_genuine_goals():
    """Verify goals with is_genuine != True are strictly rejected."""
    dataset_messages = {"M1": {"id": "M1", "chat_id": "C1", "ts": 100}}

    # False
    g_false = InterpretedGoal(
        title="Buy groceries", area="home", type="milestone", horizon="3mo",
        evidence_msgs=["M1"], is_genuine=False, confidence=0.5
    )
    # Non-bool / truthy string
    g_truthy = InterpretedGoal(
        title="Fake goal", area="home", type="milestone", horizon="3mo",
        evidence_msgs=["M1"], is_genuine="True", confidence=0.5  # type: ignore
    )
    # None
    g_none = InterpretedGoal(
        title="None goal", area="home", type="milestone", horizon="3mo",
        evidence_msgs=["M1"], is_genuine=None, confidence=0.5  # type: ignore
    )

    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([g_false, g_truthy, g_none], dataset_messages, diagnostics=diag)

    assert candidates == []
    assert diag.interpreted_goals_received == 3
    assert diag.genuine_goals_received == 0
    assert diag.candidates_created == 0
    assert diag.candidates_rejected == 3


def test_reject_invalid_enums_and_empty_titles():
    """Verify invalid area, type, horizon, or empty titles are rejected without silent repair."""
    dataset_messages = {"M1": {"id": "M1", "chat_id": "C1", "ts": 100}}

    invalid_goals = [
        # Empty title
        InterpretedGoal(title="   ", area="health", type="habit", horizon="3mo", evidence_msgs=["M1"], is_genuine=True),
        # Invalid area (e.g. 'career' instead of 'work')
        InterpretedGoal(title="Lead team", area="career", type="milestone", horizon="1yr", evidence_msgs=["M1"], is_genuine=True),
        # Invalid type (e.g. 'project' instead of 'milestone')
        InterpretedGoal(title="Build house", area="home", type="project", horizon="3yr", evidence_msgs=["M1"], is_genuine=True),
        # Invalid horizon (e.g. '10yr')
        InterpretedGoal(title="Retire early", area="finances", type="milestone", horizon="10yr", evidence_msgs=["M1"], is_genuine=True),
    ]

    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates(invalid_goals, dataset_messages, diagnostics=diag)

    assert candidates == []
    assert diag.interpreted_goals_received == 4
    assert diag.genuine_goals_received == 4
    assert diag.candidates_created == 0
    assert diag.candidates_rejected == 4


def test_reject_empty_or_none_evidence_list():
    """Verify goals with empty or None evidence lists are rejected."""
    dataset_messages = {"M1": {"id": "M1", "chat_id": "C1", "ts": 100}}

    g_empty = InterpretedGoal(
        title="Meditate daily", area="health", type="habit", horizon="3mo",
        evidence_msgs=[], is_genuine=True
    )
    g_none = InterpretedGoal(
        title="Save money", area="finances", type="milestone", horizon="1yr",
        evidence_msgs=None, is_genuine=True  # type: ignore
    )

    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([g_empty, g_none], dataset_messages, diagnostics=diag)

    assert candidates == []
    assert diag.candidates_rejected == 2


def test_reject_when_all_evidence_is_ungrounded():
    """Verify candidate is discarded when none of its evidence IDs exist in the dataset."""
    dataset_messages = {"M_REAL": {"id": "M_REAL", "chat_id": "C1", "ts": 100}}

    goal = InterpretedGoal(
        title="Visit Japan",
        area="experiences",
        type="experience",
        horizon="1yr",
        evidence_msgs=["M_GHOST_1", "M_GHOST_2"],
        is_genuine=True,
    )

    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages, diagnostics=diag)

    assert candidates == []
    assert diag.candidates_created == 1
    assert diag.candidates_validated == 0
    assert diag.candidates_rejected == 1
    assert diag.invalid_evidence_ids_discarded == 2
    assert diag.ungrounded_candidates_discarded == 1


# ==============================================================================
# 4. Evidence Grounding & Sanitization Tests
# ==============================================================================


def test_filter_partial_invalid_and_non_string_evidence():
    """Verify non-string IDs and non-existent IDs are discarded while preserving valid grounded IDs."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": "C01", "ts": 100},
        "M03": {"id": "M03", "chat_id": "C02", "ts": 120},
    }
    goal = InterpretedGoal(
        title="Maintain running routine",
        area="health",
        type="habit",
        horizon="3mo",
        evidence_msgs=["M01", "M_NONEXISTENT", 999, None, "M03"],  # type: ignore
        is_genuine=True,
    )

    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages, diagnostics=diag)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.evidence == ["M01", "M03"]
    assert cand.source_chats == ["C01", "C02"]

    # 5 total items in evidence_msgs; 2 valid retained -> 3 discarded
    assert diag.invalid_evidence_ids_discarded == 3
    assert diag.candidates_validated == 1
    assert diag.ungrounded_candidates_discarded == 0


def test_duplicate_evidence_ids_deduplicated():
    """Verify duplicate evidence IDs within a goal are deduplicated preserving the single entry."""
    dataset_messages = {"M01": {"id": "M01", "chat_id": "C01", "ts": 100}}
    goal = InterpretedGoal(
        title="Save 20 lakhs for down payment",
        area="home",
        type="milestone",
        horizon="1yr",
        evidence_msgs=["M01", "M01", "M01"],
        is_genuine=True,
    )

    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages, diagnostics=diag)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.evidence == ["M01"]
    assert cand.evidence_count == 1
    # 3 total submitted - 1 retained = 2 duplicate/invalid discarded
    assert diag.invalid_evidence_ids_discarded == 2
    assert diag.candidates_validated == 1


def test_missing_or_empty_chat_id_in_message():
    """Verify message with missing or empty chat_id is retained as evidence but does not inject empty chat."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": "", "ts": 100},
        "M02": {"id": "M02", "ts": 101},  # chat_id missing entirely
        "M03": {"id": "M03", "chat_id": "C03", "ts": 102},
    }
    goal = InterpretedGoal(
        title="Buy a 3BHK",
        area="home",
        type="milestone",
        horizon="3yr",
        evidence_msgs=["M01", "M02", "M03"],
        is_genuine=True,
    )

    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.evidence == ["M01", "M02", "M03"]
    # Only C03 is valid non-empty
    assert cand.source_chats == ["C03"]


def test_confidence_normalization_policy():
    """Verify invalid or non-finite confidence values are safely normalized using resolver policy."""
    dataset_messages = {"M1": {"id": "M1", "chat_id": "C1", "ts": 100}}

    test_cases = [
        (True, 0.5),           # boolean True -> 0.5
        (False, 0.5),          # boolean False -> 0.5
        ("0.9", 0.5),          # string -> 0.5
        (None, 0.5),           # None -> 0.5
        (float("nan"), 0.5),   # NaN -> 0.5
        (float("inf"), 0.5),   # inf -> 0.5
        (-0.2, 0.5),           # out-of-range negative -> 0.5
        (1.5, 0.5),            # out-of-range positive -> 0.5
        (0.75, 0.75),          # valid float -> preserved
        (1.0, 1.0),            # valid boundary -> preserved
        (0.0, 0.0),            # valid boundary -> preserved
    ]

    for raw_conf, expected_conf in test_cases:
        goal = InterpretedGoal(
            title="Start running four times a week",
            area="health",
            type="habit",
            horizon="3mo",
            evidence_msgs=["M1"],
            is_genuine=True,
            confidence=raw_conf,  # type: ignore
        )
        cands = convert_interpreted_goals_to_candidates([goal], dataset_messages)
        assert len(cands) == 1
        assert cands[0].confidence == expected_conf


def test_input_immutability():
    """Verify input InterpretedGoal objects and dataset_messages dictionary are not mutated."""
    original_msg = {"id": "M01", "chat_id": "C01", "ts": 100}
    dataset_messages = {"M01": copy.deepcopy(original_msg)}
    goal = InterpretedGoal(
        title="Build a running routine",
        area="health",
        type="habit",
        horizon="3mo",
        evidence_msgs=["M01", "M_GHOST"],
        is_genuine=True,
        confidence=0.7,
    )
    goal_copy = copy.deepcopy(goal)

    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages)

    assert len(candidates) == 1
    # Check that original goal was not mutated
    assert goal.evidence_msgs == goal_copy.evidence_msgs
    assert goal.title == goal_copy.title
    assert goal.confidence == goal_copy.confidence
    # Check that dataset message was not mutated
    assert dataset_messages["M01"] == original_msg


# ==============================================================================
# 5. Regression Tests for Evidence-ID Exact Grounding (Tests A-E)
# ==============================================================================


def test_regression_test_a_padded_id_rejected():
    """Test A — Padded ID rejected: ' M01 ' is not normalized to 'M01' and is rejected."""
    dataset_messages = {"M01": {"id": "M01", "chat_id": "C01", "ts": 100}}
    goal = InterpretedGoal(
        title="Start running four times a week",
        area="health",
        type="habit",
        horizon="3mo",
        evidence_msgs=[" M01 "],
        is_genuine=True,
    )
    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages, diagnostics=diag)

    assert candidates == []
    assert diag.ungrounded_candidates_discarded == 1
    assert diag.invalid_evidence_ids_discarded == 1
    assert diag.candidates_created == 1
    assert diag.candidates_validated == 0
    assert diag.candidates_rejected == 1


def test_regression_test_b_exact_id_accepted():
    """Test B — Exact ID accepted: 'M01' matches exactly, deriving the correct chat."""
    dataset_messages = {"M01": {"id": "M01", "chat_id": "C01", "ts": 100}}
    goal = InterpretedGoal(
        title="Run the half marathon in January",
        area="health",
        type="milestone",
        horizon="1yr",
        evidence_msgs=["M01"],
        is_genuine=True,
    )
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.evidence == ["M01"]
    assert cand.source_chats == ["C01"]


def test_regression_test_c_whitespace_only_id_rejected():
    """Test C — Whitespace-only ID rejected: '   ' rejected and candidate rejected."""
    dataset_messages = {"M01": {"id": "M01", "chat_id": "C01", "ts": 100}}
    goal = InterpretedGoal(
        title="Maintain daily meditation",
        area="health",
        type="habit",
        horizon="3mo",
        evidence_msgs=["   "],
        is_genuine=True,
    )
    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages, diagnostics=diag)

    assert candidates == []
    assert diag.invalid_evidence_ids_discarded == 1
    assert diag.candidates_rejected == 1
    assert diag.candidates_validated == 0


def test_regression_test_d_mixed_exact_padded_and_nonexistent_ids():
    """Test D — Mixed exact, padded, and nonexistent IDs: ['M01', ' M01 ', 'M_GHOST', 'M02']."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": "C01", "ts": 100},
        "M02": {"id": "M02", "chat_id": "C02", "ts": 200},
    }
    goal = InterpretedGoal(
        title="Family trip to Coorg",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence_msgs=["M01", " M01 ", "M_GHOST", "M02"],
        is_genuine=True,
    )
    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages, diagnostics=diag)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.evidence == ["M01", "M02"]
    assert cand.source_chats == ["C01", "C02"]
    # 4 submitted - 2 valid retained = 2 discarded (' M01 ' and 'M_GHOST')
    assert diag.invalid_evidence_ids_discarded == 2
    assert diag.candidates_validated == 1
    assert diag.ungrounded_candidates_discarded == 0


def test_regression_test_e_duplicate_behavior_remains_correct():
    """Test E — Duplicate behavior remains correct: ['M01', 'M01', ' M01 ']."""
    dataset_messages = {"M01": {"id": "M01", "chat_id": "C01", "ts": 100}}
    goal = InterpretedGoal(
        title="Save 20 lakhs for down payment",
        area="home",
        type="milestone",
        horizon="1yr",
        evidence_msgs=["M01", "M01", " M01 "],
        is_genuine=True,
    )
    diag = GoalExtractionDiagnostics()
    candidates = convert_interpreted_goals_to_candidates([goal], dataset_messages, diagnostics=diag)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.evidence == ["M01"]
    assert cand.evidence_count == 1
    assert cand.source_chats == ["C01"]
    # 3 submitted - 1 valid retained = 2 discarded (duplicate 'M01' and ungrounded ' M01 ')
    assert diag.invalid_evidence_ids_discarded == 2
    assert diag.candidates_validated == 1


# ==============================================================================
# 6. Milestone 3: Evidence Batcher + Interpreter Orchestration Tests
# ==============================================================================


def _create_test_context(
    chat_messages: Dict[str, List[Dict[str, Any]]],
    seed: Optional[Dict[str, Any]] = None,
) -> ConversationContext:
    """Helper to build a valid ConversationContext from per-chat messages."""
    all_msgs: List[Dict[str, Any]] = []
    messages_by_chat: Dict[str, List[Dict[str, Any]]] = {}
    messages_by_id: Dict[str, Dict[str, Any]] = {}
    chats_by_id: Dict[str, Dict[str, Any]] = {}

    for cid, msgs in chat_messages.items():
        chats_by_id[cid] = {
            "id": cid,
            "name": f"Chat {cid}",
            "is_group": False,
            "participants": ["ME", "P01"],
        }
        messages_by_chat[cid] = msgs
        for m in msgs:
            all_msgs.append(m)
            messages_by_id[m["id"]] = m

    dataset = {
        "chats": list(chats_by_id.values()),
        "chats_by_id": chats_by_id,
        "messages": all_msgs,
        "messages_by_id": messages_by_id,
        "messages_by_chat": messages_by_chat,
        "participants": {
            "ME": {"sender_names": ["Me"]},
            "P01": {"sender_names": ["Partner"]},
        },
        "seed": dict(seed) if isinstance(seed, dict) else {},
    }
    return build_conversation_context(dataset)


class MockEvidenceInterpreter:
    """Mock evidence interpreter for testing chat batch orchestration without network or LLM calls."""

    def __init__(
        self,
        responses: Optional[Dict[str, Tuple[List[Any], List[Any], Optional[BatchFailureDiagnostic]]]] = None,
        default_goals: Optional[List[Any]] = None,
        raise_error: Optional[Exception] = None,
    ):
        self.responses = responses or {}
        self.default_goals = default_goals or []
        self.raise_error = raise_error
        self.chat_batch_calls: List[Tuple[Any, Optional[ConversationContext]]] = []
        self.participant_batch_calls: List[Any] = []

    def interpret_chat_batch(
        self, batch: Any, context: Optional[ConversationContext] = None
    ) -> Tuple[List[Any], List[Any], Optional[BatchFailureDiagnostic]]:
        self.chat_batch_calls.append((batch, context))
        if self.raise_error is not None:
            raise self.raise_error
        if hasattr(batch, "chat_id") and batch.chat_id in self.responses:
            return self.responses[batch.chat_id]
        return [], list(self.default_goals), None

    def interpret_participant_batch(
        self, dossiers: Any, chat_metadata: Any, context: Optional[ConversationContext] = None
    ) -> Tuple[List[Any], List[Any], Optional[BatchFailureDiagnostic]]:
        self.participant_batch_calls.append((dossiers, chat_metadata, context))
        return [], [], None


def test_interpret_goal_evidence_empty_candidates():
    """Test A: Empty candidates input returns [] immediately without calling interpreter or creating batches."""
    context = _create_test_context(
        {"C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "test", "kind": "text"}]}
    )
    mock = MockEvidenceInterpreter()
    diag = GoalExtractionDiagnostics()

    result = interpret_goal_evidence(context, [], mock, diagnostics=diag)

    assert result == []
    assert len(mock.chat_batch_calls) == 0
    assert diag.goal_candidates_received == 0
    assert diag.batches_created == 0
    assert diag.batches_succeeded == 0
    assert diag.batches_failed == 0
    assert diag.batches_unattempted == 0
    assert diag.invalid_records_discarded == 0


def test_interpret_goal_evidence_single_batch():
    """Test B: Single batch successfully executes and returns typed InterpretedGoal records."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Planning trip to Paris", "kind": "text"}
        ]
    }
    context = _create_test_context(chat_msgs)
    candidates = [{"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100}]

    goal = InterpretedGoal(
        title="Trip to Paris",
        area="experiences",
        type="experience",
        horizon="1yr",
        evidence_msgs=["M01"],
        is_genuine=True,
        confidence=0.9,
    )
    mock = MockEvidenceInterpreter(default_goals=[goal])
    diag = GoalExtractionDiagnostics()

    result = interpret_goal_evidence(context, candidates, mock, diagnostics=diag)

    assert len(result) == 1
    assert result[0] == goal
    assert len(mock.chat_batch_calls) == 1
    batch_arg, ctx_arg = mock.chat_batch_calls[0]
    assert batch_arg.chat_id == "C01"
    assert ctx_arg is context
    assert diag.goal_candidates_received == 1
    assert diag.batches_created == 1
    assert diag.batches_succeeded == 1
    assert diag.batches_failed == 0
    assert diag.invalid_records_discarded == 0


def test_interpret_goal_evidence_multiple_batches_preserves_order():
    """Test C: Multi-batch execution across multiple chats preserves deterministic ordering."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal 1", "kind": "text"}],
        "C02": [{"id": "M02", "chat_id": "C02", "sender_id": "ME", "ts": 200, "body": "Goal 2", "kind": "text"}],
    }
    context = _create_test_context(chat_msgs)
    candidates = [
        {"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100},
        {"message_id": "M02", "chat_id": "C02", "category": "goal", "timestamp": 200},
    ]

    g1 = InterpretedGoal(
        title="Goal 1", area="work", type="milestone", horizon="3mo",
        evidence_msgs=["M01"], is_genuine=True
    )
    g2 = InterpretedGoal(
        title="Goal 2", area="health", type="habit", horizon="1yr",
        evidence_msgs=["M02"], is_genuine=True
    )

    mock = MockEvidenceInterpreter(responses={
        "C01": ([], [g1], None),
        "C02": ([], [g2], None),
    })
    diag = GoalExtractionDiagnostics()

    result = interpret_goal_evidence(context, candidates, mock, diagnostics=diag)

    assert result == [g1, g2]
    assert len(mock.chat_batch_calls) == 2
    assert mock.chat_batch_calls[0][0].chat_id == "C01"
    assert mock.chat_batch_calls[1][0].chat_id == "C02"
    assert diag.batches_created == 2
    assert diag.batches_succeeded == 2
    assert diag.batches_failed == 0


def test_interpret_goal_evidence_one_batch_fails_others_continue():
    """Test D: Partial failure tolerance: failing batch is isolated and other batches continue."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal 1", "kind": "text"}],
        "C02": [{"id": "M02", "chat_id": "C02", "sender_id": "ME", "ts": 200, "body": "Goal 2", "kind": "text"}],
    }
    context = _create_test_context(chat_msgs)
    candidates = [
        {"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100},
        {"message_id": "M02", "chat_id": "C02", "category": "goal", "timestamp": 200},
    ]

    fail_diag = BatchFailureDiagnostic(
        phase="chat",
        batch_id="C01",
        attempt_count=3,
        error_type="ResponseParsingError",
        error_message="Invalid JSON payload",
        timestamp=12345.0,
    )
    g2 = InterpretedGoal(
        title="Goal 2", area="health", type="habit", horizon="1yr",
        evidence_msgs=["M02"], is_genuine=True
    )

    mock = MockEvidenceInterpreter(responses={
        "C01": ([], [], fail_diag),
        "C02": ([], [g2], None),
    })
    diag = GoalExtractionDiagnostics()

    result = interpret_goal_evidence(context, candidates, mock, diagnostics=diag)

    # C01 failed, C02 succeeded
    assert result == [g2]
    assert diag.batches_created == 2
    assert diag.batches_succeeded == 1
    assert diag.batches_failed == 1
    assert diag.invalid_records_discarded == 0


def test_interpret_goal_evidence_all_batches_fail():
    """Test E: Total failure tolerance: all batches fail, returns [] safely without raising unhandled exceptions."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal 1", "kind": "text"}],
        "C02": [{"id": "M02", "chat_id": "C02", "sender_id": "ME", "ts": 200, "body": "Goal 2", "kind": "text"}],
    }
    context = _create_test_context(chat_msgs)
    candidates = [
        {"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100},
        {"message_id": "M02", "chat_id": "C02", "category": "goal", "timestamp": 200},
    ]

    d1 = BatchFailureDiagnostic(
        phase="chat", batch_id="C01", attempt_count=3,
        error_type="HTTPError:500", error_message="Internal Server Error", timestamp=100.0
    )
    d2 = BatchFailureDiagnostic(
        phase="chat", batch_id="C02", attempt_count=3,
        error_type="HTTPError:503", error_message="Service Unavailable", timestamp=200.0
    )

    mock = MockEvidenceInterpreter(responses={
        "C01": ([], [], d1),
        "C02": ([], [], d2),
    })
    diag = GoalExtractionDiagnostics()

    result = interpret_goal_evidence(context, candidates, mock, diagnostics=diag)

    assert result == []
    assert diag.batches_created == 2
    assert diag.batches_succeeded == 0
    assert diag.batches_failed == 2


def test_interpret_goal_evidence_discards_invalid_records():
    """Test F: Invalid records returned in goals list are discarded and increment invalid_records_discarded."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal 1", "kind": "text"}]
    }
    context = _create_test_context(chat_msgs)
    candidates = [{"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100}]

    valid_goal = InterpretedGoal(
        title="Valid Goal", area="finances", type="milestone", horizon="3mo",
        evidence_msgs=["M01"], is_genuine=True
    )
    invalid_records = [
        valid_goal,
        {"title": "Not an InterpretedGoal"},
        "invalid_string_record",
        None,
        42,
    ]

    mock = MockEvidenceInterpreter(responses={"C01": ([], invalid_records, None)})
    diag = GoalExtractionDiagnostics()

    result = interpret_goal_evidence(context, candidates, mock, diagnostics=diag)

    assert result == [valid_goal]
    assert diag.batches_succeeded == 1
    assert diag.invalid_records_discarded == 4


def test_interpret_goal_evidence_no_participant_interpretation():
    """Test G: Confirms interpret_participant_batch is never called during goal evidence interpretation."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal 1", "kind": "text"}]
    }
    context = _create_test_context(chat_msgs)
    candidates = [{"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100}]

    mock = MockEvidenceInterpreter()
    interpret_goal_evidence(context, candidates, mock)

    assert len(mock.chat_batch_calls) == 1
    assert len(mock.participant_batch_calls) == 0


def test_interpret_goal_evidence_input_immutability():
    """Test H: Verify context and candidate data structures are deeply unmodified after execution."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal 1", "kind": "text"}]
    }
    context = _create_test_context(chat_msgs)
    candidates = [{"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100}]

    context_chats_before = copy.deepcopy(context.chats)
    context_dataset_before = copy.deepcopy(context.dataset)
    candidates_before = copy.deepcopy(candidates)

    goal = InterpretedGoal(
        title="Goal 1", area="work", type="milestone", horizon="3mo",
        evidence_msgs=["M01"], is_genuine=True
    )
    mock = MockEvidenceInterpreter(default_goals=[goal])

    interpret_goal_evidence(context, candidates, mock)

    assert candidates == candidates_before
    assert context.chats == context_chats_before
    assert context.dataset == context_dataset_before


def test_interpret_goal_evidence_retry_behavior_preserved():
    """Test I: Verify interpreter is called once per batch and orchestration does not add duplicate retry loop."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal 1", "kind": "text"}],
        "C02": [{"id": "M02", "chat_id": "C02", "sender_id": "ME", "ts": 200, "body": "Goal 2", "kind": "text"}],
    }
    context = _create_test_context(chat_msgs)
    candidates = [
        {"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100},
        {"message_id": "M02", "chat_id": "C02", "category": "goal", "timestamp": 200},
    ]

    mock = MockEvidenceInterpreter()
    interpret_goal_evidence(context, candidates, mock)

    # Exactly 2 calls for 2 batches - no extra retry attempts in orchestration layer
    assert len(mock.chat_batch_calls) == 2


def test_interpret_goal_evidence_diagnostics_isolation():
    """Test J: Verify all diagnostic counters are integers and no raw messages or error strings are retained."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Secret message", "kind": "text"}],
        "C02": [{"id": "M02", "chat_id": "C02", "sender_id": "ME", "ts": 200, "body": "Another message", "kind": "text"}],
    }
    context = _create_test_context(chat_msgs)
    candidates = [
        {"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100},
        {"message_id": "M02", "chat_id": "C02", "category": "goal", "timestamp": 200},
    ]

    fail_diag = BatchFailureDiagnostic(
        phase="chat",
        batch_id="C01",
        attempt_count=3,
        error_type="HTTPError:500",
        error_message="Database secret error details",
        timestamp=100.0,
    )
    mock = MockEvidenceInterpreter(responses={
        "C01": ([], [], fail_diag),
        "C02": ([], [InterpretedGoal(title="Private Goal", area="health", type="habit", horizon="1yr", evidence_msgs=["M02"], is_genuine=True)], None),
    })
    diag = GoalExtractionDiagnostics()

    interpret_goal_evidence(context, candidates, mock, diagnostics=diag)

    # Validate all fields in diag are integers and contain no strings, personal names, or error texts
    for field_name, value in diag.__dict__.items():
        assert isinstance(value, int), f"Field '{field_name}' must be int, got {type(value)}"

    # Check that known sensitive substrings do not appear in the string representation of diag
    diag_str = str(diag)
    assert "Secret message" not in diag_str
    assert "Another message" not in diag_str
    assert "Private Goal" not in diag_str
    assert "Database secret error details" not in diag_str


def test_interpret_goal_evidence_configuration_error_bubbles_up():
    """Test K: Fatal LLMConfigurationError bubbles up immediately without suppression."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal 1", "kind": "text"}]
    }
    context = _create_test_context(chat_msgs)
    candidates = [{"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100}]

    mock = MockEvidenceInterpreter(raise_error=LLMConfigurationError("LLM API authentication failed (HTTP 401)"))

    with pytest.raises(LLMConfigurationError, match="HTTP 401"):
        interpret_goal_evidence(context, candidates, mock)


def test_interpret_goal_evidence_unexpected_exception_recorded_as_failed_batch():
    """Test L: Unexpected non-fatal exception during batch execution increments batches_failed and continues."""
    chat_msgs = {
        "C01": [{"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal 1", "kind": "text"}],
        "C02": [{"id": "M02", "chat_id": "C02", "sender_id": "ME", "ts": 200, "body": "Goal 2", "kind": "text"}],
    }
    context = _create_test_context(chat_msgs)
    candidates = [
        {"message_id": "M01", "chat_id": "C01", "category": "goal", "timestamp": 100},
        {"message_id": "M02", "chat_id": "C02", "category": "goal", "timestamp": 200},
    ]

    class FlakyMockInterpreter:
        def __init__(self):
            self.calls = 0

        def interpret_chat_batch(self, batch, context=None):
            self.calls += 1
            if batch.chat_id == "C01":
                raise RuntimeError("Unexpected internal crash")
            goal = InterpretedGoal(title="Goal 2", area="health", type="habit", horizon="1yr", evidence_msgs=["M02"], is_genuine=True)
            return [], [goal], None

    flaky = FlakyMockInterpreter()
    diag = GoalExtractionDiagnostics()

    result = interpret_goal_evidence(context, candidates, flaky, diagnostics=diag)

    assert len(result) == 1
    assert result[0].title == "Goal 2"
    assert diag.batches_created == 2
    assert diag.batches_failed == 1
    assert diag.batches_succeeded == 1


# ==============================================================================
# 7. Milestone 4: Main Goal Extraction Pipeline Tests
# ==============================================================================


def test_extract_goals_empty_conversation():
    """Test A: Empty conversation / no goal candidates returns valid empty result without LLM calls."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Hello, how are you?", "kind": "text"},
            {"id": "M02", "chat_id": "C01", "sender_id": "P01", "ts": 110, "body": "All good thanks!", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)
    mock = MockEvidenceInterpreter()
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag)

    assert isinstance(result, GoalExtractionResult)
    assert result.goals == []
    assert result.goal_count == 0
    assert len(mock.chat_batch_calls) == 0
    assert result.diagnostics.final_goals_emitted == 0
    assert result.diagnostics.goal_candidates_received == 0


def test_extract_goals_one_genuine_grounded_goal():
    """Test B: One genuine grounded goal is extracted, interpreted, and emitted in schema shape."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Planning a trip to Paris next summer", "kind": "text"}
        ]
    }
    context = _create_test_context(chat_msgs)
    goal = InterpretedGoal(
        title="Trip to Paris",
        area="experiences",
        type="experience",
        horizon="1yr",
        evidence_msgs=["M01"],
        is_genuine=True,
        confidence=0.85,
    )
    mock = MockEvidenceInterpreter(default_goals=[goal])
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag)

    assert result.goal_count == 1
    assert result.goals[0] == {
        "title": "Trip to Paris",
        "area": "experiences",
        "type": "experience",
        "horizon": "1yr",
        "evidence": ["M01"],
    }
    assert result.diagnostics.final_goals_emitted == 1
    assert result.diagnostics.candidates_validated == 1


def test_extract_goals_ungrounded_evidence():
    """Test C: Goal with ungrounded (nonexistent) evidence IDs is completely discarded."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Planning a trip to Paris next summer", "kind": "text"}
        ]
    }
    context = _create_test_context(chat_msgs)
    ghost_goal = InterpretedGoal(
        title="Trip to Paris",
        area="experiences",
        type="experience",
        horizon="1yr",
        evidence_msgs=["M_NONEXISTENT_GHOST"],
        is_genuine=True,
        confidence=0.9,
    )
    mock = MockEvidenceInterpreter(default_goals=[ghost_goal])
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag)

    assert result.goal_count == 0
    assert result.goals == []
    assert result.diagnostics.ungrounded_candidates_discarded >= 1
    assert result.diagnostics.final_goals_emitted == 0


def test_extract_goals_exact_evidence_id_protection():
    """Test D: Padded evidence IDs like ' M01 ' are strictly rejected and not matched to 'M01'."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Planning a trip to Paris next summer", "kind": "text"}
        ]
    }
    context = _create_test_context(chat_msgs)
    padded_goal = InterpretedGoal(
        title="Trip to Paris",
        area="experiences",
        type="experience",
        horizon="1yr",
        evidence_msgs=[" M01 "],
        is_genuine=True,
        confidence=0.85,
    )
    mock = MockEvidenceInterpreter(default_goals=[padded_goal])
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag)

    assert result.goal_count == 0
    assert result.goals == []


def test_extract_goals_semantic_duplicate_goals():
    """Test E: Semantically equivalent goals are merged and their evidence unioned chronologically."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "We want to visit Coorg this December", "kind": "text"},
            {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 200, "body": "Family trip to Coorg in December", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)
    g1 = InterpretedGoal(
        title="Family trip to Coorg", area="experiences", type="experience", horizon="3mo",
        evidence_msgs=["M01"], is_genuine=True, confidence=0.8
    )
    g2 = InterpretedGoal(
        title="Trip to Coorg", area="experiences", type="experience", horizon="3mo",
        evidence_msgs=["M02"], is_genuine=True, confidence=0.9
    )
    mock = MockEvidenceInterpreter(default_goals=[g1, g2])
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag)

    assert result.goal_count == 1
    goal = result.goals[0]
    assert goal["area"] == "experiences"
    assert goal["type"] == "experience"
    assert goal["evidence"] == ["M01", "M02"]
    assert result.diagnostics.deduplicated_candidates >= 1


def test_extract_goals_distinct_goals_remain_separate():
    """Test F: Goals with distinct actions or targets are not merged."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Family trip to Coorg in December", "kind": "text"},
            {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 200, "body": "Planning a trip to Paris next summer", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)
    g1 = InterpretedGoal(
        title="Family trip to Coorg", area="experiences", type="experience", horizon="3mo",
        evidence_msgs=["M01"], is_genuine=True
    )
    g2 = InterpretedGoal(
        title="Trip to Paris", area="experiences", type="experience", horizon="1yr",
        evidence_msgs=["M02"], is_genuine=True
    )
    mock = MockEvidenceInterpreter(default_goals=[g1, g2])
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag)

    assert result.goal_count == 2
    titles = {g["title"] for g in result.goals}
    assert "Family trip to Coorg" in titles
    assert "Trip to Paris" in titles


def test_extract_goals_seed_reconciliation():
    """Test G: Seed reconciliation adopts authoritative horizon and discards unmatched seed goals."""
    seed = {
        "goals": [
            {
                "title": "Reach 10 retainer clients by March",
                "area": "work",
                "type": "milestone",
                "horizon": "1yr",
            },
            {
                "title": "Buy a vacation home in Goa",
                "area": "home",
                "type": "milestone",
                "horizon": "3yr",
            },
        ]
    }
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "We want to reach 10 retainer clients by March", "kind": "text"}
        ]
    }
    context = _create_test_context(chat_msgs, seed=seed)
    cand_goal = InterpretedGoal(
        title="Reach 10 retainer clients by March",
        area="work",
        type="milestone",
        horizon="3mo",
        evidence_msgs=["M01"],
        is_genuine=True,
        confidence=0.85,
    )
    mock = MockEvidenceInterpreter(default_goals=[cand_goal])
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag)

    # Only 1 goal emitted (the matched seed goal); unmatched Goa vacation home is discarded
    assert result.goal_count == 1
    g = result.goals[0]
    assert g["title"] == "Reach 10 retainer clients by March"
    assert g["horizon"] == "1yr"  # Authoritative seed horizon adopted over candidate's 3mo
    assert g["evidence"] == ["M01"]  # Candidate's authentic conversation evidence retained
    assert result.diagnostics.seed_goals_matched == 1


def test_extract_goals_empty_seed_collection():
    """Test H: Reconciliation works properly when no seed goals exist in the dataset."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Planning a trip to Paris next summer", "kind": "text"}
        ]
    }
    context = _create_test_context(chat_msgs, seed={})
    goal = InterpretedGoal(
        title="Trip to Paris", area="experiences", type="experience", horizon="1yr",
        evidence_msgs=["M01"], is_genuine=True
    )
    mock = MockEvidenceInterpreter(default_goals=[goal])
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag)

    assert result.goal_count == 1
    assert result.goals[0]["title"] == "Trip to Paris"
    assert result.diagnostics.seed_goals_matched == 0


def test_extract_goals_max_goals_clamping():
    """Test I: Output is clamped to config.max_goals (default 10 or custom)."""
    # 12 distinct messages matching goal patterns
    chat_msgs = {
        "C01": [
            {"id": f"M{i:02d}", "chat_id": "C01", "sender_id": "ME", "ts": 100 + i * 10, "body": f"We want to reach {i+1} retainer clients by March", "kind": "text"}
            for i in range(12)
        ]
    }
    context = _create_test_context(chat_msgs)
    interpreted_goals = [
        InterpretedGoal(
            title=f"Reach {i+1} retainer clients by March",
            area="work",
            type="milestone",
            horizon="1yr",
            evidence_msgs=[f"M{i:02d}"],
            is_genuine=True,
            confidence=0.8,
        )
        for i in range(12)
    ]
    mock = MockEvidenceInterpreter(default_goals=interpreted_goals)

    # 1. Default config: max 10
    res_default = extract_goals(context, mock)
    assert res_default.goal_count == 10

    # 2. Custom config: max 4
    config_4 = GoalExtractionConfig(max_goals=4)
    res_custom = extract_goals(context, mock, config=config_4)
    assert res_custom.goal_count == 4


def test_extract_goals_final_evidence_validation():
    """Test J: Revalidates evidence at the final output barrier so every emitted goal has grounded evidence."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Planning a trip to Paris next summer", "kind": "text"},
            {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 200, "body": "Family trip to Coorg in December", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)
    g1 = InterpretedGoal(
        title="Trip to Paris", area="experiences", type="experience", horizon="1yr",
        evidence_msgs=["M01"], is_genuine=True
    )
    g2 = InterpretedGoal(
        title="Family trip to Coorg", area="experiences", type="experience", horizon="3mo",
        evidence_msgs=["M02"], is_genuine=True
    )
    mock = MockEvidenceInterpreter(default_goals=[g1, g2])

    result = extract_goals(context, mock)

    for g in result.goals:
        assert len(g["evidence"]) >= 1
        for mid in g["evidence"]:
            assert mid in context.dataset["messages_by_id"]


def test_extract_goals_timestamp_handling():
    """Test K: Robust timestamp index construction and evidence ordering under mixed/invalid timestamps."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 200, "body": "Family trip to Coorg", "kind": "text"},
            {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "We want to visit Coorg", "kind": "text"},
            {"id": "M03", "chat_id": "C01", "sender_id": "ME", "ts": True, "body": "Planning a trip to Coorg", "kind": "text"},
            {"id": "M04", "chat_id": "C01", "sender_id": "ME", "ts": float("nan"), "body": "Trip to Coorg next week", "kind": "text"},
            {"id": "M05", "chat_id": "C01", "sender_id": "ME", "ts": None, "body": "Another trip to Coorg", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)

    # Test build_timestamp_index directly
    ts_idx = build_timestamp_index(context.dataset["messages_by_id"])
    assert ts_idx["M01"] == 200
    assert ts_idx["M02"] == 100
    assert "M03" not in ts_idx  # bool True rejected
    assert "M04" not in ts_idx  # nan rejected
    assert "M05" not in ts_idx  # None rejected

    # Run full pipeline with semantically equivalent goals across these messages
    goals = [
        InterpretedGoal(title="Trip to Coorg", area="experiences", type="experience", horizon="3mo", evidence_msgs=["M01"], is_genuine=True),
        InterpretedGoal(title="Trip to Coorg", area="experiences", type="experience", horizon="3mo", evidence_msgs=["M02"], is_genuine=True),
        InterpretedGoal(title="Trip to Coorg", area="experiences", type="experience", horizon="3mo", evidence_msgs=["M03"], is_genuine=True),
    ]
    mock = MockEvidenceInterpreter(default_goals=goals)

    result = extract_goals(context, mock)

    assert result.goal_count == 1
    # Known timestamps sorted chronologically (100 -> M02, 200 -> M01), unknown/invalid placed in tail (M03)
    assert result.goals[0]["evidence"] == ["M02", "M01", "M03"]


def test_extract_goals_diagnostics_aggregation():
    """Test L: Diagnostics aggregate across all pipeline stages without resetting or leaking text."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Family trip to Coorg", "kind": "text"},
            {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 200, "body": "Planning a trip to Paris", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)
    g1 = InterpretedGoal(title="Family trip to Coorg", area="experiences", type="experience", horizon="3mo", evidence_msgs=["M01"], is_genuine=True)
    g2 = InterpretedGoal(title="Trip to Paris", area="experiences", type="experience", horizon="1yr", evidence_msgs=["M02"], is_genuine=True)
    mock = MockEvidenceInterpreter(default_goals=[g1, g2])
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag)

    # Diagnostic fields must all be integers
    for k, v in result.diagnostics.__dict__.items():
        assert isinstance(v, int), f"Field '{k}' must be int, got {type(v)}"

    assert result.diagnostics.goal_candidates_received >= 2
    assert result.diagnostics.batches_created >= 1
    assert result.diagnostics.batches_succeeded >= 1
    assert result.diagnostics.final_goals_emitted == 2

    # String representation must contain zero raw messages or personal text
    diag_str = str(result.diagnostics)
    assert "Family trip to Coorg" not in diag_str
    assert "Planning a trip to Paris" not in diag_str


def test_extract_goals_configuration_error_propagation():
    """Test M: LLMConfigurationError bubbles up immediately through extract_goals."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Planning a trip to Paris", "kind": "text"}
        ]
    }
    context = _create_test_context(chat_msgs)
    mock = MockEvidenceInterpreter(raise_error=LLMConfigurationError("LLM API authentication failed (HTTP 401)"))

    with pytest.raises(LLMConfigurationError, match="HTTP 401"):
        extract_goals(context, mock)


def test_extract_goals_input_immutability():
    """Test N: Verifies original ConversationContext, messages, and seed data are not mutated."""
    seed = {
        "goals": [
            {"title": "Brightside to 10 retainer clients by March", "area": "work", "type": "milestone", "horizon": "1yr"}
        ]
    }
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Hit 10 retainer clients by March", "kind": "text"}
        ]
    }
    context = _create_test_context(chat_msgs, seed=seed)

    context_chats_before = copy.deepcopy(context.chats)
    context_dataset_before = copy.deepcopy(context.dataset)

    cand_goal = InterpretedGoal(
        title="Hit 10 retainer clients by March",
        area="work",
        type="milestone",
        horizon="3mo",
        evidence_msgs=["M01"],
        is_genuine=True,
    )
    mock = MockEvidenceInterpreter(default_goals=[cand_goal])

    extract_goals(context, mock)

    assert context.chats == context_chats_before
    assert context.dataset == context_dataset_before


def test_extract_goals_strict_output_shape():
    """Test O: Verifies every emitted goal conforms strictly to the 5-field schema.json contract."""
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Planning a trip to Paris next summer", "kind": "text"}
        ]
    }
    context = _create_test_context(chat_msgs)
    goal = InterpretedGoal(
        title="Trip to Paris",
        area="experiences",
        type="experience",
        horizon="1yr",
        evidence_msgs=["M01"],
        is_genuine=True,
    )
    mock = MockEvidenceInterpreter(default_goals=[goal])

    result = extract_goals(context, mock)

    assert result.goal_count == 1
    g = result.goals[0]

    # Strictly 5 required fields
    assert set(g.keys()) == {"title", "area", "type", "horizon", "evidence"}
    assert isinstance(g["title"], str) and g["title"]
    assert g["area"] in {"self", "spouse", "kids", "parents", "work", "friends", "experiences", "finances", "health", "home"}
    assert g["type"] in {"habit", "milestone", "experience"}
    assert g["horizon"] in {"3mo", "1yr", "3yr", "5yr"}
    assert isinstance(g["evidence"], list) and len(g["evidence"]) >= 1
    assert all(isinstance(m, str) for m in g["evidence"])


def test_extract_goals_diagnostics_counters_accumulate_without_overwriting():
    """Verify deduplicated_candidates, seed_goals_matched, and final_goals_emitted accumulate using +=."""
    seed = {
        "goals": [
            {
                "title": "Reach 10 retainer clients by March",
                "area": "work",
                "type": "milestone",
                "horizon": "1yr",
            }
        ]
    }
    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "We want to reach 10 retainer clients by March", "kind": "text"},
            {"id": "M02", "chat_id": "C01", "sender_id": "ME", "ts": 200, "body": "We want to reach 10 retainer clients by March", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs, seed=seed)

    # Two identical goals that will deduplicate to 1, then match 1 seed goal, emitting 1 final goal
    g1 = InterpretedGoal(
        title="Reach 10 retainer clients by March",
        area="work",
        type="milestone",
        horizon="3mo",
        evidence_msgs=["M01"],
        is_genuine=True,
    )
    g2 = InterpretedGoal(
        title="Reach 10 retainer clients by March",
        area="work",
        type="milestone",
        horizon="3mo",
        evidence_msgs=["M02"],
        is_genuine=True,
    )
    mock = MockEvidenceInterpreter(default_goals=[g1, g2])

    diag = GoalExtractionDiagnostics(
        deduplicated_candidates=10,
        seed_goals_matched=20,
        final_goals_emitted=30,
        goal_candidates_received=5,
    )

    result = extract_goals(context, mock, diagnostics=diag)

    # In this run:
    # 2 candidate goals deduplicate into 1 -> contribution: +1
    # 1 deduped goal matches seed goal -> contribution: +1
    # 1 final goal emitted -> contribution: +1
    assert result.diagnostics is diag
    assert diag.deduplicated_candidates == 10 + 1
    assert diag.seed_goals_matched == 20 + 1
    assert diag.final_goals_emitted == 30 + 1
    assert diag.goal_candidates_received == 5 + 2
    assert diag.candidates_validated == 2


# ==============================================================================
# 11. Unified Execution Budget Tests
# ==============================================================================


class BudgetAwareMockInterpreter:
    """Mock interpreter supporting the 'budget' parameter in interpret_chat_batch."""

    def __init__(self, goals: Optional[List[Any]] = None, return_diag: Optional[BatchFailureDiagnostic] = None):
        self.goals = goals or []
        self.return_diag = return_diag
        self.received_budgets: List[Any] = []
        self.calls: int = 0

    def interpret_chat_batch(
        self, batch: Any, context: Optional[ConversationContext] = None, budget: Optional[Any] = None
    ) -> Tuple[List[Any], List[Any], Optional[BatchFailureDiagnostic]]:
        self.calls += 1
        self.received_budgets.append(budget)
        if self.return_diag:
            return [], [], self.return_diag
        return [], list(self.goals), None


def test_extract_goals_passes_budget_to_interpreter():
    """Verify that extract_goals passes the active execution budget to the interpreter."""
    from src.interpretation.cache import ExecutionBudget

    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal: run 5k by June", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)
    g = InterpretedGoal(
        title="Run 5k by June",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence_msgs=["M01"],
        is_genuine=True,
    )
    mock = BudgetAwareMockInterpreter(goals=[g])
    budget = ExecutionBudget(total_seconds=1650.0)

    result = extract_goals(context, mock, budget=budget)

    assert mock.calls >= 1
    assert all(b is budget for b in mock.received_budgets)
    assert result.goal_count == 1
    assert result.goals[0]["title"] == "Run 5k by June"


def test_extract_goals_exhausted_budget_diagnoses_and_degrades_gracefully():
    """Verify that when budget is exhausted, goal extraction records unattempted batches and degrades gracefully."""
    import time
    from src.interpretation.cache import ExecutionBudget

    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal: run 5k by June", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)

    timeout_diag = BatchFailureDiagnostic(
        phase="chat",
        batch_id="C01",
        attempt_count=0,
        error_type="TimeoutBudgetExceeded",
        error_message="Execution budget exhausted during pacing",
        timestamp=time.time(),
    )
    mock = BudgetAwareMockInterpreter(return_diag=timeout_diag)
    budget = ExecutionBudget(total_seconds=100.0)
    diag = GoalExtractionDiagnostics()

    result = extract_goals(context, mock, diagnostics=diag, budget=budget)

    assert result.goal_count == 0
    assert diag.batches_unattempted >= 1
    assert diag.batches_failed >= 1


def test_extract_goals_does_not_reset_existing_interpreter_session():
    """Verify extract_goals respects an existing interpreter session without resetting it."""
    from src.interpretation.interpreter import LLMEvidenceInterpreter

    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal: run 5k by June", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)

    def dummy_chat(**kwargs):
        return '{"goals": [], "open_loops": []}'

    interp = LLMEvidenceInterpreter(chat_fn=dummy_chat, cache_enabled=False)
    initial_budget = interp.start_session(max_runtime_seconds=1650.0)

    # Call extract_goals without explicit budget; it must reuse interp.active_budget without resetting
    result = extract_goals(context, interp)

    assert interp.active_budget is initial_budget
    assert interp.active_budget.total_seconds == 1650.0


def test_extract_goals_legacy_interpreter_without_budget_parameter():
    """Verify backward compatibility: legacy interpreters without 'budget' in signature still work."""
    class LegacyInterpreter:
        def __init__(self, goals):
            self.goals = goals
            self.calls = 0

        def interpret_chat_batch(self, batch, context=None):
            self.calls += 1
            return [], list(self.goals), None

    chat_msgs = {
        "C01": [
            {"id": "M01", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "Goal: run 5k by June", "kind": "text"},
        ]
    }
    context = _create_test_context(chat_msgs)
    g = InterpretedGoal(
        title="Run 5k by June",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence_msgs=["M01"],
        is_genuine=True,
    )
    legacy = LegacyInterpreter(goals=[g])

    result = extract_goals(context, legacy)

    assert legacy.calls >= 1
    assert result.goal_count == 1
    assert result.goals[0]["title"] == "Run 5k by June"

    assert result.goal_count == 1

