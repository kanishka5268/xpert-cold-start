"""Focused regression tests for Phase 6.2 corrections.

Covers:
1. Correction A: Goal horizon prompt/parser resilience:
   - Valid horizon unchanged.
   - Null horizon with explicit near-term target.
   - Null horizon with explicit long-term target.
   - Null horizon without a usable target date.
   - Invalid horizon enum.
   - Invalid evidence still rejected.
   - Non-genuine goals still rejected.
   - Existing parser return contract preserved.
2. Correction B: Supporting habit and milestone cross-type subsumption:
   - Supporting habit and milestone in same conversation with disjoint evidence.
   - Genuinely independent habit and milestone preserved.
3. Correction C: Commercial decision question help precedence:
   - Commercial decision questions enforce 'remind' over generic 'assist'.
   - Executable preparation / assist tasks preserve 'assist'.
   - Counterpart-owned tasks strictly preserve 'leave'.
"""

import pytest
from src.goals.models import CandidateGoal
from src.goals.resolver import (
    are_semantically_equivalent_goals,
    deduplicate_goals,
    is_supporting_habit_of_milestone,
)
from src.interpretation.batcher import EvidenceBatch
from src.interpretation.parser import (
    ChatParserDiagnostics,
    normalize_goal_horizon,
    parse_chat_interpretation_response,
)
from src.interpretation.prompts import CHAT_SYSTEM_PROMPT
from src.loops.models import OpenLoopExtractionConfig
from src.loops.resolver import (
    DECISION_QUESTION_PATTERN,
    HelpAction,
    resolve_help_action,
)
from src.preprocessing.context_builder import build_conversation_context


# =============================================================================
# 1. Correction A: Goal Horizon Normalization and Parser Resilience
# =============================================================================

class TestGoalHorizonPromptAndParser:
    """Validate prompt rules and deterministic parser normalization for goal horizons."""

    def test_prompt_instructs_mandatory_horizon_and_no_null(self):
        """Prompt explicitly defines mandatory horizon and instructs never to return null for horizon."""
        assert "MANDATORY for every genuine goal" in CHAT_SYSTEM_PROMPT
        assert "NEVER return null for horizon" in CHAT_SYSTEM_PROMPT
        assert '"3mo": Near-term target within 1-3 months' in CHAT_SYSTEM_PROMPT
        assert '"1yr": Targets within 4-12 months' in CHAT_SYSTEM_PROMPT

    def test_valid_horizon_unchanged(self):
        """Valid horizon enums are returned unchanged with was_normalized=False."""
        for h in ["3mo", "1yr", "3yr", "5yr"]:
            res, was_norm = normalize_goal_horizon(h, title="Goal", goal_type="milestone")
            assert res == h
            assert was_norm is False

    def test_null_horizon_with_explicit_near_term_target(self):
        """Null horizon with explicit near-term target month (Jan from Oct anchor) infers '3mo'."""
        oct_ts = 1792459800  # Oct 2026
        res, was_norm = normalize_goal_horizon(
            None,
            title="Half marathon in Jan",
            goal_type="milestone",
            anchor_timestamp=oct_ts,
        )
        assert res == "3mo"
        assert was_norm is True

        # December target from October
        res_dec, was_norm_dec = normalize_goal_horizon(
            None,
            title="Family trip to Coorg in December",
            goal_type="experience",
            anchor_timestamp=oct_ts,
        )
        assert res_dec == "3mo"
        assert was_norm_dec is True

    def test_null_horizon_with_explicit_long_term_target(self):
        """Null horizon with explicit long-term target month (March from Oct anchor) or multi-year infers '1yr' or '5yr'."""
        oct_ts = 1792459800  # Oct 2026
        res_mar, was_norm_mar = normalize_goal_horizon(
            None,
            title="Brightside to 10 retainer clients by March",
            goal_type="milestone",
            anchor_timestamp=oct_ts,
        )
        assert res_mar == "1yr"
        assert was_norm_mar is True

        res_5yr, was_norm_5yr = normalize_goal_horizon(
            None,
            title="Retire comfortably in 5 years",
            goal_type="milestone",
        )
        assert res_5yr == "5yr"
        assert was_norm_5yr is True

    def test_null_horizon_without_usable_target_date(self):
        """Null horizon without explicit date uses type-informed fallback: habit/experience -> 3mo, milestone -> 1yr."""
        res_habit, was_norm_habit = normalize_goal_horizon(None, title="Daily morning meditation", goal_type="habit")
        assert res_habit == "3mo"
        assert was_norm_habit is True

        res_exp, was_norm_exp = normalize_goal_horizon(None, title="Explore new city cafes", goal_type="experience")
        assert res_exp == "3mo"
        assert was_norm_exp is True

        res_mile, was_norm_mile = normalize_goal_horizon(None, title="Launch enterprise platform", goal_type="milestone")
        assert res_mile == "1yr"
        assert was_norm_mile is True

    def test_invalid_horizon_enum_normalized(self):
        """Invalid enum strings like '6mo' or '2yr' are normalized deterministically."""
        res_6mo, was_norm_6mo = normalize_goal_horizon("6mo", title="Goal", goal_type="milestone")
        assert res_6mo == "1yr"
        assert was_norm_6mo is True

        res_2yr, was_norm_2yr = normalize_goal_horizon("2yr", title="Goal", goal_type="milestone")
        assert res_2yr == "3yr"
        assert was_norm_2yr is True

    def test_parser_normalizes_null_horizon_and_preserves_goals(self):
        """Parser preserves goals that have horizon: null, normalizing them cleanly and incrementing counter."""
        m026 = {"id": "M026", "chat_id": "C06", "sender_id": "P08", "ts": 1792459800, "body": "Ran 14k today! You still on for the half marathon in Jan?"}
        m027 = {"id": "M027", "chat_id": "C06", "sender_id": "ME", "ts": 1792461000, "body": "Yes!! Doing 4 runs a week. I'll send you my training plan"}
        messages = [m026, m027]
        batch = EvidenceBatch(
            chat_id="C06",
            messages=messages,
            candidates=[],
            candidate_message_ids=["M026", "M027"],
            categories=["goal"],
        )

        raw_json = (
            '{"open_loops":[],"goals":['
            '{"title":"Half marathon in Jan","area":"health","type":"milestone","horizon":null,"evidence_msgs":["M026"],"is_genuine":true,"confidence":0.9},'
            '{"title":"Run 4 times a week","area":"health","type":"habit","horizon":null,"evidence_msgs":["M027"],"is_genuine":true,"confidence":0.9}'
            ']}'
        )

        diag = ChatParserDiagnostics()
        loops, goals = parse_chat_interpretation_response(
            raw_text=raw_json,
            batch=batch,
            context=None,
            diagnostics=diag,
        )

        assert len(goals) == 2
        assert goals[0].title == "Half marathon in Jan"
        assert goals[0].area == "health"
        assert goals[0].horizon == "3mo"
        assert goals[0].evidence_msgs == ["M026"]

        assert goals[1].title == "Run 4 times a week"
        assert goals[1].area == "health"
        assert goals[1].horizon == "3mo"
        assert goals[1].evidence_msgs == ["M027"]

        assert diag.goals_horizon_normalized == 2
        assert diag.goals_accepted == 2

    def test_parser_still_rejects_non_genuine_goals(self):
        """Goals marked is_genuine: false are strictly discarded."""
        m009 = {"id": "M009", "chat_id": "C02", "sender_id": "ME", "ts": 1000, "body": "Book gas cylinder"}
        batch = EvidenceBatch(
            chat_id="C02",
            messages=[m009],
            candidates=[],
            candidate_message_ids=["M009"],
            categories=["goal"],
        )

        raw_json = (
            '{"open_loops":[],"goals":['
            '{"title":"Buy groceries","area":"home","type":"milestone","horizon":null,"evidence_msgs":["M009"],"is_genuine":false,"confidence":0.9}'
            ']}'
        )

        diag = ChatParserDiagnostics()
        loops, goals = parse_chat_interpretation_response(
            raw_text=raw_json,
            batch=batch,
            diagnostics=diag,
        )
        assert len(goals) == 0
        assert diag.goals_rejected_not_genuine == 1

    def test_parser_still_rejects_ungrounded_evidence(self):
        """Goals referencing invented message IDs not in batch are strictly discarded."""
        m009 = {"id": "M009", "chat_id": "C02", "sender_id": "ME", "ts": 1000, "body": "Book gas cylinder"}
        batch = EvidenceBatch(
            chat_id="C02",
            messages=[m009],
            candidates=[],
            candidate_message_ids=["M009"],
            categories=["goal"],
        )

        raw_json = (
            '{"open_loops":[],"goals":['
            '{"title":"Learn Spanish","area":"self","type":"habit","horizon":null,"evidence_msgs":["M999"],"is_genuine":true,"confidence":0.9}'
            ']}'
        )

        diag = ChatParserDiagnostics()
        loops, goals = parse_chat_interpretation_response(
            raw_text=raw_json,
            batch=batch,
            diagnostics=diag,
        )
        assert len(goals) == 0
        assert diag.goals_invalid_or_ungrounded_evidence == 1


# =============================================================================
# 2. Correction B: Supporting Habit and Milestone Cross-Type Subsumption
# =============================================================================

class TestSupportingHabitSubsumption:
    """Validate subsumption of supporting routines into parent milestones within the same conversation."""

    def test_marathon_and_running_habit_subsumption_disjoint_evidence(self):
        """Marathon milestone (M026) and running habit (M027) in same chat C06 subsume cleanly."""
        milestone = CandidateGoal(
            title="Half marathon in Jan",
            area="health",
            type="milestone",
            horizon="3mo",
            evidence=["M026"],
            confidence=0.90,
            source_chats=["C06"],
        )
        habit = CandidateGoal(
            title="Run 4 times a week",
            area="health",
            type="habit",
            horizon="3mo",
            evidence=["M027"],
            confidence=0.90,
            source_chats=["C06"],
        )

        assert is_supporting_habit_of_milestone(milestone, habit) is True
        assert are_semantically_equivalent_goals(milestone, habit) is True

        ts_index = {"M026": 1792459800, "M027": 1792461000}
        deduped = deduplicate_goals([milestone, habit], timestamp_index=ts_index)

        assert len(deduped) == 1
        assert deduped[0].title == "Half marathon in Jan"
        assert deduped[0].type == "milestone"
        assert deduped[0].area == "health"
        assert deduped[0].evidence == ["M026", "M027"]

    def test_independent_habit_in_same_chat_not_subsumed(self):
        """Meditation habit and marathon milestone in the same chat are genuinely independent and preserved."""
        milestone = CandidateGoal(
            title="Half marathon in Jan",
            area="health",
            type="milestone",
            horizon="3mo",
            evidence=["M026"],
            confidence=0.90,
            source_chats=["C06"],
        )
        independent_habit = CandidateGoal(
            title="Meditate 10 minutes every morning",
            area="self",
            type="habit",
            horizon="3mo",
            evidence=["M027"],
            confidence=0.90,
            source_chats=["C06"],
        )

        assert is_supporting_habit_of_milestone(milestone, independent_habit) is False
        assert are_semantically_equivalent_goals(milestone, independent_habit) is False

        ts_index = {"M026": 1792459800, "M027": 1792461000}
        deduped = deduplicate_goals([milestone, independent_habit], timestamp_index=ts_index)
        assert len(deduped) == 2

    def test_goals_in_different_chats_with_disjoint_evidence_not_subsumed(self):
        """Running habit in C01 and marathon milestone in C06 without shared evidence are preserved as independent."""
        milestone = CandidateGoal(
            title="Half marathon in Jan",
            area="health",
            type="milestone",
            horizon="3mo",
            evidence=["M026"],
            confidence=0.90,
            source_chats=["C06"],
        )
        diff_chat_habit = CandidateGoal(
            title="Run 4 times a week",
            area="health",
            type="habit",
            horizon="3mo",
            evidence=["M005"],
            confidence=0.90,
            source_chats=["C01"],
        )

        assert is_supporting_habit_of_milestone(milestone, diff_chat_habit) is False
        deduped = deduplicate_goals([milestone, diff_chat_habit])
        assert len(deduped) == 2


# =============================================================================
# 3. Correction C: Commercial Decision Help Precedence
# =============================================================================

class TestCommercialDecisionHelpPrecedence:
    """Validate that commercial and personal decision questions enforce 'remind' over generic 'assist'."""

    def _make_dummy_context(self):
        raw = {
            "messages": [],
            "chats": [{"chat_id": "C04", "is_group": True, "participants": ["ME", "P05"]}],
            "chats_by_id": {"C04": {"chat_id": "C04", "is_group": True, "participants": ["ME", "P05"]}},
            "participants": {"ME": {"id": "ME"}, "P05": {"id": "P05"}},
        }
        return build_conversation_context(raw)

    def test_zest_proposal_decision_enforces_remind(self):
        """Commercial decision question 'Decide whether to accept Zest offer' enforces 'remind'."""
        msg = {
            "id": "M016",
            "chat_id": "C04",
            "sender_id": "ME",
            "ts": 1792472400,
            "body": "Zest wants us at 60k/month instead of 80. Should we take it",
        }
        config = OpenLoopExtractionConfig()
        context = self._make_dummy_context()

        # LLM suggests 'assist', but decision question pattern overrides to 'remind'
        act = resolve_help_action(
            loop_title="Decide whether to accept Zest's 60k/month offer",
            owner="ME",
            source_msg=msg,
            context=context,
            config=config,
            suggested_action="assist",
            anchor_ts=msg["ts"],
            chat_id="C04",
        )
        assert act.action == "remind"
        assert act.remind_at is not None

    def test_decision_question_pattern_matching(self):
        """Pattern matches various personal and commercial decision formulations."""
        assert DECISION_QUESTION_PATTERN.search("Decide whether to accept Zest offer")
        assert DECISION_QUESTION_PATTERN.search("Decision on Zest proposal (60k/month)")
        assert DECISION_QUESTION_PATTERN.search("Should we take the 60k offer")
        assert DECISION_QUESTION_PATTERN.search("Whether to hire the candidate")
        assert DECISION_QUESTION_PATTERN.search("Should I sign the contract")

    def test_executable_assist_tasks_preserve_assist(self):
        """Non-decision actionable tasks preserve 'assist' when suggested."""
        msg = {
            "id": "M002",
            "chat_id": "C01",
            "sender_id": "ME",
            "ts": 1792472400,
            "body": "I'll book the tickets and the homestay this week, promise",
        }
        config = OpenLoopExtractionConfig()
        context = self._make_dummy_context()

        act = resolve_help_action(
            loop_title="Book tickets and homestay for Coorg trip",
            owner="ME",
            source_msg=msg,
            context=context,
            config=config,
            suggested_action="assist",
            anchor_ts=msg["ts"],
            chat_id="C01",
        )
        assert act.action == "assist"
        assert act.needs_yes is True

    def test_counterpart_owned_decision_preserves_leave(self):
        """Even if the task is a decision, if owned by a counterpart, action is strictly 'leave'."""
        msg = {
            "id": "M016",
            "chat_id": "C04",
            "sender_id": "P05",
            "ts": 1792472400,
            "body": "Should we take Zest offer?",
        }
        config = OpenLoopExtractionConfig()
        context = self._make_dummy_context()

        act = resolve_help_action(
            loop_title="Decide whether to accept Zest offer",
            owner="P05",
            source_msg=msg,
            context=context,
            config=config,
            suggested_action="assist",
            anchor_ts=msg["ts"],
            chat_id="C04",
        )
        assert act.action == "leave"
        assert "P05 owns this task" in act.reason
