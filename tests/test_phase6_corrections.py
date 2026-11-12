"""Focused regression tests for Phase 6 live LLM corrections.

Covers:
1. Correction A: Goal area semantic distinction (health vs self).
2. Correction B: Goal cross-type subsumption (milestone vs supporting habit).
3. Correction C: Domestic replenishment contextual actionability.
4. Correction D: Multilingual closure verification (English title, Hindi/Hinglish conversation).
5. Correction E: Deadline detection, tier classification, and help-action precedence.
"""

import pytest
from src.goals.models import CandidateGoal
from src.goals.resolver import (
    are_semantically_equivalent_goals,
    deduplicate_goals,
    is_supporting_habit_of_milestone,
)
from src.interpretation.prompts import CHAT_SYSTEM_PROMPT
from src.loops.extractor import (
    _evaluate_fallback_actionability,
    classify_candidate_tier,
)
from src.loops.models import OpenLoopExtractionConfig
from src.loops.resolver import (
    HelpAction,
    _evaluate_closing_message,
    resolve_help_action,
    verify_full_chat_closure,
)
from src.preprocessing.context_builder import build_conversation_context


# =============================================================================
# 1. Correction A: Goal Area Semantics
# =============================================================================

class TestGoalAreaSemantics:
    """Validate prompt definitions distinguishing physical health from intellectual self."""

    def test_health_and_self_definitions_in_system_prompt(self):
        """Prompt explicitly defines health as physical fitness and self as intellectual development."""
        assert '"health": physical fitness' in CHAT_SYSTEM_PROMPT
        assert 'marathon/race running' in CHAT_SYSTEM_PROMPT
        assert 'Physical athletic milestones' in CHAT_SYSTEM_PROMPT
        assert 'belong strictly to "health", NEVER automatically assigned to "self"' in CHAT_SYSTEM_PROMPT
        assert '"self": intellectual development' in CHAT_SYSTEM_PROMPT


# =============================================================================
# 2. Correction B: Goal Cross-Type Deduplication
# =============================================================================

class TestGoalCrossTypeDeduplication:
    """Validate conservative cross-type subsumption between milestones and supporting habits."""

    def test_milestone_and_supporting_habit_sharing_evidence(self):
        """Case 1: An athletic milestone and supporting running habit sharing evidence deduplicate to milestone."""
        g_milestone = CandidateGoal(
            title="Run the half marathon in January",
            area="health",
            type="milestone",
            horizon="3mo",
            evidence=["M026", "M027"],
            confidence=0.95,
        )
        g_habit = CandidateGoal(
            title="Run four times a week",
            area="health",
            type="habit",
            horizon="3mo",
            evidence=["M027"],
            confidence=0.90,
        )

        assert is_supporting_habit_of_milestone(g_milestone, g_habit) is True
        assert are_semantically_equivalent_goals(g_milestone, g_habit) is True

        merged = deduplicate_goals([g_milestone, g_habit])
        assert len(merged) == 1
        assert merged[0].type == "milestone"
        assert merged[0].title == "Run the half marathon in January"
        assert merged[0].area == "health"
        assert set(merged[0].evidence) == {"M026", "M027"}

        # Commutative order test
        merged_rev = deduplicate_goals([g_habit, g_milestone])
        assert len(merged_rev) == 1
        assert merged_rev[0].type == "milestone"
        assert merged_rev[0].title == "Run the half marathon in January"

    def test_two_genuinely_independent_goals_different_types(self):
        """Case 2: Two independent goals of different types (even in same chat/evidence) are preserved."""
        g_work_milestone = CandidateGoal(
            title="Achieve 10 retainer clients by March",
            area="work",
            type="milestone",
            horizon="1yr",
            evidence=["M019"],
            confidence=0.90,
        )
        g_running_habit = CandidateGoal(
            title="Run four times a week",
            area="health",
            type="habit",
            horizon="3mo",
            evidence=["M019"],  # Shared message
            confidence=0.90,
        )

        assert is_supporting_habit_of_milestone(g_work_milestone, g_running_habit) is False
        assert are_semantically_equivalent_goals(g_work_milestone, g_running_habit) is False

        merged = deduplicate_goals([g_work_milestone, g_running_habit])
        assert len(merged) == 2
        titles = {g.title for g in merged}
        assert "Achieve 10 retainer clients by March" in titles
        assert "Run four times a week" in titles

    def test_habit_without_related_milestone(self):
        """Case 3: An intentional habit without any related milestone is preserved as a habit."""
        g_meditation = CandidateGoal(
            title="Daily meditation for 15 minutes",
            area="self",
            type="habit",
            horizon="3mo",
            evidence=["M030"],
            confidence=0.85,
        )

        merged = deduplicate_goals([g_meditation])
        assert len(merged) == 1
        assert merged[0].type == "habit"
        assert merged[0].title == "Daily meditation for 15 minutes"

    def test_goals_with_no_overlapping_evidence(self):
        """Case 4: A milestone and a habit with no overlapping evidence are strictly preserved independently."""
        g_milestone = CandidateGoal(
            title="Run the half marathon in January",
            area="health",
            type="milestone",
            horizon="3mo",
            evidence=["M026"],
            confidence=0.95,
        )
        g_habit = CandidateGoal(
            title="Run four times a week",
            area="health",
            type="habit",
            horizon="3mo",
            evidence=["M050"],  # Disjoint evidence
            confidence=0.90,
        )

        assert is_supporting_habit_of_milestone(g_milestone, g_habit) is False
        assert are_semantically_equivalent_goals(g_milestone, g_habit) is False

        merged = deduplicate_goals([g_milestone, g_habit])
        assert len(merged) == 2
        types = {g.type for g in merged}
        assert types == {"milestone", "habit"}


# =============================================================================
# 3. Correction C: Contextual Domestic Replenishment
# =============================================================================

class TestDomesticReplenishmentActionability:
    """Validate prompt rules for contextual replenishment actionability."""

    def test_prompt_contains_domestic_replenishment_rule(self):
        """Prompt instructs that domestic replenishment statements accepted by ME are genuine."""
        assert "DOMESTIC REPLENISHMENT CONTEXT" in CHAT_SYSTEM_PROMPT
        assert "Gas cylinder khatam hone wala hai" in CHAT_SYSTEM_PROMPT
        assert "is_genuine: true" in CHAT_SYSTEM_PROMPT
        assert "Preserve conservative handling (is_genuine: false) for general observations" in CHAT_SYSTEM_PROMPT


# =============================================================================
# 4. Correction D: Multilingual Closure Verification
# =============================================================================

class TestMultilingualClosureVerification:
    """Validate closure verification across English-translated titles and Hindi/Hinglish messages."""

    def _make_dummy_context(self, messages, chat_id="C02", is_group=False):
        chats = [{
            "chat_id": chat_id,
            "is_group": is_group,
            "participants": ["ME", "P02"] if not is_group else ["ME", "P02", "P03"],
            "messages": messages,
        }]
        participants = {
            "ME": {"id": "ME", "sender_names": ["Me"], "direct_chat_names": ["Me"]},
            "P02": {"id": "P02", "sender_names": ["User_P02"], "direct_chat_names": ["User_P02"]},
            "P03": {"id": "P03", "sender_names": ["User_P03"], "direct_chat_names": ["User_P03"]},
        }
        raw = {
            "messages": messages,
            "chats": chats,
            "chats_by_id": {c["chat_id"]: c for c in chats},
            "messages_by_chat": {chat_id: messages},
            "participants": participants,
            "messages_by_id": {m["id"]: m for m in messages},
        }
        return build_conversation_context(raw)

    def test_m006_m007_realistic_cross_lingual_closure(self):
        """Realistic M006/M007: English title verified against Hindi response using source message grounding."""
        m006 = {
            "id": "M006",
            "chat_id": "C02",
            "sender_id": "P02",
            "ts": 1792499700,
            "body": "Didi kal guests ke liye kya banana hai? Paneer ya chicken?",
        }
        m007 = {
            "id": "M007",
            "chat_id": "C02",
            "sender_id": "ME",
            "ts": 1792501200,
            "body": "Paneer butter masala aur dal makhani. 8 log honge",
        }
        messages = [m006, m007]
        context = self._make_dummy_context(messages, chat_id="C02")

        status, closed_by = verify_full_chat_closure(
            source_msg=m006,
            chat_id="C02",
            owner="ME",
            loop_type="question",
            title="Ask what to cook for tomorrow's guests",
            context=context,
            raw_status="done",
            raw_closed_by="M007",
        )
        assert status == "done"
        assert closed_by == "M007"

    def test_unrelated_subsequent_message_does_not_close(self):
        """An adjacent or subsequent message about an unrelated topic does NOT close the question."""
        m006 = {
            "id": "M006",
            "chat_id": "C02",
            "sender_id": "P02",
            "ts": 1792499700,
            "body": "Didi kal guests ke liye kya banana hai? Paneer ya chicken?",
        }
        m_unrelated = {
            "id": "M007",
            "chat_id": "C02",
            "sender_id": "ME",
            "ts": 1792501200,
            "body": "Office weather is quite cold today",
        }
        messages = [m006, m_unrelated]
        context = self._make_dummy_context(messages, chat_id="C02")

        status, closed_by = verify_full_chat_closure(
            source_msg=m006,
            chat_id="C02",
            owner="ME",
            loop_type="question",
            title="Ask what to cook for tomorrow's guests",
            context=context,
            raw_status="done",
            raw_closed_by="M007",
        )
        assert status == "open"
        assert closed_by is None

    def test_unauthorised_actor_response_does_not_close(self):
        """In a group chat, a response from an unauthorized third party does not close the obligation."""
        m_q = {
            "id": "M100",
            "chat_id": "C99",
            "sender_id": "P02",
            "ts": 1792499700,
            "body": "Arjun can you confirm the budget by tonight?",
        }
        m_unauth = {
            "id": "M101",
            "chat_id": "C99",
            "sender_id": "P03",  # Unrelated third party, not ME
            "ts": 1792501200,
            "body": "Budget numbers are ready on sheet",
        }
        messages = [m_q, m_unauth]
        context = self._make_dummy_context(messages, chat_id="C99", is_group=True)

        status, closed_by = verify_full_chat_closure(
            source_msg=m_q,
            chat_id="C99",
            owner="ME",
            loop_type="question",
            title="Confirm budget proposal",
            context=context,
            raw_status="done",
            raw_closed_by="M101",
        )
        assert status == "open"
        assert closed_by is None

    def test_explicitly_unresolved_hedging_does_not_close(self):
        """Hedging or postponement responses ('Not sure', 'Pata nahi') do not close the loop."""
        m_q = {
            "id": "M100",
            "chat_id": "C02",
            "sender_id": "P02",
            "ts": 1792499700,
            "body": "Didi kal guests ke liye kya banana hai?",
        }
        m_hedge = {
            "id": "M101",
            "chat_id": "C02",
            "sender_id": "ME",
            "ts": 1792501200,
            "body": "Abhi pata nahi didi, let you know later",
        }
        messages = [m_q, m_hedge]
        context = self._make_dummy_context(messages, chat_id="C02")

        status, closed_by = verify_full_chat_closure(
            source_msg=m_q,
            chat_id="C02",
            owner="ME",
            loop_type="question",
            title="Ask what to cook for tomorrow's guests",
            context=context,
            raw_status="done",
            raw_closed_by="M101",
        )
        assert status == "open"
        assert closed_by is None


# =============================================================================
# 5. Correction E: Deadline Detection & Help Precedence
# =============================================================================

class TestDeadlineDetectionAndHelpPrecedence:
    """Validate deadline regex expansion, tier routing, and reminder precedence over generic assist."""

    @pytest.mark.parametrize("phrase", ["due Friday", "due tomorrow", "due Monday", "due today"])
    def test_deadline_actionability_natural_expressions(self, phrase):
        """Natural deadline expressions are recognized as actionable."""
        msg = {
            "id": "M999",
            "chat_id": "C04",
            "sender_id": "ME",
            "ts": 1792472400,
            "body": f"Pitch deck for Halcyon is {phrase}, I'm on it.",
        }
        cand = {"message_id": "M999", "chat_id": "C04", "category": "deadline"}
        chat = {"chat_id": "C04", "is_group": True, "participants": ["ME", "P05", "P06"], "messages": [msg]}
        raw = {
            "messages": [msg],
            "chats": [chat],
            "chats_by_id": {"C04": chat},
            "participants": {
                "ME": {"id": "ME", "sender_names": ["Me"], "direct_chat_names": ["Me"]},
                "P05": {"id": "P05", "sender_names": ["User5"], "direct_chat_names": ["User5"]},
                "P06": {"id": "P06", "sender_names": ["User6"], "direct_chat_names": ["User6"]},
            },
            "messages_by_id": {msg["id"]: msg},
        }
        context = build_conversation_context(raw)

        is_act, cat, owner, conf = _evaluate_fallback_actionability(
            cand=cand, msg=msg, context=context, chat=chat
        )
        assert is_act is True
        assert cat == "deadline"
        assert conf >= 0.80

    def test_help_action_future_deadline_takes_precedence_over_generic_assist(self):
        """Explicit future deadline forces action='remind' even when LLM passes suggested_action='assist'."""
        msg = {
            "id": "M018",
            "chat_id": "C04",
            "sender_id": "ME",
            "ts": 1792472400,
            "body": "Pitch deck for Halcyon is due Friday, I'm on it. Neha let me think about Zest",
        }
        config = OpenLoopExtractionConfig()

        act = resolve_help_action(
            loop_title="Prepare Halcyon pitch deck due Friday",
            owner="ME",
            source_msg=msg,
            context=None,
            config=config,
            suggested_action="assist",  # Generic LLM suggestion
        )
        assert act.action == "remind"
        assert act.remind_at is not None

    def test_help_action_past_deadline_does_not_force_remind(self):
        """Past deadline references ('due yesterday') do not force remind."""
        msg = {
            "id": "M998",
            "chat_id": "C01",
            "sender_id": "ME",
            "ts": 1792472400,
            "body": "The report was due yesterday. Researching updated options now.",
        }
        config = OpenLoopExtractionConfig()

        act = resolve_help_action(
            loop_title="Research options for updated report",
            owner="ME",
            source_msg=msg,
            context=None,
            config=config,
            suggested_action="assist",
        )
        assert act.action == "assist"

    def test_help_action_counterpart_owned_task_strictly_leave(self):
        """Tasks owned by someone other than ME strictly resolve to 'leave'."""
        msg = {
            "id": "M017",
            "chat_id": "C04",
            "sender_id": "P05",
            "ts": 1792472400,
            "body": "I'll send Orbit their October invoice due Friday",
        }
        config = OpenLoopExtractionConfig()

        act = resolve_help_action(
            loop_title="Send Orbit October invoice due Friday",
            owner="P05",  # Owned by counterpart
            source_msg=msg,
            context=None,
            config=config,
            suggested_action="assist",
        )
        assert act.action == "leave"
        assert "P05 owns this task" in (act.reason or "")
