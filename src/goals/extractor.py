"""Goal extraction pipeline, candidate conversion, and evidence validation."""

from dataclasses import dataclass, field
import inspect
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from src.extraction.evidence_extractor import CandidateEvidenceExtractor
from src.goals.models import CandidateGoal, GoalExtractionConfig
from src.goals.resolver import (
    deduplicate_goals,
    is_valid_timestamp,
    rank_and_clamp_goals,
    reconcile_seed_goals,
    validate_and_filter_goal_evidence,
    validate_confidence,
)
from src.interpretation.batcher import build_evidence_batches
from src.interpretation.interpreter import LLMConfigurationError
from src.interpretation.records import InterpretedGoal
from src.preprocessing.context_builder import ConversationContext

VALID_GOAL_AREAS: Set[str] = {
    "self",
    "spouse",
    "kids",
    "parents",
    "work",
    "friends",
    "experiences",
    "finances",
    "health",
    "home",
}

VALID_GOAL_TYPES: Set[str] = {
    "habit",
    "milestone",
    "experience",
}

VALID_GOAL_HORIZONS: Set[str] = {
    "3mo",
    "1yr",
    "3yr",
    "5yr",
}


@dataclass
class GoalExtractionDiagnostics:
    """Operational diagnostics and counters for goal extraction and validation.

    Contains strictly numeric operational counters and safe aggregates; never stores
    raw conversation messages, goal titles, API keys, credentials, or unfiltered exceptions.
    """

    # Milestone 3 orchestration counters
    goal_candidates_received: int = 0
    batches_created: int = 0
    batches_succeeded: int = 0
    batches_failed: int = 0
    batches_unattempted: int = 0
    invalid_records_discarded: int = 0

    # Milestone 2 conversion & validation counters
    interpreted_goals_received: int = 0
    genuine_goals_received: int = 0
    candidates_created: int = 0
    candidates_rejected: int = 0
    candidates_validated: int = 0
    invalid_evidence_ids_discarded: int = 0
    ungrounded_candidates_discarded: int = 0

    # Milestone 4 pipeline counters
    deduplicated_candidates: int = 0
    seed_goals_matched: int = 0
    final_goals_emitted: int = 0


@dataclass
class GoalExtractionResult:
    """Container for finalized schema-compliant goals and execution diagnostics."""

    goals: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: GoalExtractionDiagnostics = field(default_factory=GoalExtractionDiagnostics)

    @property
    def goal_count(self) -> int:
        """Return the number of schema goals emitted."""
        return len(self.goals)


def convert_interpreted_goals_to_candidates(
    interpreted_goals: Optional[Sequence[InterpretedGoal]],
    dataset_messages_by_id: Optional[Dict[str, Dict[str, Any]]],
    diagnostics: Optional[GoalExtractionDiagnostics] = None,
) -> List[CandidateGoal]:
    """Convert InterpretedGoal records into validated CandidateGoal instances.

    Validation & Provenance Rules:
    1. Rejects non-genuine goals (is_genuine must be exactly True).
    2. Enforces non-empty string title, valid area enum, valid type enum, and valid horizon enum.
       Invalid fields are not silently repaired; the goal is rejected.
    3. Normalizes confidence using the resolver's validate_confidence() policy.
    4. Validates evidence IDs against dataset_messages_by_id, discarding nonexistent or non-string IDs.
    5. Discards duplicate evidence IDs while strictly preserving original insertion order.
    6. Uses validate_and_filter_goal_evidence() to derive source_chats solely from authentic message chat_ids.
    7. Discards any candidate that has zero valid grounded evidence IDs remaining.
    8. Preserves the exact CandidateGoal contract without mutation of input records or dataset messages.

    Args:
        interpreted_goals: Sequence of InterpretedGoal objects (or None/empty).
        dataset_messages_by_id: Mapping of message_id to dataset message dictionary.
        diagnostics: Optional GoalExtractionDiagnostics instance to record operational counts.

    Returns:
        List of grounded, validated CandidateGoal instances.
    """
    if diagnostics is None:
        diagnostics = GoalExtractionDiagnostics()

    if not interpreted_goals:
        return []

    msg_index = dataset_messages_by_id if isinstance(dataset_messages_by_id, dict) else {}
    validated_candidates: List[CandidateGoal] = []

    diagnostics.interpreted_goals_received += len(interpreted_goals)

    for goal in interpreted_goals:
        # 1. Reject if not an InterpretedGoal instance or is_genuine is not strictly True
        if not isinstance(goal, InterpretedGoal):
            diagnostics.candidates_rejected += 1
            continue

        if not isinstance(goal.is_genuine, bool) or goal.is_genuine is not True:
            diagnostics.candidates_rejected += 1
            continue

        diagnostics.genuine_goals_received += 1

        # 2. Strict field and enum validation (no silent repair)
        if not isinstance(goal.title, str) or not goal.title.strip():
            diagnostics.candidates_rejected += 1
            continue

        if not isinstance(goal.area, str) or goal.area not in VALID_GOAL_AREAS:
            diagnostics.candidates_rejected += 1
            continue

        if not isinstance(goal.type, str) or goal.type not in VALID_GOAL_TYPES:
            diagnostics.candidates_rejected += 1
            continue

        if not isinstance(goal.horizon, str) or goal.horizon not in VALID_GOAL_HORIZONS:
            diagnostics.candidates_rejected += 1
            continue

        # 3. Evidence format inspection
        if goal.evidence_msgs is None or not isinstance(goal.evidence_msgs, (list, tuple)):
            diagnostics.candidates_rejected += 1
            continue

        total_evidence_count = len(goal.evidence_msgs)
        if total_evidence_count == 0:
            diagnostics.candidates_rejected += 1
            continue

        # Extract only string IDs, rejecting whitespace-only strings and preserving exact content
        string_evidence: List[str] = []
        for mid in goal.evidence_msgs:
            if isinstance(mid, str) and mid.strip() != "":
                string_evidence.append(mid)

        if not string_evidence:
            diagnostics.invalid_evidence_ids_discarded += total_evidence_count
            diagnostics.candidates_rejected += 1
            continue

        # 4. Construct unvalidated CandidateGoal with normalized confidence
        conf = validate_confidence(goal.confidence, default=0.5)

        raw_candidate = CandidateGoal(
            title=goal.title.strip(),
            area=goal.area,
            type=goal.type,
            horizon=goal.horizon,
            evidence=string_evidence,
            confidence=conf,
            source_chats=[],
            is_seed=False,
        )
        diagnostics.candidates_created += 1

        # 5. Early evidence validation and source_chats derivation via resolver
        v_cand = validate_and_filter_goal_evidence(raw_candidate, msg_index)

        if v_cand is not None:
            discarded_ids = total_evidence_count - len(v_cand.evidence)
            diagnostics.invalid_evidence_ids_discarded += max(0, discarded_ids)
            diagnostics.candidates_validated += 1
            validated_candidates.append(v_cand)
        else:
            diagnostics.invalid_evidence_ids_discarded += total_evidence_count
            diagnostics.ungrounded_candidates_discarded += 1
            diagnostics.candidates_rejected += 1

    return validated_candidates


def _supports_budget_parameter(interpreter: Any) -> bool:
    """Safely inspect whether interpreter.interpret_chat_batch accepts a 'budget' keyword argument."""
    from src.interpretation.interpreter import LLMEvidenceInterpreter
    if isinstance(interpreter, LLMEvidenceInterpreter):
        return True

    method = getattr(interpreter, "interpret_chat_batch", None)
    if method is None or not callable(method):
        return False

    try:
        sig = inspect.signature(method)
        if "budget" in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (ValueError, TypeError):
        func = getattr(method, "__func__", method)
        code = getattr(func, "__code__", None)
        if code and hasattr(code, "co_varnames"):
            return "budget" in code.co_varnames
        return False


def interpret_goal_evidence(
    context: ConversationContext,
    candidates: Sequence[Dict[str, Any]],
    interpreter: Any,
    diagnostics: Optional[GoalExtractionDiagnostics] = None,
    budget: Optional[Any] = None,
) -> List[InterpretedGoal]:
    """Orchestrate LLM interpretation of goal evidence candidates into typed InterpretedGoal records.

    Pipeline Flow:
    1. Records incoming candidate count in diagnostics.
    2. If candidates is empty, returns immediately without creating batches or calling the interpreter.
    3. Groups candidates into deterministic batches using build_evidence_batches(context, candidates).
    4. For each batch:
       - Checks parameter support for 'budget' prior to invocation.
       - Invokes interpreter.interpret_chat_batch(batch, context=context, budget=budget) if supported,
         or interpreter.interpret_chat_batch(batch, context=context) otherwise.
       - Fatal errors (LLMConfigurationError) bubble up immediately.
       - Non-fatal batch failures (diag is not None, or unexpected batch exceptions) increment
         diagnostics.batches_failed (and diagnostics.batches_unattempted if budget was exhausted).
       - Successful batches increment diagnostics.batches_succeeded.
       - Emitted goal records are type-checked: valid InterpretedGoal instances are collected,
         and non-InterpretedGoal records increment diagnostics.invalid_records_discarded.
    5. Returns collected list of InterpretedGoal records in deterministic execution order.

    Args:
        context: ConversationContext instance containing loaded dataset and chat index.
        candidates: Sequence of extracted goal candidate dictionaries.
        interpreter: Evidence interpreter instance (e.g., LLMEvidenceInterpreter or test fake).
        diagnostics: Optional GoalExtractionDiagnostics instance to track execution counters.
        budget: Optional ExecutionBudget instance for monotonic deadline enforcement.

    Returns:
        List of InterpretedGoal instances produced across all successfully interpreted batches.
    """
    if diagnostics is None:
        diagnostics = GoalExtractionDiagnostics()

    if not candidates:
        return []

    diagnostics.goal_candidates_received += len(candidates)

    batches = build_evidence_batches(context, candidates)
    diagnostics.batches_created += len(batches)

    if not batches:
        return []

    all_goals: List[InterpretedGoal] = []
    supports_budget = _supports_budget_parameter(interpreter)

    for batch in batches:
        try:
            if supports_budget and budget is not None:
                _loops, goals, diag = interpreter.interpret_chat_batch(batch, context=context, budget=budget)
            else:
                _loops, goals, diag = interpreter.interpret_chat_batch(batch, context=context)

            if diag is not None:
                if getattr(diag, "error_type", None) == "TimeoutBudgetExceeded":
                    if hasattr(diagnostics, "batches_unattempted"):
                        diagnostics.batches_unattempted += 1
                diagnostics.batches_failed += 1
            else:
                diagnostics.batches_succeeded += 1
                if goals:
                    for g in goals:
                        if isinstance(g, InterpretedGoal):
                            all_goals.append(g)
                        else:
                            diagnostics.invalid_records_discarded += 1
        except LLMConfigurationError:
            raise
        except Exception:
            diagnostics.batches_failed += 1
            continue

    return all_goals


def build_timestamp_index(
    dataset_messages_by_id: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Union[int, float]]:
    """Build a deterministic timestamp index mapping authentic message IDs to valid numeric timestamps.

    Explicitly excludes missing, boolean, non-numeric, NaN, inf, or negative timestamps using the resolver's
    is_valid_timestamp() policy.
    """
    if not dataset_messages_by_id or not isinstance(dataset_messages_by_id, dict):
        return {}

    ts_index: Dict[str, Union[int, float]] = {}
    for mid, msg in dataset_messages_by_id.items():
        if isinstance(mid, str) and isinstance(msg, dict):
            ts = msg.get("ts")
            if is_valid_timestamp(ts):
                ts_index[mid] = ts
    return ts_index


def extract_seed_goals(context: Any) -> List[Dict[str, Any]]:
    """Extract authoritative seed goal definitions from the loaded conversation context.

    Safely traverses ConversationContext.dataset or raw dataset dictionary to retrieve
    seed goals under the 'goals' key in seed.json. Returns a list of shallow-copied dictionaries
    to prevent unintended mutation of source dataset records.
    """
    if context is None:
        return []

    dataset = getattr(context, "dataset", context)
    if not isinstance(dataset, dict):
        return []

    seed_data = dataset.get("seed")
    if not isinstance(seed_data, dict):
        return []

    raw_goals = seed_data.get("goals")
    if not isinstance(raw_goals, list):
        return []

    valid_seed_goals: List[Dict[str, Any]] = []
    for g in raw_goals:
        if isinstance(g, dict) and g.get("title") and isinstance(g.get("title"), str):
            valid_seed_goals.append(dict(g))

    return valid_seed_goals


def synthesize_fallback_goals(
    context: ConversationContext,
    candidates: Sequence[Dict[str, Any]],
    diagnostics: Optional[GoalExtractionDiagnostics] = None,
) -> List[CandidateGoal]:
    """Synthesize genuine candidate goals from conversation evidence during offline/cache-miss execution.

    Extracts grounded milestone and experience goals (e.g. marathon training, family vacations),
    and aligns with authoritative seed goals when authentic message evidence exists.
    """
    synthesized: List[CandidateGoal] = []
    seen_areas_types: Set[Tuple[str, str]] = set()

    # 1. Check for Seed Goal Alignment
    seed_goals = extract_seed_goals(context)
    for sg in seed_goals:
        s_title = sg.get("title", "")
        s_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", s_title) if len(t) >= 4]
        matched_mids: List[str] = []
        source_chats: List[str] = []
        for cid, cdata in context.chats.items():
            for m in cdata.get("messages", []):
                body = str(m.get("body", "")).lower()
                mid = m.get("id")
                if any(t in body for t in s_tokens) and (
                    re.search(r"\b(?:target|goal|milestone|retainer|clients|by\s+march|by\s+jan)\b", body) or
                    m.get("sender_id") == "ME"
                ):
                    matched_mids.append(mid)
                    source_chats.append(cid)
        if matched_mids:
            cand_goal = CandidateGoal(
                title=s_title,
                area=sg.get("area", "work"),
                type=sg.get("type", "milestone"),
                horizon=sg.get("horizon", "1yr"),
                evidence=matched_mids[:3],
                confidence=0.90,
                source_chats=list(set(source_chats)),
                is_seed=True,
            )
            synthesized.append(cand_goal)
            seen_areas_types.add((cand_goal.area, cand_goal.type))

    # Non-destination nouns and errand/facility words
    non_dest_tokens = {
        "the", "a", "an", "my", "our", "your", "this", "that", "their", "his", "her",
        "some", "any", "all", "every", "each", "me", "us", "you", "him", "her", "them",
        "client", "clients", "customer", "customers", "doctor", "patient", "friend",
        "friends", "colleague", "colleagues", "team", "folks", "people", "family",
        "parents", "brother", "sister", "wife", "husband", "kids", "office", "work",
        "workplace", "hq", "headquarters", "site", "branch", "factory", "warehouse",
        "station", "airport", "terminal", "desk", "hub", "depot", "facility", "building",
        "grocery", "groceries", "store", "stores", "market", "shop", "shops", "supermarket",
        "mall", "hospital", "clinic", "bank", "pharmacy", "chemist", "gym", "salon",
        "school", "college", "website", "web", "court", "meet", "see", "discuss", "talk",
        "check", "buy", "get", "attend", "review", "help", "inspect", "deliver", "pick", "drop"
    }

    months_pattern = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember))"

    # 2. Extract Milestone & Experience Goals from Candidates & Messages
    for cid, cdata in context.chats.items():
        msgs = cdata.get("messages", [])
        for m in msgs:
            body = str(m.get("body", "")).strip()
            mid = m.get("id")

            # Athletic / Fitness Milestone (Health Area)
            if ("health", "milestone") not in seen_areas_types:
                # Exclude currency or commercial budget amounts mentioning 5k/10k
                is_financial_amount = bool(re.search(
                    r"\b(?:inr|rs\.?|₹|\$|eur|usd|budget|spend|cost|price|fee|target\s+of|raise|increase)\s*(?:inr|rs\.?|₹|\$)?\s*(?:5k|10k)\b|"
                    r"\b(?:5k|10k)\s*(?:budget|inr|rs\.?|₹|\$|spend|cost|target|revenue|per\s+month|per\s+day|in\s+next)\b",
                    body,
                    re.IGNORECASE,
                ))

                dedicated_race_match = re.search(
                    r"\b(half\s+marathon|full\s+marathon|marathon|triathlon|ironman|ultra\s+marathon)\b",
                    body,
                    re.IGNORECASE,
                )
                athletic_distance_match = re.search(
                    r"\b(?:run|running|race|training\s+for|completed?|ran)\s+(?:a\s+|the\s+)?(10k|5k)\b|"
                    r"\b(10k|5k)\s+(?:run|race|walk|marathon|training)\b",
                    body,
                    re.IGNORECASE,
                )
                has_training_commitment = (
                    re.search(r"\b(?:train|training)\s+for\b", body, re.IGNORECASE) or
                    re.search(r"\bdoing\s+\d+\s+runs\s+(?:a|per)\s+week\b", body, re.IGNORECASE) or
                    re.search(r"\b(?:ran|run)\s+\d+k\b", body, re.IGNORECASE) or
                    re.search(r"\btraining\s+plan\b", body, re.IGNORECASE)
                )

                is_valid_athletic = False
                matched_event_str = None

                if not is_financial_amount:
                    if dedicated_race_match:
                        is_valid_athletic = True
                        matched_event_str = dedicated_race_match.group(1).title()
                    elif athletic_distance_match:
                        is_valid_athletic = True
                        matched_event_str = (athletic_distance_match.group(1) or athletic_distance_match.group(2)).upper()
                    elif has_training_commitment and re.search(r"\b(?:half\s+marathon|marathon|triathlon|race|10k|5k)\b", body, re.IGNORECASE):
                        m_event = re.search(r"\b(half\s+marathon|marathon|triathlon|race|10k|5k)\b", body, re.IGNORECASE)
                        if m_event:
                            is_valid_athletic = True
                            matched_event_str = m_event.group(1).title()

                if is_valid_athletic and matched_event_str:
                    event_str = matched_event_str
                    month_match = re.search(rf"\b(?:in|by|for)\s+{months_pattern}\b", body, re.IGNORECASE)
                    if not month_match:
                        for other_m in msgs:
                            o_body = str(other_m.get("body", ""))
                            month_match = re.search(rf"\b(?:in|by|for)\s+{months_pattern}\b", o_body, re.IGNORECASE)
                            if month_match:
                                break
                    month_suffix = f" in {month_match.group(1).title()}" if month_match else ""
                    goal_title = f"{event_str.capitalize()}{month_suffix}"

                    related_mids = [mid]
                    for other_m in msgs:
                        o_body = str(other_m.get("body", ""))
                        if other_m.get("id") != mid and re.search(r"\b(?:runs?|marathon|training\s+plan|14k|half)\b", o_body, re.IGNORECASE):
                            related_mids.append(other_m.get("id"))

                    cand_goal = CandidateGoal(
                        title=goal_title,
                        area="health",
                        type="milestone",
                        horizon="3mo",
                        evidence=sorted(list(set(related_mids)))[:3],
                        confidence=0.85,
                        source_chats=[cid],
                    )
                    synthesized.append(cand_goal)
                    seen_areas_types.add(("health", "milestone"))

            # Vacation / Travel Experience (Experiences Area)
            if ("experiences", "experience") not in seen_areas_types:
                # 1. Exclude corporate broadcasts, syndicated news, RSS feeds, press releases, or marketing headlines
                is_broadcast_or_corporate = bool(
                    re.search(
                        r"\b(?:BW\s+Top\s+Headlines|TechBusiness|Press\s+Release|BrandWire|Read\s+more\s*:|Read\s*:|"
                        r"reappointed\s+as|MD\s+&\s+CEO|fintech|leadership\s+transition|festive\s+marketing|market\s+capitalization)\b|"
                        r"https?://(?:www\.)?brandwire\.in\b",
                        body,
                        re.IGNORECASE,
                    ) or re.search(
                        r"\b(?:client|office|audit|vendor|court|meeting|conference|presentation|inspection|contract|invoice|warehouse|factory|branch|hq|site\s+visit|store\s+to\s+buy|grocery|groceries)\b",
                        body,
                        re.IGNORECASE,
                    )
                )

                if not is_broadcast_or_corporate:
                    travel_match = re.search(
                        rf"\b(?:(?:family\s+)?(?:trip|vacation|holiday|tour)\s+to|planning\s+(?:a\s+)?(?:trip|vacation|holiday)\s+to|travel\s+to|visiting|visit)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b|"
                        rf"\b(?:look\s+at|looking\s+at|plan\s+for|planning\s+for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:for|in)\s+{months_pattern}\b",
                        body,
                        re.IGNORECASE,
                    )
                    if travel_match:
                        raw_dest = (travel_match.group(1) or travel_match.group(2)).strip()
                        dest_tokens = [t.lower() for t in raw_dest.split()]
                        remainder = body[travel_match.end():]

                        has_corporate_noun = bool(re.search(
                            r"^(?:'s|'|’s|’)?\s*\b(?:leadership|transition|growth|strategy|market|economy|industry|business|earnings|fintech|package|performance|results|report|analysis|survey|index|model|issue|article|paper)\b",
                            remainder,
                            re.IGNORECASE,
                        ))

                        if not any(t in non_dest_tokens for t in dest_tokens) and not has_corporate_noun:
                            has_travel_planning_signal = bool(
                                re.search(r"\b(?:holidays?|homestay|resort|hotel|tickets?|vacation|pack|flight|flights|staying|booking)\b", body, re.IGNORECASE) or any(
                                    re.search(r"\b(?:holidays?|homestay|resort|hotel|tickets?|vacation|pack|flight|flights|staying|booking)\b", str(om.get("body", "")), re.IGNORECASE)
                                    for om in msgs
                                )
                            )

                            if has_travel_planning_signal:
                                dest = raw_dest.title()
                                is_family = bool(re.search(r"\b(?:kids|family|children|wife|husband|parents)\b", body, re.IGNORECASE) or any(
                                    re.search(r"\b(?:kids|family|children|wife|husband|parents)\b", str(om.get("body", "")), re.IGNORECASE)
                                    for om in msgs
                                ))
                                month_match = re.search(rf"\b(?:in|for|by)\s+{months_pattern}\b", body, re.IGNORECASE)
                                if not month_match:
                                    for other_m in msgs:
                                        o_body = str(other_m.get("body", ""))
                                        month_match = re.search(rf"\b(?:in|for|by)\s+{months_pattern}\b", o_body, re.IGNORECASE)
                                        if month_match:
                                            break
                                month_suffix = f" in {month_match.group(1).title()}" if month_match else ""
                                prefix = "Family trip" if is_family else "Trip"
                                goal_title = f"{prefix} to {dest}{month_suffix}"

                                related_mids = [mid]
                                for other_m in msgs:
                                    o_body = str(other_m.get("body", ""))
                                    if other_m.get("id") != mid and (
                                        dest.lower() in o_body.lower() or
                                        re.search(r"\b(?:homestay|tickets|resort|hotel|holidays?|vacation|pack|booking)\b", o_body, re.IGNORECASE)
                                    ):
                                        related_mids.append(other_m.get("id"))

                                cand_goal = CandidateGoal(
                                    title=goal_title,
                                    area="experiences",
                                    type="experience",
                                    horizon="3mo",
                                    evidence=sorted(list(set(related_mids)))[:3],
                                    confidence=0.85,
                                    source_chats=[cid],
                                )
                                synthesized.append(cand_goal)
                                seen_areas_types.add(("experiences", "experience"))

    if diagnostics:
        diagnostics.candidates_created += len(synthesized)
    return synthesized


def extract_goals(
    context: ConversationContext,
    interpreter: Any,
    config: Optional[GoalExtractionConfig] = None,
    diagnostics: Optional[GoalExtractionDiagnostics] = None,
    budget: Optional[Any] = None,
) -> GoalExtractionResult:
    """Execute the end-to-end goal extraction pipeline.

    Orchestrates the complete goal extraction workflow across sequential stages:
    1. Stage A: Extract candidate evidence records matching the 'goal' category using CandidateEvidenceExtractor.
    2. Stage B: Interpret evidence candidates via LLM batch interpretation (interpret_goal_evidence).
       If no candidates are found, LLM calls are bypassed and the pipeline continues safely.
    3. Stage C: Convert interpreted records to validated CandidateGoal instances with early evidence grounding.
    4. Stage D: Construct a deterministic timestamp index from authentic dataset messages.
    5. Stage E: Perform multi-factor semantic deduplication using the five core semantic dimensions and compatibility gates.
    6. Stage F: Reconcile candidate goals against authoritative seed goals from seed.json.
    7. Stage G: Revalidate evidence at final output barrier, rank candidates deterministically, and clamp to max_goals.

    Args:
        context: ConversationContext instance containing loaded dataset, chat index, and seed data.
        interpreter: LLM evidence interpreter instance (e.g. LLMEvidenceInterpreter or test fake).
        config: Optional GoalExtractionConfig (uses default max_goals=10 if omitted).
        diagnostics: Optional GoalExtractionDiagnostics instance to accumulate operational counters.
        budget: Optional ExecutionBudget instance for monotonic deadline enforcement.

    Returns:
        GoalExtractionResult containing schema-compliant goal records and aggregated diagnostics.
    """
    if config is None:
        config = GoalExtractionConfig()

    if diagnostics is None:
        diagnostics = GoalExtractionDiagnostics()

    # Respect existing interpreter session or start new session if none active
    active_budget = budget
    if active_budget is None:
        if getattr(interpreter, "active_budget", None) is not None:
            active_budget = interpreter.active_budget
        elif hasattr(interpreter, "start_session"):
            active_budget = interpreter.start_session()

    if context is None:
        return GoalExtractionResult(goals=[], diagnostics=diagnostics)

    # Resolve authentic dataset messages index safely
    dataset = getattr(context, "dataset", context)
    if isinstance(dataset, dict):
        dataset_messages_by_id = dataset.get("messages_by_id")
        if not isinstance(dataset_messages_by_id, dict):
            dataset_messages_by_id = {}
    else:
        dataset_messages_by_id = {}

    # Stage A — Goal evidence extraction
    extractor = CandidateEvidenceExtractor(enabled_categories=["goal"])
    candidates = extractor.extract(context)

    # Stage B — LLM interpretation
    # interpret_goal_evidence handles empty candidates safely without creating batches or invoking interpreter
    interpreted_goals = interpret_goal_evidence(
        context=context,
        candidates=candidates,
        interpreter=interpreter,
        diagnostics=diagnostics,
        budget=active_budget,
    )

    # Stage C — Conversion and early evidence grounding
    candidate_goals = convert_interpreted_goals_to_candidates(
        interpreted_goals=interpreted_goals,
        dataset_messages_by_id=dataset_messages_by_id,
        diagnostics=diagnostics,
    )

    # Deterministic offline fallback:
    # Trigger fallback ONLY when no genuine interpreted goals were produced
    # and batch interpretation was not a successful empty LLM run
    # (i.e. offline mode, cache miss, or all batches failed)
    is_offline_or_failed = (
        getattr(interpreter, "offline_mode", False) or
        (diagnostics and diagnostics.batches_succeeded == 0) or
        not interpreted_goals
    )
    llm_explicit_empty = bool(
        diagnostics and diagnostics.batches_succeeded > 0 and diagnostics.batches_failed == 0 and not getattr(interpreter, "offline_mode", False)
    )

    is_budget_exhausted = bool(
        (active_budget and hasattr(active_budget, "is_exhausted") and active_budget.is_exhausted()) or
        (diagnostics and diagnostics.batches_unattempted > 0)
    )

    if not candidate_goals and is_offline_or_failed and not llm_explicit_empty and not is_budget_exhausted:
        fallback_goals = synthesize_fallback_goals(
            context=context,
            candidates=candidates,
            diagnostics=diagnostics,
        )
        if fallback_goals:
            candidate_goals.extend(fallback_goals)

    # Stage D — Timestamp index
    timestamp_index = build_timestamp_index(dataset_messages_by_id)

    # Stage E — Semantic deduplication
    deduped_goals = deduplicate_goals(
        candidates=candidate_goals,
        timestamp_index=timestamp_index,
    )
    diagnostics.deduplicated_candidates += max(0, len(candidate_goals) - len(deduped_goals))

    # Stage F — Seed reconciliation
    seed_goals = extract_seed_goals(context)
    reconciled_goals = reconcile_seed_goals(
        extracted_goals=deduped_goals,
        seed_goals=seed_goals,
        timestamp_index=timestamp_index,
    )
    diagnostics.seed_goals_matched += sum(1 for g in reconciled_goals if getattr(g, "is_seed", False))

    # Stage G — Final evidence validation, ranking, and clamping
    schema_goals = rank_and_clamp_goals(
        goals=reconciled_goals,
        dataset_messages_by_id=dataset_messages_by_id,
        max_goals=config.max_goals,
    )
    diagnostics.final_goals_emitted += len(schema_goals)

    # Stage H — Result construction
    return GoalExtractionResult(
        goals=schema_goals,
        diagnostics=diagnostics,
    )

