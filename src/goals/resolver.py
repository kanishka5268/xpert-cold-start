"""Multi-factor semantic deduplication and strict seed reconciliation for candidate goals."""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.goals.models import CandidateGoal, GoalMetrics
from src.interpretation.records import InterpretedGoal

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "our", "my", "we", "us", "and", "or", "from", "this",
    "it", "that", "i", "me", "you", "will", "be"
}


def is_valid_timestamp(ts: Any) -> bool:
    """Validate that ts is a finite, non-negative numeric timestamp.

    Explicitly rejects booleans (True, False), None, strings, collections,
    NaN, inf, and negative numbers.
    """
    if isinstance(ts, bool):  # In Python, bool is subclass of int; reject immediately
        return False
    if isinstance(ts, (int, float)):
        return math.isfinite(ts) and ts >= 0
    return False


def validate_confidence(val: Any, default: float = 0.5) -> float:
    """Validate and normalize a confidence score to a finite float in [0.0, 1.0].

    Explicitly rejects booleans, None, strings, NaN, inf, and out-of-range values,
    defaulting safely to 0.5.
    """
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        if math.isfinite(val) and 0.0 <= val <= 1.0:
            return float(val)
    return default


def tokenize_goal_title(title: str) -> Set[str]:
    """Extract normalized content tokens from a goal title."""
    cleaned = re.sub(r"[^\w\s]", "", title.lower())
    return {t for t in cleaned.split() if t not in STOP_WORDS and len(t) > 1}


def extract_goal_metrics(title: str) -> GoalMetrics:
    """Extract quantifiable metrics, units, percentages, and athletic distances from a goal title,
    using contextual evidence to resolve potential ambiguities.

    Contextual Disambiguation Rules:
    - Large Western quantities ('10M', '10 M', '1.5 Million', '2B', '2 B') are classified as quantities
      with unit 'm' or 'b' (e.g. quantities_with_units={'10m'}).
    - '10k', '10 k', '5k', or '5 k' is classified as an athletic distance ONLY when accompanied by athletic context
      (e.g., '10k race', '10 k race', 'run 10k', 'run 10 k', '10 km race', '10k run', 'training for 10k').
    - '10k' or '10 k' is classified as a quantity ONLY when accompanied by financial or commercial context
      (e.g., 'save 10k', 'save 10 k', '10k revenue', '10 k customers', 'reach 10k mrr', 'earn 10k').
    - If context is insufficient (e.g. standalone '10k by December', '10 k by December', 'Target 10k',
      'Achieve 10 k') or conflicting, the token is preserved in ambiguous_metrics, preventing unsafe deduplication.
    - Captures multiple distinct metrics in a single title (e.g., 'Save 20k and run 10k race'
      populates both quantities_with_units={'20k'} and distances={'10k'}).
    - Plain numbers exclude spans already captured by percentages, units, distances, or ambiguous metrics.
    """
    cleaned = f" {title.lower()} "
    captured_spans: List[Tuple[int, int]] = []
    pct_matches: Set[str] = set()
    distances: Set[str] = set()
    quantities: Set[str] = set()
    ambiguous_metrics: Set[str] = set()

    # 1. Percentages (compact and spaced, integer and decimal)
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*%", cleaned):
        num_str = m.group(1)
        pct_matches.add(f"{num_str}%")
        captured_spans.append((m.start(), m.end()))

    # 2. Inherent athletic distances and named events
    event_patterns = [
        (r"\bhalf\s+marathon\b", "half_marathon"),
        (r"\bfull\s+marathon\b", "full_marathon"),
        (r"\bmarathon\b", "marathon"),
        (r"\btriathlon\b", "triathlon"),
        (r"\bironman\b", "ironman"),
    ]
    for pattern, label in event_patterns:
        for m in re.finditer(pattern, cleaned):
            # Avoid duplicate subsumption (e.g. 'marathon' within 'half marathon')
            if not any(c_start <= m.start() and m.end() <= c_end for c_start, c_end in captured_spans):
                distances.add(label)
                captured_spans.append((m.start(), m.end()))

    # 3. Quantities with Indian units (Lakhs, Crores)
    indian_unit_pattern = r"\b(\d+(?:\.\d+)?)\s*(lakhs?|lakh|l|crores?|crore|cr)\b"
    for m in re.finditer(indian_unit_pattern, cleaned):
        num_str, unit_str = m.group(1), m.group(2)
        norm_unit = "l" if unit_str in {"lakh", "lakhs", "l"} else "cr"
        quantities.add(f"{num_str}{norm_unit}")
        captured_spans.append((m.start(), m.end()))

    # 4. Large Western quantities (Million, Billion)
    western_large_pattern = r"\b(\d+(?:\.\d+)?)\s*(million|m|billion|b)\b"
    for m in re.finditer(western_large_pattern, cleaned):
        num_str, unit_str = m.group(1), m.group(2)
        norm_unit = "b" if "b" in unit_str else "m"
        quantities.add(f"{num_str}{norm_unit}")
        captured_spans.append((m.start(), m.end()))

    # 5. Contextual Disambiguation for 'k' and 'km' tokens
    k_pattern = r"\b(\d+(?:\.\d+)?)\s*(k|km)\b"
    for m in re.finditer(k_pattern, cleaned):
        start, end = m.start(), m.end()
        # Avoid spans already captured (e.g. by larger phrases)
        if any(c_start <= start and end <= c_end for c_start, c_end in captured_spans):
            continue

        num_str, unit_str = m.group(1), m.group(2)
        val_norm = f"{int(float(num_str)) if float(num_str).is_integer() else num_str}k"

        # Local context windows (+/- 30 characters)
        prefix_window = cleaned[max(0, start - 30):start]
        suffix_window = cleaned[end:min(len(cleaned), end + 30)]

        # Contextual athletic signals
        athletic_following = bool(re.search(r"^\s*(race|run|marathon|triathlon|walk|jog|track|lap|laps|event|trail|course)\b", suffix_window))
        athletic_preceding = bool(re.search(r"\b(run|running|ran|jog|jogging|race|racing|walk|walking|train|training|compete|participate|finish|finishing|complete|completing)\s+(?:for\s+|in\s+|a\s+)?$", prefix_window))
        is_km_event = (unit_str == "km" and (athletic_following or athletic_preceding or "race" in cleaned or "run" in cleaned))
        has_athletic_context = athletic_following or athletic_preceding or is_km_event

        # Contextual financial/commercial signals
        financial_following = bool(re.search(r"^\s*(revenue|mrr|arr|sales|turnover|profit|income|salary|capital|funding|budget|cost|fee|fees|deposit|clients?|customers?|users?|subscribers?|accounts?|followers?|members?|leads?|downloads?|views|units?|items?|copies|dollars?|rs|rupees?|inr|usd)\b", suffix_window))
        financial_preceding = bool(re.search(r"\b(save|saving|saved|savings|invest|investing|invested|earn|earning|earned|spend|spending|budget|budgeting|fund|funding|raise|raising|deposit|acquire|acquiring|close|closing|generate|generating)\s+(?:for\s+|in\s+|a\s+)?$", prefix_window))
        has_financial_context = financial_following or financial_preceding

        # Disambiguation Decision
        if has_athletic_context and not has_financial_context:
            distances.add(val_norm)
            captured_spans.append((start, end))
        elif has_financial_context and not has_athletic_context:
            quantities.add(val_norm)
            captured_spans.append((start, end))
        elif has_athletic_context and has_financial_context:
            # Conflicting local signals on the same token
            ambiguous_metrics.add(val_norm)
            captured_spans.append((start, end))
        else:
            # Neither local context found; check broader title-level context
            title_athletic = bool(re.search(r"\b(race|marathon|triathlon|runner|running|jog|jogging|track|course)\b", cleaned))
            title_financial = bool(re.search(r"\b(revenue|mrr|arr|customers?|clients?|savings?|salary|invest|budget|finances?|sales)\b", cleaned))
            if title_athletic and not title_financial:
                distances.add(val_norm)
                captured_spans.append((start, end))
            elif title_financial and not title_athletic:
                quantities.add(val_norm)
                captured_spans.append((start, end))
            else:
                # Insufficient context (standalone '10k') or conflicting title signals -> preserve ambiguity
                ambiguous_metrics.add(val_norm)
                captured_spans.append((start, end))

    # 6. Word numbers (e.g. 'four times a week' -> 4.0)
    numbers: Set[float] = set()
    WORD_NUMBERS = {
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
        "six": 6.0,
        "seven": 7.0,
        "eight": 8.0,
        "nine": 9.0,
        "ten": 10.0,
    }
    for word, val in WORD_NUMBERS.items():
        for m in re.finditer(rf"\b{word}\b", cleaned):
            start, end = m.start(), m.end()
            if not any(c_start <= start and end <= c_end for c_start, c_end in captured_spans):
                numbers.add(val)
                captured_spans.append((start, end))

    # 7. Plain numbers (integer & decimal, excluding spans captured by units, percentages, or distances)
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", cleaned):
        start, end = m.start(), m.end()
        is_captured = any(c_start <= start and end <= c_end for c_start, c_end in captured_spans)
        if not is_captured:
            val = float(m.group(1))
            if val not in {2024.0, 2025.0, 2026.0, 2027.0, 2028.0, 2029.0, 2030.0}:
                numbers.add(val)

    return GoalMetrics(
        numbers=numbers,
        quantities_with_units=quantities,
        percentages=pct_matches,
        distances=distances,
        ambiguous_metrics=ambiguous_metrics,
    )


def are_metrics_compatible(m1: GoalMetrics, m2: GoalMetrics) -> bool:
    """Evaluate whether two sets of goal metrics are semantically compatible.

    Conservative Cross-Category & Ambiguity Policy:
    1. Ambiguity Guard: If either goal contains ambiguous metrics (m1.ambiguous_metrics
       or m2.ambiguous_metrics), return False. Ambiguity is strictly preserved to prevent
       unsafe deduplication.
    2. Empty Check: If neither goal specifies metrics (both is_empty()), return True.
    3. Asymmetry Check: If one specifies metrics and the other does not, return False.
    4. Category Presence Match: Both metric objects must possess the exact same populated categories:
       - bool(m1.numbers) == bool(m2.numbers)
       - bool(m1.quantities_with_units) == bool(m2.quantities_with_units)
       - bool(m1.percentages) == bool(m2.percentages)
       - bool(m1.distances) == bool(m2.distances)
       If categories differ (e.g. distance '10k' vs monetary quantity '10k', quantity '20l' vs plain number '20.0',
       or percentage '50%' vs plain number '50.0'), return False.
    5. Category Value Match: Within each populated category, values must match exactly:
       - m1.numbers == m2.numbers
       - m1.quantities_with_units == m2.quantities_with_units
       - m1.percentages == m2.percentages
       - m1.distances == m2.distances
    """
    # 1. Ambiguity guard (insufficient or conflicting context prevents unsafe merge)
    if m1.ambiguous_metrics or m2.ambiguous_metrics:
        return False

    # 2. Empty check
    if m1.is_empty() and m2.is_empty():
        return True
    if m1.is_empty() != m2.is_empty():
        return False

    # 4. Category presence match
    if bool(m1.numbers) != bool(m2.numbers):
        return False
    if bool(m1.quantities_with_units) != bool(m2.quantities_with_units):
        return False
    if bool(m1.percentages) != bool(m2.percentages):
        return False
    if bool(m1.distances) != bool(m2.distances):
        return False

    # 5. Value equality within identical categories
    if m1.numbers != m2.numbers:
        return False
    if m1.quantities_with_units != m2.quantities_with_units:
        return False
    if m1.percentages != m2.percentages:
        return False
    if m1.distances != m2.distances:
        return False

    return True


def resolve_core_action(title: str, area: str) -> Optional[str]:
    """Contextually resolve the core action category for a goal title.

    Guarantees:
    - 'run' maps to 'athletic' ONLY when supported by explicit athletic event markers
      (e.g., 'marathon', 'half', 'race', 'triathlon', 'ironman', '5k', '10k', distance markers).
      Area ('health') supports, but cannot independently force an athletic classification.
    - 'run a half marathon' and 'participate in a half marathon' both resolve to 'athletic'.
    - 'participate in a race' with explicit race-event context resolves to 'athletic'.
    - 'race' and 'racing' require independent athletic event markers (e.g., 'marathon', 'half', '10k', '5k', 'triathlon', 'ironman', distance markers).
      Area ('health') alone must NOT independently force an athletic classification.
      'Race' with area 'health' and 'Racing every morning' with area 'health' return None (unresolved).
    - 'run a company' and 'run a business' resolve to 'commercial_growth' ONLY when supported
      by explicit business context tokens ('company', 'business', etc.). Area ('work') alone does NOT establish commercial growth.
    - Planned travel ambitions ('Family trip to Coorg', 'We want to visit Coorg') resolve to 'travel_experience'.
    - Operational booking tasks ('Book flight tickets to Coorg', 'Reserve hotel') resolve to 'operational_booking'.
    - Plural running activity support: recognizes 'run', 'runs', 'running', 'ran', 'jog', 'jogging' consistently.
    - Intentional habit fitness routines ('Start running four times a week', 'Maintain four runs a week',
      'Continue my running routine', 'Build a running routine') resolve to 'fitness_habit'.
    - Milestone preparation ('Doing four runs a week to train for the marathon') resolves to 'athletic'.
    - Intentional mindfulness habits ('Start daily meditation', 'Maintain a meditation routine',
      'Start a meditation routine', 'Keep up my meditation routine') resolve to 'mindfulness_habit'.
    - 'routine' alone is NOT sufficient evidence of adoption or maintenance intent:
      Factual descriptions of an existing routine ('I have a meditation routine', 'My running routine is four days a week')
      and descriptive routines ('I run four times a week', 'I meditate every morning') strictly return None.
    - Area alone ('health', 'self', 'work', 'home') must NEVER convert a routine, errand, or generic task into a goal:
      Generic technical tasks ('Run tests'), generic errands ('Buy groceries' with area 'home'), and ungrounded racing ('Race', 'Racing every morning') return None.
    - Generic tactical execution ('Run it tomorrow', 'Run daily' without habit commitment, 'Run tests') returns None (unresolved).
    - Ambiguous or uncertain actions strictly return None (NEVER arbitrarily forced).
    """
    tokens = tokenize_goal_title(title)
    cleaned = f" {title.lower()} "

    # Independent athletic event evidence (excluding the bare action verb 'race' / 'racing')
    independent_event_evidence = (
        bool(tokens & {"marathon", "half", "full", "10k", "5k", "triathlon", "ironman", "km", "miles", "track"})
        or "half marathon" in cleaned
        or "full marathon" in cleaned
    )
    # Athletic event context for verbs like run, participate, compete, complete, finish, enter
    athletic_event_evidence = independent_event_evidence or bool(tokens & {"race"})
    business_context = bool(tokens & {"company", "business", "agency", "firm", "startup", "operations", "team", "organization"})

    # 1. Operational booking tasks (tactical logistics)
    if tokens & {"book", "booking", "reserve", "reserving", "ticket", "tickets", "cab", "cabs", "hotel", "hotels"}:
        if any(w in cleaned for w in ["flight ticket", "hotel", "ticket", "cab", "booking", "reserve"]):
            return "operational_booking"

    # 2. Travel & Experience ambitions
    if tokens & {"trip", "visit", "visiting", "travel", "travelling", "vacation", "holiday", "tour", "explore", "exploring"}:
        return "travel_experience"

    # 3. Running & Fitness: Milestone preparation vs. Fitness habits vs. Commercial vs. Unresolved
    has_frequency_routine = (
        bool(re.search(r"\b(\d+|four|three|five|two)\s+(?:runs?|times?|days?)\s+(?:a|per)\s+week\b", cleaned))
        or "runs a week" in cleaned
        or "runs per week" in cleaned
        or "workout" in cleaned
        or "workouts" in cleaned
    )
    # Explicit adoption or maintenance intent required for habits (area alone cannot convert routine to goal)
    # CRITICAL: "routine" alone is NOT sufficient evidence of adoption or maintenance intent.
    # Requires explicit volitional commitment verbs (adoption: start, begin, adopt, build, establish; maintenance: maintain, keep, sustain, continue, keep up).
    has_habit_intent = (
        bool(tokens & {
            "start", "starting", "begin", "beginning", "adopt", "adopting", "introduce", "build", "building", "establish",
            "maintain", "maintaining", "keep", "keeping", "sustain", "sustaining", "continue", "continuing"
        })
        or "keep up" in cleaned
    )

    if tokens & {"run", "runs", "running", "ran", "jog", "jogging"}:
        # Milestone preparation: e.g. "Doing four runs a week to train for the marathon"
        if athletic_event_evidence:
            return "athletic"
        # Business operational leadership: e.g. "Run a company", "Run a business"
        # Explicit business context required; area "work" alone must NOT independently establish commercial growth
        if business_context:
            return "commercial_growth"
        # Fitness habit with intentional commitment:
        # e.g. "Start running four times a week", "Maintain four runs a week", "Continue my running routine", "Build a running routine"
        # Note: has_habit_intent is strictly required; factual routines ("My running routine is four days a week") return None.
        if has_habit_intent and (has_frequency_routine or tokens & {"routine"}):
            return "fitness_habit"
        # Isolated status quo ("I run four times a week", "My running routine is four days a week") or ungrounded ("Run daily", "Run it tomorrow", "Run tests") -> unresolved
        return None

    # 4. 'participate', 'compete', 'complete', 'finish', 'enter'
    # Explicit athletic event context (e.g. 'Participate in a race', 'Finish the marathon') resolves to 'athletic'
    if tokens & {"participate", "participating", "compete", "competing", "complete", "completing", "finish", "finishing", "enter"}:
        if athletic_event_evidence:
            return "athletic"
        return None

    # 5. 'race', 'racing'
    # Requires independent athletic event evidence (e.g. 'Race a 10k', 'Race the marathon');
    # Area ('health') alone must NOT independently establish 'athletic' without event context.
    # Bare 'Race' or 'Racing every morning' with area 'health' strictly returns None.
    if tokens & {"race", "racing"}:
        if independent_event_evidence:
            return "athletic"
        return None

    # 6. Mindfulness & Meditation habits (requires explicit intentional habit intent; area alone NEVER converts routine)
    # e.g. "Start daily meditation", "Maintain a meditation routine", "Keep up my meditation routine", "Start a meditation routine" -> mindfulness_habit
    # Factual descriptions lacking volitional commitment ("I have a meditation routine", "I meditate every morning") -> None
    if tokens & {"meditation", "meditate", "mindfulness"}:
        if has_habit_intent:
            return "mindfulness_habit"
        return None

    # 7. Unambiguous action indicators
    if tokens & {"save", "saving", "invest", "investing", "accumulate", "budget"}:
        return "saving"
    if tokens & {"hire", "recruit"}:
        return "hiring"
    if tokens & {"learn", "study", "master"}:
        return "learning"

    # 8. Contextual resolution for overlapping verb 'acquire'
    if tokens & {"acquire", "acquiring", "acquisition"}:
        if tokens & {"client", "clients", "customer", "customers", "retainer", "account", "revenue", "mrr", "arr"}:
            return "commercial_growth"
        if tokens & {"flat", "house", "property", "land", "plot"} or "real estate" in cleaned:
            return "property"
        if tokens & {"skill", "certification", "language"}:
            return "learning"
        return None

    # 9. Contextual resolution for 'buy', 'purchase', 'own'
    # Requires explicit property tokens; area ('home') alone must NOT independently establish property action
    if tokens & {"buy", "purchase", "own", "renovate", "move"}:
        if tokens & {"flat", "house", "property", "home", "villa", "apartment", "land", "plot"} or "real estate" in cleaned:
            return "property"
        return None

    # 10. Contextual resolution for 'reach', 'hit', 'scale', 'grow', 'close'
    if tokens & {"reach", "hit", "scale", "grow", "close", "bring"}:
        if tokens & {"client", "clients", "customer", "customers", "retainer", "revenue", "cr", "lakh", "mrr", "arr"}:
            return "commercial_growth"
        if athletic_event_evidence:
            return "athletic"
        return None

    # 11. General habit markers without specific domain keywords
    if tokens & {"start", "build", "maintain", "keep", "adopt"}:
        if tokens & {"company", "startup", "product", "agency", "business"}:
            return "commercial_growth"
        if tokens & {"house", "home"}:
            return "property"
        return None

    return None


def extract_target_entity(title: str, area: str) -> Optional[str]:
    """Extract the normalized domain entity category, returning None if ambiguous or conflicting.

    Matches multi-word phrases against normalized boundary-padded text.
    Extracts travel destinations at specific destination-level granularity (e.g. 'destination_coorg').
    Distinguishes running habits ('habit_running') from milestone athletic events ('athletic_event').
    Detects multiple conflicting entity signals and returns None.
    """
    tokens = tokenize_goal_title(title)
    cleaned = f" {re.sub(r'[^a-z0-9\s]', ' ', title.lower())} "
    matched_categories: Set[str] = set()

    # 1. Travel Destinations (Experience goals)
    KNOWN_DESTINATIONS = {
        "coorg", "paris", "goa", "london", "japan", "tokyo", "europe", "bali",
        "himalayas", "kashmir", "ladakh", "vietnam", "thailand", "italy",
        "dubai", "singapore", "switzerland", "new york", "hawaii"
    }
    matched_destinations: Set[str] = set()

    # Match known destinations (checking multi-word before single-word)
    for dest in sorted(KNOWN_DESTINATIONS, key=len, reverse=True):
        if f" {dest} " in cleaned:
            dest_slug = dest.replace(" ", "_")
            matched_destinations.add(f"destination_{dest_slug}")

    # Fallback: single-word travel phrase match ONLY if no known destination matched.
    # This avoids emitting partial sub-word entities (e.g. 'destination_new') when a
    # complete multi-word known destination ('destination_new_york') is present.
    if not matched_destinations:
        dest_match = re.search(
            r"\b(?:trip\s+to|visit|visiting|travel\s+to|travelling\s+to|vacation\s+in|holiday\s+in|tour\s+of|explore|exploring|flight\s+to)\s+([a-z]+)\b",
            cleaned
        )
        if dest_match:
            candidate_dest = dest_match.group(1)
            if candidate_dest not in STOP_WORDS and len(candidate_dest) > 2:
                matched_destinations.add(f"destination_{candidate_dest}")

    matched_categories.update(matched_destinations)

    # 2. Athletic Events vs. Running Habits vs. Meditation Habits
    athletic_event_markers = (
        bool(tokens & {"marathon", "half", "full", "10k", "5k", "race", "triathlon", "ironman"})
        or "half marathon" in cleaned
        or "full marathon" in cleaned
    )
    if athletic_event_markers:
        matched_categories.add("athletic_event")
    elif tokens & {"run", "runs", "running", "ran", "jog", "jogging"} or "runs a week" in cleaned:
        matched_categories.add("habit_running")

    if tokens & {"meditation", "meditate", "mindfulness"}:
        matched_categories.add("habit_meditation")

    # 3. Work entities
    if area == "work" or tokens & {"client", "engineer", "revenue", "saas"}:
        if tokens & {"client", "clients", "customer", "customers", "retainer", "retainers", "account", "accounts"}:
            matched_categories.add("client_acquisition")
        if tokens & {"engineer", "engineers", "developer", "developers", "hire", "recruits", "recruitment", "staff"}:
            matched_categories.add("talent_hiring")
        if tokens & {"saas", "product", "app", "platform", "tool", "website"}:
            matched_categories.add("product_launch")
        if tokens & {"arr", "mrr", "revenue", "funding", "capital"}:
            matched_categories.add("financial_metrics")

    # 4. Property entities
    if area in {"home", "finances"} or tokens & {"flat", "house", "interior"}:
        if tokens & {"flat", "house", "home", "apartment", "villa", "property", "plot", "land"} or " real estate " in cleaned:
            matched_categories.add("residential_property")
        if tokens & {"interior", "interiors", "renovation", "renovations", "furniture", "decor"} or " interior design " in cleaned:
            matched_categories.add("property_enhancement")

    # Multiple conflicting signals or no signals return None
    if len(matched_categories) == 1:
        return next(iter(matched_categories))
    return None


def extract_contextual_stage(title: str) -> Optional[str]:
    """Extract the temporal lifecycle stage, returning None if ambiguous, conflicting, or ungrounded.

    Supports 7 explicit lifecycle stages:
    1. stage_accumulation: saving, budgeting, accumulating funds, down payments
    2. stage_transaction: buying, purchasing, closing on property, or operational booking tasks
    3. stage_enhancement: renovating, remodeling, furnishing, upgrading
    4. stage_preparation: training, studying, conditioning, preparing for an event/milestone
    5. stage_execution: event participation/completion, race execution, metric target achievement,
                         or undertaking planned travel experiences
    6. stage_initiation: starting, beginning, adopting a new habit or process
    7. stage_maintenance: maintaining, keeping up an established habit or process against decay

    Detects multiple conflicting stage signals and returns None.
    Lacking explicit stage signals (e.g. ungrounded phrases) returns None.
    Unknown stages (None) strictly prevent deduplication under the safety barrier.
    """
    tokens = tokenize_goal_title(title)
    cleaned = f" {re.sub(r'[^a-z0-9\s]', ' ', title.lower())} "
    matched_stages: Set[str] = set()

    # 1. stage_accumulation (saving, budgeting)
    if tokens & {"save", "saving", "savings", "fund", "funding", "budget"} or " down payment " in cleaned:
        matched_stages.add("stage_accumulation")

    # 2. stage_transaction (purchasing property or operational booking tasks)
    if tokens & {"buy", "buying", "purchase", "purchasing", "own"} or " close on " in cleaned:
        matched_stages.add("stage_transaction")
    if tokens & {"book", "booking", "reserve", "reserving"} or "flight ticket" in cleaned or "hotel" in cleaned:
        matched_stages.add("stage_transaction")

    # 3. stage_enhancement (renovation, decor)
    if tokens & {"renovate", "renovating", "remodeling", "furnish", "upgrade", "interior"}:
        matched_stages.add("stage_enhancement")

    # 4. stage_preparation (training, preparation for events/milestones)
    # e.g. "Doing four runs a week to train for the marathon", "Train for the half marathon"
    if tokens & {"train", "training", "prepare", "preparing", "study", "studying", "practice", "practicing"}:
        matched_stages.add("stage_preparation")

    # 5. stage_execution (executing an event, achieving a milestone target, or undertaking planned travel)
    athletic_signals = (
        bool(tokens & {"marathon", "half", "full", "race", "triathlon", "ironman", "5k", "10k"})
        or "half marathon" in cleaned
        or "full marathon" in cleaned
    )
    is_athletic_execution = (
        bool(tokens & {"run", "runs", "running", "ran", "jog", "jogging", "participate", "participating", "compete", "competing", "complete", "completing", "finish", "finishing"})
        and athletic_signals
        and not (tokens & {"train", "training", "prepare", "preparing"})
    )
    is_commercial_execution = (tokens & {"reach", "hit", "scale", "acquire", "attain", "close"}) and (tokens & {"client", "clients", "customer", "customers", "mrr", "arr", "revenue"})
    is_travel_execution = (tokens & {"trip", "visit", "visiting", "travel", "travelling", "vacation", "holiday", "tour", "explore"}) and not (tokens & {"save", "saving", "budget", "book", "booking", "reserve"})

    if is_athletic_execution or is_commercial_execution or is_travel_execution:
        matched_stages.add("stage_execution")

    # 6. stage_initiation (adopting new habits/routines)
    if (
        tokens & {"start", "starting", "begin", "beginning", "adopt", "adopting", "introduce", "build", "building", "establish"}
    ):
        matched_stages.add("stage_initiation")

    # 7. stage_maintenance (sustaining ongoing habits/routines)
    # Note: "routine" alone is NOT a stage signal; requires explicit maintenance verbs (maintain, keep, sustain, continue, keep up).
    if (
        tokens & {"maintain", "maintaining", "keep", "keeping", "sustain", "sustaining", "continue", "continuing"}
        or "keep up" in cleaned
    ):
        if not (tokens & {"train", "training", "start", "begin"}):
            matched_stages.add("stage_maintenance")

    # Multiple conflicting signals or no signals return None
    if len(matched_stages) == 1:
        return next(iter(matched_stages))
    return None


def are_horizons_compatible(h1: str, h2: str, type1: str, type2: str) -> bool:
    """Determine whether two horizons are compatible for goal merging.

    Policy:
    - Identical horizons are always compatible.
    - Milestones and experiences have disjoint day boundaries: conflicting horizons strictly fail.
    - Habit goals: '3mo' (adoption) and '1yr' (annual maintenance) are compatible.
      Non-adjacent habit horizons ('3mo' vs '3yr', '3mo' vs '5yr') strictly fail.
    """
    if h1 == h2:
        return True
    if type1 == "habit" and type2 == "habit":
        if {h1, h2} == {"3mo", "1yr"}:
            return True
    return False


def select_canonical_horizon(g1: CandidateGoal, g2: CandidateGoal) -> str:
    """Select the canonical horizon when merging two compatible candidate goals during candidate deduplication.

    Since deduplication operates strictly on candidate goals prior to seed reconciliation,
    both g1 and g2 have is_seed=False. Seed horizon metadata is authoritative exclusively
    during reconcile_seed_goals(), not during candidate deduplication.

    Policy:
    - If horizons are identical, adopt that horizon.
    - Select the horizon of the candidate with greater verified evidence count.
    - If evidence counts are tied, select the candidate with higher validated confidence.
    - If tied, default to '1yr' for compatible habit goals ('3mo' vs '1yr').
    - Never silently inherit the first candidate's horizon based on iteration order.
    """
    if g1.horizon == g2.horizon:
        return g1.horizon
    if len(g1.evidence) > len(g2.evidence):
        return g1.horizon
    if len(g2.evidence) > len(g1.evidence):
        return g2.horizon
    conf1 = validate_confidence(g1.confidence)
    conf2 = validate_confidence(g2.confidence)
    if conf1 > conf2:
        return g1.horizon
    if conf2 > conf1:
        return g2.horizon
    if {g1.horizon, g2.horizon} == {"3mo", "1yr"}:
        return "1yr"
    return g1.horizon


def union_evidence(
    ev1: Sequence[str],
    ev2: Sequence[str],
    timestamp_index: Optional[Dict[str, Union[int, float]]] = None,
) -> List[str]:
    """Union two sequences of evidence message IDs preserving deduplication and explicit ordering.

    Four Operational Ordering Cases:
    1. Case A: All timestamps available & valid:
       Sorted strictly chronologically by (ts ascending, mid ascending).
    2. Case B: No timestamp index (None or empty):
       Deterministic lexicographical fallback (sorted by mid ascending).
       Explicitly documented as non-chronological fallback.
    3. Case C: Partial timestamps available:
       - Known valid timestamps are sorted chronologically: (ts ascending, mid ascending).
       - Unknown or unindexed IDs are sorted deterministically: (mid ascending).
       - Result: known_ts_ids + unknown_ts_ids.
       Explicitly documented that the tail sequence is a non-chronological fallback.
    4. Case D: Invalid timestamp values (NaN, inf, negative, booleans):
       Treated identically to unknown timestamps (placed in unknown_ts_ids).
    """
    seen: Set[str] = set()
    unique_ids: List[str] = []
    for mid in list(ev1) + list(ev2):
        if mid not in seen:
            seen.add(mid)
            unique_ids.append(mid)

    if not timestamp_index:
        return sorted(unique_ids)

    known_ts_ids: List[Tuple[float, str]] = []
    unknown_ts_ids: List[str] = []

    for mid in unique_ids:
        ts = timestamp_index.get(mid)
        if is_valid_timestamp(ts):
            known_ts_ids.append((float(ts), mid))  # type: ignore[arg-type]
        else:
            unknown_ts_ids.append(mid)

    known_sorted = [mid for _, mid in sorted(known_ts_ids, key=lambda x: (x[0], x[1]))]
    unknown_sorted = sorted(unknown_ts_ids)

    return known_sorted + unknown_sorted


def from_interpreted_goal(
    goal: InterpretedGoal,
    chat_id: Optional[str] = None,
    timestamp_index: Optional[Dict[str, Union[int, float]]] = None,
) -> CandidateGoal:
    """Create a new CandidateGoal from an InterpretedGoal without mutating the source."""
    ordered_evidence = union_evidence(goal.evidence_msgs, [], timestamp_index=timestamp_index)
    conf = validate_confidence(goal.confidence, default=0.5)
    return CandidateGoal(
        title=goal.title,
        area=goal.area,
        type=goal.type,
        horizon=goal.horizon,
        evidence=ordered_evidence,
        confidence=conf,
        source_chats=[chat_id] if chat_id else [],
        is_seed=False,
    )


def _stem_word(w: str) -> str:
    """Lightweight morphological stemmer for activity and domain words."""
    w = w.lower()
    for suffix in ("ing", "tion", "tions", "ies", "es", "ed", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            w = w[:-len(suffix)]
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


FREQ_HABIT_WORDS = {
    "time", "times", "week", "day", "days", "daily", "weekly", "month", "monthly",
    "year", "routine", "habit", "schedule", "per", "every", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten", "start", "maintain",
    "begin", "keep", "build", "adopt"
}


ACTIVITY_DOMAINS: List[Tuple[Set[str], Set[str]]] = [
    # Running / endurance athletics
    ({"marathon", "half", "full", "10k", "5k", "race", "triathlon", "ironman", "run", "running", "jog", "jogging", "runner", "sprint"},
     {"run", "runs", "running", "jog", "jogging", "mile", "miles", "km", "track", "treadmill"}),
    # Swimming
    ({"swim", "swimming", "swimmer", "triathlon", "ironman", "meet", "aquatic"},
     {"swim", "swimming", "laps", "pool"}),
    # Cycling
    ({"cycle", "cycling", "cyclist", "bike", "biking", "triathlon", "century", "tour", "velodrome"},
     {"cycle", "cycling", "bike", "biking", "ride", "riding", "spin"}),
    # Strength / gym fitness
    ({"bodybuilding", "powerlifting", "crossfit", "physique", "weight", "press", "squat", "deadlift"},
     {"gym", "workout", "workouts", "lift", "lifting", "weights", "exercise"}),
    # Learning / academic
    ({"exam", "degree", "graduation", "thesis", "dissertation", "certification", "license"},
     {"study", "studying", "read", "reading", "practice", "revise", "course"}),
    # Writing / publishing
    ({"book", "novel", "memoir", "paper", "manuscript", "publish", "publishing"},
     {"write", "writing", "draft", "words"}),
    # Mindfulness / mental wellbeing
    ({"retreat", "mindfulness", "vipassana", "zen"},
     {"meditate", "meditation", "breath", "journal", "journaling", "yoga"}),
]


def is_supporting_habit_of_milestone(g1: CandidateGoal, g2: CandidateGoal) -> bool:
    """Determine whether one goal is a supporting habit for a more specific milestone ambition.

    Conservative Cross-Type Subsumption Criteria:
    1. Exactly one goal is 'milestone' and the other is 'habit'.
    2. Context linkage: Shared grounded evidence messages OR shared conversation chat.
       Goals across different conversation chats without shared evidence are strictly preserved as independent.
    3. Area compatibility: Areas must be identical or share the health/self athletic boundary.
    4. Semantic relationship: The habit must represent an activity or training routine
       directly supporting the milestone (shared activity stems, explicit training relationship,
       or domain activity linkage), rather than an independent ambition.
    """
    types = {g1.type, g2.type}
    if types != {"milestone", "habit"}:
        return False

    milestone = g1 if g1.type == "milestone" else g2
    habit = g2 if g1.type == "milestone" else g1

    # 1. Provenance / context linkage: Overlapping evidence OR same conversation chat
    has_overlapping_evidence = bool(set(milestone.evidence) & set(habit.evidence))
    has_same_chat = bool(set(milestone.source_chats) & set(habit.source_chats))
    if not (has_overlapping_evidence or has_same_chat):
        return False

    # 2. Area compatibility
    if milestone.area != habit.area and not ({milestone.area, habit.area} == {"health", "self"}):
        return False

    # 3. Semantic relationship:
    # A. Shared substantive activity stems
    m_tokens = tokenize_goal_title(milestone.title)
    h_tokens = tokenize_goal_title(habit.title)
    m_stems = {_stem_word(t) for t in m_tokens} - FREQ_HABIT_WORDS
    h_stems = {_stem_word(t) for t in h_tokens} - FREQ_HABIT_WORDS

    shared_stems = m_stems & h_stems
    if shared_stems:
        return True

    # B. Check if habit explicitly represents training/preparation for the milestone
    if ("train" in h_stems or "prepar" in h_stems or "practic" in h_stems) and m_stems:
        return True

    # C. Domain activity relationship between milestone and supporting routine
    for m_domain_words, h_domain_words in ACTIVITY_DOMAINS:
        if (m_tokens & m_domain_words) and (h_tokens & h_domain_words):
            return True

    return False


def are_semantically_equivalent_goals(
    g1: CandidateGoal,
    g2: CandidateGoal,
) -> bool:
    """Determine whether two candidate goals represent the exact same underlying ambition.

    Evaluates Core Semantic Equivalence across Five Semantic Dimensions:
    1. Area: g1.area == g2.area (or compatible health/self boundary for supporting habits).
    2. Goal Type: g1.type == g2.type, OR conservative cross-type milestone-habit subsumption.
    3. Core Action: Compatible contextually resolved actions (resolve_core_action).
    4. Target Entity: Both entities must be explicitly known and equal (NO None == None).
    5. Contextual Stage: Both stages must be explicitly known and equal (NO None == None).

    Evaluated in tandem with Two Compatibility Gates:
    6. Horizon Compatibility: are_horizons_compatible(g1.horizon, g2.horizon, g1.type, g2.type).
    7. Metric Compatibility: are_metrics_compatible(m1, m2) under conservative cross-category policy.

    Semantic Decider Guarantee:
    Once all five core semantic dimensions and both compatibility gates pass with known,
    non-empty values, semantic equivalence is fully established. Lexical token overlap is
    NOT an independent rejection condition, allowing genuine natural-language paraphrases
    (e.g., 'Family trip to Coorg in December' and 'We want to visit Coorg in December',
    'Start running four times a week' and 'Begin running 4 times a week') to merge cleanly.

    If equivalence is uncertain or unknown in ANY dimension or gate, returns False
    to preserve separate goals (No None == None merging).
    """
    if g1.type != g2.type:
        if is_supporting_habit_of_milestone(g1, g2):
            return True
        return False

    if g1.area != g2.area:
        return False

    # 1. Horizon compatibility
    if not are_horizons_compatible(g1.horizon, g2.horizon, g1.type, g2.type):
        return False

    # 2. Metric compatibility (cross-category conservative policy)
    m1 = extract_goal_metrics(g1.title)
    m2 = extract_goal_metrics(g2.title)
    if not are_metrics_compatible(m1, m2):
        return False

    # 3. Contextual action compatibility
    action1 = resolve_core_action(g1.title, g1.area)
    action2 = resolve_core_action(g2.title, g2.area)
    if action1 is None or action2 is None or action1 != action2:
        return False

    # 4. Target entity compatibility (MUST BE KNOWN AND EQUAL, NO None == None)
    entity1 = extract_target_entity(g1.title, g1.area)
    entity2 = extract_target_entity(g2.title, g2.area)
    if entity1 is None or entity2 is None or entity1 != entity2:
        return False

    # 5. Contextual stage compatibility (MUST BE KNOWN AND EQUAL, NO None == None)
    stage1 = extract_contextual_stage(g1.title)
    stage2 = extract_contextual_stage(g2.title)
    if stage1 is None or stage2 is None or stage1 != stage2:
        return False

    return True


def deduplicate_goals(
    candidates: List[CandidateGoal],
    timestamp_index: Optional[Dict[str, Union[int, float]]] = None,
) -> List[CandidateGoal]:
    """Safely deduplicate candidate goals, returning new CandidateGoal instances
    with combined evidence ordered chronologically. Never mutates input instances.
    Selects canonical horizon deterministically.

    Operates strictly across candidate goals prior to seed reconciliation (all inputs have is_seed=False).
    """
    merged: List[CandidateGoal] = []

    for cand in candidates:
        matched_idx = -1
        for idx, existing in enumerate(merged):
            if are_semantically_equivalent_goals(cand, existing):
                matched_idx = idx
                break

        if matched_idx >= 0:
            existing = merged[matched_idx]
            combined_evidence = union_evidence(
                existing.evidence, cand.evidence, timestamp_index=timestamp_index
            )
            combined_chats = sorted(set(existing.source_chats) | set(cand.source_chats))

            # Cross-type subsumption: prefer the more specific milestone over a supporting habit
            if existing.type == "milestone" and cand.type == "habit":
                canonical_type = "milestone"
                canonical_title = existing.title
                canonical_horizon = existing.horizon
                canonical_area = existing.area if existing.area == "health" or cand.area != "health" else cand.area
            elif existing.type == "habit" and cand.type == "milestone":
                canonical_type = "milestone"
                canonical_title = cand.title
                canonical_horizon = cand.horizon
                canonical_area = cand.area if cand.area == "health" or existing.area != "health" else existing.area
            else:
                canonical_type = existing.type
                canonical_title = existing.title if len(existing.title) >= len(cand.title) else cand.title
                canonical_horizon = select_canonical_horizon(existing, cand)
                canonical_area = existing.area

            canonical_conf = max(
                validate_confidence(existing.confidence),
                validate_confidence(cand.confidence),
            )
            merged[matched_idx] = CandidateGoal(
                title=canonical_title,
                area=canonical_area,
                type=canonical_type,
                horizon=canonical_horizon,
                evidence=combined_evidence,
                confidence=canonical_conf,
                source_chats=combined_chats,
                is_seed=False,
            )
        else:
            merged.append(
                CandidateGoal(
                    title=cand.title,
                    area=cand.area,
                    type=cand.type,
                    horizon=cand.horizon,
                    evidence=list(cand.evidence),
                    confidence=validate_confidence(cand.confidence),
                    source_chats=list(cand.source_chats),
                    is_seed=False,
                )
            )

    return merged


def are_seed_and_candidate_equivalent(
    seed_item: Dict[str, Any],
    cand: CandidateGoal,
) -> bool:
    """Strict semantic check between a seed goal and an extracted candidate goal.

    Requires explicit compatibility across the Five Core Semantic Dimensions
    (Area, Goal Type, Core Action, Target Entity, Contextual Stage)
    evaluated in tandem with the Metric Compatibility Gate.

    NOTE ON HORIZON DECOUPLING:
    Seed reconciliation does NOT reject semantically equivalent candidates on horizon differences.
    Seed horizon metadata in seed.json is authoritative and adopted upon match in reconcile_seed_goals().
    Candidate-to-candidate horizon compatibility remains strictly enforced in deduplicate_goals().

    Both target entity and contextual stage MUST be known and equal (NO None == None).
    Once all five semantic dimensions and metric compatibility pass strictly,
    seed equivalence is established without surface-token rejection hurdles.
    """
    s_area = seed_item.get("area", "")
    s_type = seed_item.get("type", "")
    if s_area != cand.area or s_type != cand.type:
        return False

    s_title = seed_item.get("title", "")

    # 1. Metric compatibility
    m_seed = extract_goal_metrics(s_title)
    m_cand = extract_goal_metrics(cand.title)
    if not are_metrics_compatible(m_seed, m_cand):
        return False

    # 2. Contextual action compatibility
    action_seed = resolve_core_action(s_title, s_area)
    action_cand = resolve_core_action(cand.title, cand.area)
    if action_seed is None or action_cand is None or action_seed != action_cand:
        return False

    # 3. Target entity compatibility (MUST BE KNOWN AND EQUAL, NO None == None)
    entity_seed = extract_target_entity(s_title, s_area)
    entity_cand = extract_target_entity(cand.title, cand.area)
    if entity_seed is None or entity_cand is None or entity_seed != entity_cand:
        return False

    # 4. Contextual stage compatibility (MUST BE KNOWN AND EQUAL, NO None == None)
    stage_seed = extract_contextual_stage(s_title)
    stage_cand = extract_contextual_stage(cand.title)
    if stage_seed is None or stage_cand is None or stage_seed != stage_cand:
        return False

    return True


def reconcile_seed_goals(
    extracted_goals: List[CandidateGoal],
    seed_goals: List[Dict[str, Any]],
    timestamp_index: Optional[Dict[str, Union[int, float]]] = None,
) -> List[CandidateGoal]:
    """Reconcile candidate goals with foundational seed goals without confidence inflation.

    Pipeline Order & Horizon Authority:
    - Reconciles candidate goals after deduplicate_goals has completed.
    - Seed metadata (title, area, type, horizon) is AUTHORITATIVE for schema alignment
      when genuine conversational evidence matches the seed goal.
    - Extracted candidates supply the genuine verified evidence and original confidence.
    - Unmatched seed goals are DISCARDED (never emitted without genuine conversation evidence).
    - Unmatched candidate goals are PRESERVED as distinct goals.
    """
    reconciled: List[CandidateGoal] = []
    matched_candidate_indices: Set[int] = set()

    for seed_item in seed_goals:
        s_title = seed_item.get("title", "").strip()
        s_area = seed_item.get("area", "")
        s_type = seed_item.get("type", "milestone")
        s_horizon = seed_item.get("horizon", "1yr")

        matching_evidence: List[str] = []
        matching_chats: List[str] = []
        matched_confidences: List[float] = []
        matched_for_this_seed: List[int] = []

        for idx, cand in enumerate(extracted_goals):
            if idx in matched_candidate_indices:
                continue
            if are_seed_and_candidate_equivalent(seed_item, cand):
                matching_evidence = union_evidence(
                    matching_evidence, cand.evidence, timestamp_index=timestamp_index
                )
                matching_chats = sorted(set(matching_chats) | set(cand.source_chats))
                matched_confidences.append(validate_confidence(cand.confidence))
                matched_for_this_seed.append(idx)

        # Emit seed goal ONLY IF genuine conversation evidence was matched
        if matching_evidence:
            reconciled_conf = (
                max(matched_confidences) if matched_confidences else 0.5
            )
            reconciled.append(
                CandidateGoal(
                    title=s_title,
                    area=s_area,
                    type=s_type,
                    horizon=s_horizon,
                    evidence=matching_evidence,
                    confidence=reconciled_conf,  # Preserves candidate confidence; never inflated to 1.0
                    source_chats=matching_chats,
                    is_seed=True,
                )
            )
            matched_candidate_indices.update(matched_for_this_seed)

    # Append all extracted candidates that were not matched to any seed goal
    for idx, cand in enumerate(extracted_goals):
        if idx not in matched_candidate_indices:
            if cand.evidence:
                reconciled.append(cand)

    return reconciled


def validate_and_filter_goal_evidence(
    goal: CandidateGoal,
    dataset_messages_by_id: Dict[str, Dict[str, Any]],
) -> Optional[CandidateGoal]:
    """Validate and ground a candidate goal's evidence against the dataset message index.

    Validation Contract:
    1. Every evidence ID must exist in dataset_messages_by_id.
    2. Invalid or ungrounded IDs are discarded.
    3. Duplicate IDs are eliminated while preserving order.
    4. Cross-chat relationships are derived strictly from the verified messages' actual chat_id attributes.
    5. If a goal has zero verified evidence IDs, it is discarded (returns None).
    """
    seen: Set[str] = set()
    verified_evidence: List[str] = []
    verified_chats: Set[str] = set()

    for mid in goal.evidence:
        if mid in dataset_messages_by_id and mid not in seen:
            seen.add(mid)
            verified_evidence.append(mid)
            chat_id = dataset_messages_by_id[mid].get("chat_id")
            if isinstance(chat_id, str) and chat_id.strip() != "":
                verified_chats.add(chat_id)

    if not verified_evidence:
        return None  # Discard ungrounded goal

    return CandidateGoal(
        title=goal.title,
        area=goal.area,
        type=goal.type,
        horizon=goal.horizon,
        evidence=verified_evidence,
        confidence=validate_confidence(goal.confidence),
        source_chats=sorted(verified_chats),
        is_seed=goal.is_seed,
    )


def rank_and_clamp_goals(
    goals: List[CandidateGoal],
    dataset_messages_by_id: Dict[str, Dict[str, Any]],
    max_goals: int = 10,
) -> List[Dict[str, Any]]:
    """Rank candidate goals deterministically using evidence-first precedence and clamp to schema limit (<= 10).
    Emits schema-compliant dictionaries containing strictly the 5 official fields.

    Ranking Order:
    1. Evidence validity — invalid, duplicate, or ungrounded IDs are eliminated.
    2. Independent support — cross-chat confirmation (>= 2 verified chats) and distinct verified messages (>= 2).
    3. Interpretation confidence — model's interpreted confidence score (descending).
    4. Verified evidence count — total number of distinct verified evidence messages (descending).
    5. Deterministic title tie-break — alphabetical ascending on title.

    CRITICAL RULE:
    Seed status does NOT appear in the sort tuple and does NOT independently improve ranking.
    """
    validated_goals: List[CandidateGoal] = []
    for g in goals:
        v_goal = validate_and_filter_goal_evidence(g, dataset_messages_by_id)
        if v_goal is not None:
            validated_goals.append(v_goal)

    def _rank_sort_tuple(g: CandidateGoal) -> Tuple[int, int, float, int, str]:
        cross_chat = 1 if len(set(g.source_chats)) >= 2 else 0
        multi_msg = 1 if len(g.evidence) >= 2 else 0
        conf = validate_confidence(g.confidence, default=0.5)
        ev_count = len(g.evidence)
        return (cross_chat, multi_msg, conf, ev_count, g.title)

    sorted_goals = sorted(
        validated_goals,
        key=lambda g: (
            -_rank_sort_tuple(g)[0],
            -_rank_sort_tuple(g)[1],
            -_rank_sort_tuple(g)[2],
            -_rank_sort_tuple(g)[3],
            _rank_sort_tuple(g)[4],
        ),
    )

    effective_limit = max(0, min(10, max_goals))
    clamped = sorted_goals[:effective_limit]
    return [g.to_schema_dict() for g in clamped]
