"""Comprehensive unit tests for the goal extraction resolver (src/goals/resolver.py).

Covers all scenarios defined in Step 6.1 Goal Extraction Pipeline Design:
1. Metric Extraction & Contextual Disambiguation
2. Contextual Action Resolution
3. Experience Goal Resolution
4. Habit Goal Resolution
5. Semantic Dimensions & Lifecycle Stages (Semantic Decider & No None == None Merging)
6. Horizon Compatibility & Canonical Horizon Selection
7. Timestamp & Confidence Validation
8. Seed Goal Reconciliation
9. Evidence Grounding & Deterministic Ranking
10. Chronological Evidence Ordering
11. Input Object Immutability & Mutation Safety
"""

import math
from typing import Any, Dict, List
import pytest

from src.goals.models import CandidateGoal, GoalMetrics
from src.goals.resolver import (
    are_horizons_compatible,
    are_metrics_compatible,
    are_seed_and_candidate_equivalent,
    are_semantically_equivalent_goals,
    deduplicate_goals,
    extract_contextual_stage,
    extract_goal_metrics,
    extract_target_entity,
    from_interpreted_goal,
    is_valid_timestamp,
    rank_and_clamp_goals,
    reconcile_seed_goals,
    resolve_core_action,
    select_canonical_horizon,
    tokenize_goal_title,
    union_evidence,
    validate_and_filter_goal_evidence,
    validate_confidence,
)
from src.interpretation.records import InterpretedGoal


# ============================================================================
# 1. Metric Extraction & Contextual Disambiguation Tests
# ============================================================================


def test_metric_extraction_plain_integers_and_decimals():
    """Verify extraction of unadorned numeric values, ignoring calendar years."""
    m1 = extract_goal_metrics("Read 12 books")
    assert m1.numbers == {12.0}
    assert m1.quantities_with_units == set()
    assert m1.distances == set()

    m2 = extract_goal_metrics("Lose 4.5 kg weight")
    assert m2.numbers == {4.5}

    # Calendar year 2026 ignored from numbers
    m3 = extract_goal_metrics("Achieve milestone in 2026")
    assert m3.numbers == set()


def test_metric_extraction_indian_quantities_compact_and_spaced():
    """Verify Indian unit normalization (Lakhs -> l, Crores -> cr)."""
    m1 = extract_goal_metrics("Save 20L for house")
    assert m1.quantities_with_units == {"20l"}
    assert m1.numbers == set()

    m2 = extract_goal_metrics("Budget 20 L for down payment")
    assert m2.quantities_with_units == {"20l"}

    m3 = extract_goal_metrics("Reach 5 Cr ARR")
    assert m3.quantities_with_units == {"5cr"}

    m4 = extract_goal_metrics("Raise 5.5 Crore funding")
    assert m4.quantities_with_units == {"5.5cr"}


def test_metric_extraction_western_quantities_millions_compact_and_spaced():
    """Verify Western large unit normalization (M -> m, B -> b)."""
    m1 = extract_goal_metrics("Reach 10M ARR")
    assert m1.quantities_with_units == {"10m"}
    assert m1.distances == set()
    assert m1.ambiguous_metrics == set()
    assert m1.numbers == set()

    m2 = extract_goal_metrics("Target 10 M users")
    assert m2.quantities_with_units == {"10m"}

    m3 = extract_goal_metrics("Raise 1.5 Million funding")
    assert m3.quantities_with_units == {"1.5m"}

    m4 = extract_goal_metrics("Hit 2B valuation")
    assert m4.quantities_with_units == {"2b"}


def test_metric_extraction_k_with_financial_context_compact_and_spaced():
    """Verify financial and commercial contexts classify 10k/10 k strictly as quantities."""
    m1 = extract_goal_metrics("Save 10k")
    assert m1.quantities_with_units == {"10k"}
    assert m1.distances == set()
    assert m1.ambiguous_metrics == set()

    m2 = extract_goal_metrics("Save 10 k")
    assert m2.quantities_with_units == {"10k"}

    m3 = extract_goal_metrics("Reach 10k revenue")
    assert m3.quantities_with_units == {"10k"}

    m4 = extract_goal_metrics("Acquire 10 k customers")
    assert m4.quantities_with_units == {"10k"}

    m5 = extract_goal_metrics("Earn 10k MRR")
    assert m5.quantities_with_units == {"10k"}


def test_metric_extraction_k_with_athletic_context_compact_and_spaced():
    """Verify athletic collocations classify 10k/10 k/10 km strictly as athletic distances."""
    m1 = extract_goal_metrics("Run a 10k race")
    assert m1.distances == {"10k"}
    assert m1.quantities_with_units == set()
    assert m1.ambiguous_metrics == set()

    m2 = extract_goal_metrics("10 k race")
    assert m2.distances == {"10k"}

    m3 = extract_goal_metrics("Run 10k")
    assert m3.distances == {"10k"}

    m4 = extract_goal_metrics("Run 10 k")
    assert m4.distances == {"10k"}

    m5 = extract_goal_metrics("10 km race")
    assert m5.distances == {"10k"}


def test_metric_extraction_standalone_k_without_context_ambiguous():
    """Verify ungrounded standalone 10k/10 k is classified as ambiguous."""
    for title in ["10k by December", "10 k by December", "Target 10k", "Achieve 10 k"]:
        m = extract_goal_metrics(title)
        assert m.ambiguous_metrics == {"10k"}, f"Failed on title: {title}"
        assert m.quantities_with_units == set()
        assert m.distances == set()
        assert m.numbers == set()


def test_metric_extraction_conflicting_k_context_ambiguous():
    """Verify title with conflicting athletic and commercial context preserves ambiguity."""
    m = extract_goal_metrics("Run 10k revenue")
    assert m.ambiguous_metrics == {"10k"}
    assert m.distances == set()
    assert m.quantities_with_units == set()


def test_metric_extraction_percentages_compact_and_spaced():
    """Verify percentage extraction with compact and spaced formatting."""
    m1 = extract_goal_metrics("Increase sales by 50%")
    assert m1.percentages == {"50%"}
    assert m1.numbers == set()

    m2 = extract_goal_metrics("Improve efficiency by 50 %")
    assert m2.percentages == {"50%"}

    m3 = extract_goal_metrics("Growth of 12.5%")
    assert m3.percentages == {"12.5%"}


def test_metric_extraction_excludes_captured_units_from_plain_numbers():
    """Verify spans captured by units are not re-extracted as plain numbers."""
    m = extract_goal_metrics("Save 20L and hire 5 engineers")
    assert m.quantities_with_units == {"20l"}
    assert m.numbers == {5.0}


def test_metric_compatibility_run_10k_race_vs_save_10k_fails():
    """Verify distance vs quantity metric category mismatch strictly fails compatibility."""
    m_athletic = extract_goal_metrics("Run a 10k race")
    m_financial = extract_goal_metrics("Save 10k")
    assert are_metrics_compatible(m_athletic, m_financial) is False


def test_metric_compatibility_reach_10k_revenue_vs_run_10k_fails():
    """Verify quantity vs distance metric category mismatch strictly fails compatibility."""
    m_rev = extract_goal_metrics("Reach 10k revenue")
    m_run = extract_goal_metrics("Run 10k")
    assert are_metrics_compatible(m_rev, m_run) is False


def test_metric_compatibility_10_km_race_vs_10k_customers_fails():
    """Verify distance vs quantity count mismatch strictly fails compatibility."""
    m_race = extract_goal_metrics("10 km race")
    m_cust = extract_goal_metrics("10k customers")
    assert are_metrics_compatible(m_race, m_cust) is False


def test_metric_extraction_and_compatibility_dual_metric_title():
    """Verify titles with multiple independent metrics extract cleanly and evaluate compatibility."""
    m_dual = extract_goal_metrics("Save 20k and run 10k race")
    assert m_dual.quantities_with_units == {"20k"}
    assert m_dual.distances == {"10k"}

    m_race_only = extract_goal_metrics("Run 10k race")
    # Category presence mismatch
    assert are_metrics_compatible(m_dual, m_race_only) is False

    m_dual_match = extract_goal_metrics("Save 20k and finish 10k race")
    assert are_metrics_compatible(m_dual, m_dual_match) is True


def test_metric_compatibility_rejects_ambiguous_standalone_10k():
    """Verify ambiguous metrics block compatibility against any metric."""
    m_ambig1 = extract_goal_metrics("10k by December")
    m_ambig2 = extract_goal_metrics("Achieve 10k")
    m_quant = extract_goal_metrics("Save 10k")
    m_dist = extract_goal_metrics("Run 10k")

    assert are_metrics_compatible(m_ambig1, m_ambig2) is False
    assert are_metrics_compatible(m_ambig1, m_quant) is False
    assert are_metrics_compatible(m_ambig1, m_dist) is False


def test_metric_compatibility_rejects_quantity_vs_plain_number():
    """Verify quantity with unit cannot match plain number."""
    m_qty = GoalMetrics(quantities_with_units={"20l"})
    m_num = GoalMetrics(numbers={20.0})
    assert are_metrics_compatible(m_qty, m_num) is False


def test_metric_compatibility_rejects_percentage_vs_plain_number():
    """Verify percentage cannot match plain number."""
    m_pct = GoalMetrics(percentages={"50%"})
    m_num = GoalMetrics(numbers={50.0})
    assert are_metrics_compatible(m_pct, m_num) is False


def test_metric_compatibility_empty_metrics():
    """Verify empty metric comparison rules."""
    m_empty1 = GoalMetrics()
    m_empty2 = GoalMetrics()
    m_pop = GoalMetrics(numbers={10.0})

    assert are_metrics_compatible(m_empty1, m_empty2) is True
    assert are_metrics_compatible(m_empty1, m_pop) is False
    assert are_metrics_compatible(m_pop, m_empty1) is False


# ============================================================================
# 2. Contextual Action Resolution Tests
# ============================================================================


def test_action_resolver_run_athletic_with_event_marker():
    """Verify run resolves to athletic when supported by athletic event marker."""
    assert resolve_core_action("Run the half marathon in January", "health") == "athletic"
    assert resolve_core_action("Run 10k race", "health") == "athletic"


def test_action_resolver_run_company_and_business():
    """Verify run resolves to commercial_growth only with explicit business tokens."""
    assert resolve_core_action("Run a company in tech", "work") == "commercial_growth"
    assert resolve_core_action("Run a business", "work") == "commercial_growth"


def test_action_resolver_run_tests_work_unresolved():
    """Verify generic technical task 'Run tests' returns None even with area work."""
    assert resolve_core_action("Run tests", "work") is None


def test_action_resolver_run_daily_unresolved():
    """Verify 'Run daily' with area health returns None (lacks event or habit routine)."""
    assert resolve_core_action("Run daily", "health") is None


def test_action_resolver_run_generic_task_unresolved():
    """Verify generic execution 'Run it tomorrow' returns None."""
    assert resolve_core_action("Run it tomorrow", "self") is None


def test_action_resolver_participate_in_race_athletic():
    """Verify participation verb with race context resolves to athletic."""
    assert resolve_core_action("Participate in a race", "health") == "athletic"
    assert resolve_core_action("Participate in January half marathon", "health") == "athletic"


def test_action_resolver_race_health_unresolved():
    """Verify bare 'Race' with area health lacking independent event marker returns None."""
    assert resolve_core_action("Race", "health") is None


def test_action_resolver_racing_every_morning_health_unresolved():
    """Verify 'Racing every morning' with area health lacking event context returns None."""
    assert resolve_core_action("Racing every morning", "health") is None


def test_action_resolver_race_10k_athletic():
    """Verify 'Race a 10k' with independent distance marker resolves to athletic."""
    assert resolve_core_action("Race a 10k", "health") == "athletic"


def test_action_resolver_buy_groceries_home_unresolved():
    """Verify generic errand 'Buy groceries' with area home returns None."""
    assert resolve_core_action("Buy groceries", "home") is None


def test_action_resolver_buy_house_property():
    """Verify 'Buy a house' resolves to property."""
    assert resolve_core_action("Buy a house", "home") == "property"


def test_action_resolver_travel_experience():
    """Verify travel and vacation ambitions resolve to travel_experience."""
    assert resolve_core_action("Family trip to Coorg", "experiences") == "travel_experience"
    assert resolve_core_action("We want to visit Coorg", "experiences") == "travel_experience"


def test_action_resolver_operational_booking():
    """Verify operational logistics resolve to operational_booking."""
    assert resolve_core_action("Book flight tickets to Coorg", "experiences") == "operational_booking"
    assert resolve_core_action("Reserve hotel in Coorg", "experiences") == "operational_booking"


def test_action_resolver_plural_runs_fitness_habit():
    """Verify plural inflected 'runs' resolves to fitness_habit when intentional."""
    assert resolve_core_action("Maintain four runs a week", "health") == "fitness_habit"


def test_action_resolver_plural_runs_milestone_prep():
    """Verify plural runs with milestone event marker resolves to athletic."""
    assert resolve_core_action("Doing four runs a week to train for the marathon", "health") == "athletic"


def test_action_resolver_habit_fitness():
    """Verify intentional habit adoption resolves to fitness_habit."""
    assert resolve_core_action("Start running four times a week", "health") == "fitness_habit"


def test_action_resolver_habit_mindfulness():
    """Verify intentional mindfulness habit resolves to mindfulness_habit."""
    assert resolve_core_action("Start daily meditation", "health") == "mindfulness_habit"
    assert resolve_core_action("Maintain a meditation routine", "health") == "mindfulness_habit"


def test_action_resolver_meditation_requires_intentional_intent():
    """Verify meditation requires explicit volitional intent; routine alone or status report is None."""
    assert resolve_core_action("I meditate every morning", "health") is None
    assert resolve_core_action("I have a meditation routine", "health") is None
    assert resolve_core_action("Start daily meditation", "health") == "mindfulness_habit"
    assert resolve_core_action("Maintain a meditation routine", "health") == "mindfulness_habit"
    assert resolve_core_action("Keep up my meditation routine", "health") == "mindfulness_habit"
    assert resolve_core_action("Start a meditation routine", "health") == "mindfulness_habit"


def test_action_resolver_running_requires_intentional_intent():
    """Verify running habit requires explicit volitional intent; routine alone or status report is None."""
    assert resolve_core_action("I run four times a week", "health") is None
    assert resolve_core_action("My running routine is four days a week", "health") is None
    assert resolve_core_action("Maintain four runs a week", "health") == "fitness_habit"
    assert resolve_core_action("Start running four times a week", "health") == "fitness_habit"
    assert resolve_core_action("Continue my running routine", "health") == "fitness_habit"
    assert resolve_core_action("Build a running routine", "health") == "fitness_habit"


def test_action_resolver_routine_alone_not_habit_intent():
    """Verify 'routine' alone is not sufficient evidence of habit intent without commitment verbs."""
    assert resolve_core_action("I have a meditation routine", "health") is None
    assert resolve_core_action("My running routine is four days a week", "health") is None


def test_action_resolver_explicit_routine_maintenance_constructions():
    """Verify explicit maintenance verbs with routine resolve to habit actions."""
    assert resolve_core_action("Continue my running routine", "health") == "fitness_habit"
    assert resolve_core_action("Keep up my meditation routine", "health") == "mindfulness_habit"
    assert resolve_core_action("Maintain a meditation routine", "health") == "mindfulness_habit"


def test_action_resolver_explicit_routine_adoption_constructions():
    """Verify explicit adoption verbs with routine resolve to habit actions."""
    assert resolve_core_action("Start a meditation routine", "health") == "mindfulness_habit"
    assert resolve_core_action("Build a running routine", "health") == "fitness_habit"


# ============================================================================
# 3. Experience Goal Resolution Tests
# ============================================================================


def test_experience_equivalent_travel_goals_merge():
    """Verify synonymous travel experience goals merge despite low lexical similarity."""
    g1 = CandidateGoal(
        title="Family trip to Coorg in December",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M01"],
    )
    g2 = CandidateGoal(
        title="We want to visit Coorg in December",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M02"],
    )

    assert resolve_core_action(g1.title, g1.area) == "travel_experience"
    assert resolve_core_action(g2.title, g2.area) == "travel_experience"
    assert extract_target_entity(g1.title, g1.area) == "destination_coorg"
    assert extract_target_entity(g2.title, g2.area) == "destination_coorg"
    assert extract_contextual_stage(g1.title) == "stage_execution"
    assert extract_contextual_stage(g2.title) == "stage_execution"

    assert are_semantically_equivalent_goals(g1, g2) is True

    merged = deduplicate_goals([g1, g2])
    assert len(merged) == 1
    assert "M01" in merged[0].evidence
    assert "M02" in merged[0].evidence


def test_experience_distinct_destinations_preserved_separately():
    """Verify experience goals with different destinations are preserved separately."""
    g1 = CandidateGoal(
        title="Family trip to Coorg in December",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M01"],
    )
    g2 = CandidateGoal(
        title="Family trip to Paris in December",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M02"],
    )

    assert extract_target_entity(g1.title, g1.area) == "destination_coorg"
    assert extract_target_entity(g2.title, g2.area) == "destination_paris"
    assert are_semantically_equivalent_goals(g1, g2) is False

    merged = deduplicate_goals([g1, g2])
    assert len(merged) == 2


def test_experience_ambition_vs_operational_booking_preserved_separately():
    """Verify experience travel ambition is kept separate from tactical booking errand."""
    g_exp = CandidateGoal(
        title="Family trip to Coorg in December",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M01"],
    )
    g_book = CandidateGoal(
        title="Book flight tickets to Coorg",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M02"],
    )

    assert resolve_core_action(g_exp.title, g_exp.area) == "travel_experience"
    assert resolve_core_action(g_book.title, g_book.area) == "operational_booking"
    assert extract_contextual_stage(g_exp.title) == "stage_execution"
    assert extract_contextual_stage(g_book.title) == "stage_transaction"

    assert are_semantically_equivalent_goals(g_exp, g_book) is False


def test_experience_multi_word_destination_new_york():
    """Verify multi-word destination New York extracts cleanly without partial destination_new."""
    assert extract_target_entity("Family trip to New York", "experiences") == "destination_new_york"
    assert extract_target_entity("We want to visit New York", "experiences") == "destination_new_york"


def test_experience_single_word_destinations_continue_to_work():
    """Verify existing single-word destinations continue to work as expected."""
    assert extract_target_entity("Family trip to Coorg", "experiences") == "destination_coorg"
    assert extract_target_entity("Trip to Paris", "experiences") == "destination_paris"
    assert extract_target_entity("Vacation in Goa", "experiences") == "destination_goa"
    assert extract_target_entity("Holiday in Tokyo", "experiences") == "destination_tokyo"


def test_experience_distinct_destinations_remain_distinct_including_multi_word():
    """Verify multi-word and single-word destinations remain distinct from each other."""
    g_ny = CandidateGoal(
        title="Family trip to New York",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M01"],
    )
    g_paris = CandidateGoal(
        title="Family trip to Paris",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M02"],
    )
    g_coorg = CandidateGoal(
        title="Family trip to Coorg",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M03"],
    )
    assert extract_target_entity(g_ny.title, g_ny.area) == "destination_new_york"
    assert extract_target_entity(g_paris.title, g_paris.area) == "destination_paris"
    assert extract_target_entity(g_coorg.title, g_coorg.area) == "destination_coorg"

    assert are_semantically_equivalent_goals(g_ny, g_paris) is False
    assert are_semantically_equivalent_goals(g_ny, g_coorg) is False
    assert are_semantically_equivalent_goals(g_paris, g_coorg) is False


def test_experience_equivalent_new_york_travel_goals_merge():
    """Verify semantically equivalent New York travel descriptions pass entity check and merge cleanly."""
    g1 = CandidateGoal(
        title="Family trip to New York",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M01"],
    )
    g2 = CandidateGoal(
        title="We want to visit New York",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M02"],
    )

    assert extract_target_entity(g1.title, g1.area) == "destination_new_york"
    assert extract_target_entity(g2.title, g2.area) == "destination_new_york"
    assert resolve_core_action(g1.title, g1.area) == "travel_experience"
    assert resolve_core_action(g2.title, g2.area) == "travel_experience"
    assert extract_contextual_stage(g1.title) == "stage_execution"
    assert extract_contextual_stage(g2.title) == "stage_execution"

    assert are_semantically_equivalent_goals(g1, g2) is True

    merged = deduplicate_goals([g1, g2])
    assert len(merged) == 1
    assert merged[0].title in {"Family trip to New York", "We want to visit New York"}
    assert "M01" in merged[0].evidence
    assert "M02" in merged[0].evidence


# ============================================================================
# 4. Habit Goal Resolution Tests
# ============================================================================


def test_habit_adoption_equivalent_goals_merge():
    """Verify synonymous habit adoption goals merge across start and begin."""
    h1 = CandidateGoal(
        title="Start running four times a week",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M01"],
    )
    h2 = CandidateGoal(
        title="Begin running 4 times a week",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M02"],
    )

    assert resolve_core_action(h1.title, h1.area) == "fitness_habit"
    assert resolve_core_action(h2.title, h2.area) == "fitness_habit"
    assert extract_target_entity(h1.title, h1.area) == "habit_running"
    assert extract_target_entity(h2.title, h2.area) == "habit_running"
    assert extract_contextual_stage(h1.title) == "stage_initiation"
    assert extract_contextual_stage(h2.title) == "stage_initiation"

    assert are_semantically_equivalent_goals(h1, h2) is True
    merged = deduplicate_goals([h1, h2])
    assert len(merged) == 1


def test_habit_adoption_vs_maintenance_preserved_separately():
    """Verify habit adoption (stage_initiation) and maintenance (stage_maintenance) remain separate."""
    h_adopt = CandidateGoal(
        title="Start running four times a week",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M01"],
    )
    h_maint = CandidateGoal(
        title="Maintain four runs a week",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M02"],
    )

    assert extract_contextual_stage(h_adopt.title) == "stage_initiation"
    assert extract_contextual_stage(h_maint.title) == "stage_maintenance"
    assert are_semantically_equivalent_goals(h_adopt, h_maint) is False


def test_habit_maintenance_vs_milestone_preparation_preserved_separately():
    """Verify ongoing habit maintenance is preserved separately from milestone event preparation."""
    g_habit = CandidateGoal(
        title="Maintain four runs a week",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M01"],
    )
    g_prep = CandidateGoal(
        title="Doing four runs a week to train for the marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M02"],
    )

    assert resolve_core_action(g_habit.title, g_habit.area) == "fitness_habit"
    assert resolve_core_action(g_prep.title, g_prep.area) == "athletic"
    assert extract_target_entity(g_habit.title, g_habit.area) == "habit_running"
    assert extract_target_entity(g_prep.title, g_prep.area) == "athletic_event"
    assert extract_contextual_stage(g_habit.title) == "stage_maintenance"
    assert extract_contextual_stage(g_prep.title) == "stage_preparation"

    assert are_semantically_equivalent_goals(g_habit, g_prep) is False


def test_habit_resolvers_agreement_plural_runs_maintenance():
    """Verify unanimous resolver agreement for plural runs maintenance habit."""
    title = "Maintain four runs a week"
    assert resolve_core_action(title, "health") == "fitness_habit"
    assert extract_target_entity(title, "health") == "habit_running"
    assert extract_contextual_stage(title) == "stage_maintenance"


def test_habit_resolvers_agreement_plural_runs_milestone_prep():
    """Verify unanimous resolver agreement for plural runs milestone preparation."""
    title = "Doing four runs a week to train for the marathon"
    assert resolve_core_action(title, "health") == "athletic"
    assert extract_target_entity(title, "health") == "athletic_event"
    assert extract_contextual_stage(title) == "stage_preparation"


def test_habit_distinct_activities_preserved_separately():
    """Verify running habit and meditation habit are preserved separately."""
    h_med = CandidateGoal(
        title="Start daily meditation",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M01"],
    )
    h_run = CandidateGoal(
        title="Start running four times a week",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M02"],
    )
    assert are_semantically_equivalent_goals(h_med, h_run) is False


def test_habit_meditation_adoption_and_maintenance():
    """Verify meditation adoption and meditation maintenance are preserved separately."""
    h1 = CandidateGoal(
        title="Start daily meditation",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M01"],
    )
    h2 = CandidateGoal(
        title="Maintain a meditation routine",
        area="health",
        type="habit",
        horizon="1yr",
        evidence=["M02"],
    )
    assert extract_contextual_stage(h1.title) == "stage_initiation"
    assert extract_contextual_stage(h2.title) == "stage_maintenance"
    assert are_semantically_equivalent_goals(h1, h2) is False


def test_habit_descriptive_routine_running_excluded():
    """Verify descriptive status quo running report lacks action resolution."""
    assert resolve_core_action("I run four times a week", "health") is None


def test_habit_descriptive_routine_meditation_excluded():
    """Verify descriptive status quo meditation report lacks action resolution."""
    assert resolve_core_action("I meditate every morning", "health") is None


def test_habit_factual_routine_running_excluded():
    """Verify factual running routine description without commitment returns action=None and stage=None."""
    title = "My running routine is four days a week"
    assert resolve_core_action(title, "health") is None
    assert extract_contextual_stage(title) is None


def test_habit_factual_routine_meditation_excluded():
    """Verify factual meditation routine description without commitment returns action=None and stage=None."""
    title = "I have a meditation routine"
    assert resolve_core_action(title, "health") is None
    assert extract_contextual_stage(title) is None


def test_habit_routine_explicit_maintenance_constructions():
    """Verify explicit maintenance constructions resolve to stage_maintenance and habit actions."""
    t1 = "Maintain a meditation routine"
    assert resolve_core_action(t1, "health") == "mindfulness_habit"
    assert extract_contextual_stage(t1) == "stage_maintenance"

    t2 = "Continue my running routine"
    assert resolve_core_action(t2, "health") == "fitness_habit"
    assert extract_contextual_stage(t2) == "stage_maintenance"

    t3 = "Keep up my meditation routine"
    assert resolve_core_action(t3, "health") == "mindfulness_habit"
    assert extract_contextual_stage(t3) == "stage_maintenance"


def test_habit_routine_explicit_adoption_constructions():
    """Verify explicit adoption constructions resolve to stage_initiation and habit actions."""
    t1 = "Start a meditation routine"
    assert resolve_core_action(t1, "health") == "mindfulness_habit"
    assert extract_contextual_stage(t1) == "stage_initiation"

    t2 = "Build a running routine"
    assert resolve_core_action(t2, "health") == "fitness_habit"
    assert extract_contextual_stage(t2) == "stage_initiation"


# ============================================================================
# 5. Semantic Dimensions & Lifecycle Stages (Semantic Decider & Unknown-Stage Guard)
# ============================================================================


def test_deduplication_rejects_when_both_entities_none():
    """Verify goals lacking target entities return False under uncertainty guard (No None == None)."""
    g1 = CandidateGoal(
        title="Achieve great success this year",
        area="self",
        type="milestone",
        horizon="1yr",
        evidence=["M01"],
    )
    g2 = CandidateGoal(
        title="Do something amazing this year",
        area="self",
        type="milestone",
        horizon="1yr",
        evidence=["M02"],
    )
    assert extract_target_entity(g1.title, g1.area) is None
    assert extract_target_entity(g2.title, g2.area) is None
    assert are_semantically_equivalent_goals(g1, g2) is False


def test_deduplication_rejects_when_both_stages_none():
    """Verify goals lacking contextual stages return False under unknown-stage safety barrier."""
    g1 = CandidateGoal(
        title="January half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M01"],
    )
    g2 = CandidateGoal(
        title="January half marathon race",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M02"],
    )
    assert extract_contextual_stage(g1.title) is None
    assert extract_contextual_stage(g2.title) is None
    assert are_semantically_equivalent_goals(g1, g2) is False


def test_semantic_equivalence_no_lexical_rejection_on_valid_paraphrase():
    """Verify valid natural paraphrases merge cleanly without lexical threshold hurdles."""
    g1 = CandidateGoal(
        title="Family trip to Coorg in December",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M01"],
    )
    g2 = CandidateGoal(
        title="We want to visit Coorg in December",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M02"],
    )
    # Five semantic dimensions match
    assert are_semantically_equivalent_goals(g1, g2) is True


def test_athletic_milestone_merges_with_stage_execution():
    """Verify run and participate in half marathon merge under stage_execution."""
    g1 = CandidateGoal(
        title="Run the half marathon in January",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M01"],
    )
    g2 = CandidateGoal(
        title="Participate in January half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M02"],
    )

    assert extract_target_entity(g1.title, g1.area) == "athletic_event"
    assert extract_target_entity(g2.title, g2.area) == "athletic_event"
    assert extract_contextual_stage(g1.title) == "stage_execution"
    assert extract_contextual_stage(g2.title) == "stage_execution"
    assert resolve_core_action(g1.title, g1.area) == "athletic"
    assert resolve_core_action(g2.title, g2.area) == "athletic"

    assert are_semantically_equivalent_goals(g1, g2) is True


def test_athletic_preparation_vs_execution_separation():
    """Verify train for marathon (stage_preparation) is preserved separately from run marathon (stage_execution)."""
    g_prep = CandidateGoal(
        title="Train for the half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M01"],
    )
    g_exec = CandidateGoal(
        title="Run the half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M02"],
    )
    assert extract_contextual_stage(g_prep.title) == "stage_preparation"
    assert extract_contextual_stage(g_exec.title) == "stage_execution"
    assert are_semantically_equivalent_goals(g_prep, g_exec) is False


def test_ungrounded_phrase_lacking_stage_preserved_separately():
    """Verify ungrounded title lacking stage is preserved separately from stage-grounded goal."""
    g_ungrounded = CandidateGoal(
        title="January half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M01"],
    )
    g_exec = CandidateGoal(
        title="Run the half marathon in January",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M02"],
    )
    assert extract_contextual_stage(g_ungrounded.title) is None
    assert are_semantically_equivalent_goals(g_ungrounded, g_exec) is False


def test_deduplication_rejects_conflicting_entity_signals():
    """Verify conflicting entity signals return None and prevent deduplication."""
    title = "Hire 5 engineers and acquire 10 clients"
    assert extract_target_entity(title, "work") is None


def test_deduplication_rejects_conflicting_stage_signals():
    """Verify conflicting stage signals return None and prevent deduplication."""
    title = "Save for and buy a house"
    assert extract_contextual_stage(title) is None


def test_multi_word_phrase_matching_real_estate_and_close_on():
    """Verify boundary-preserved multi-word phrases match accurately."""
    assert extract_target_entity("Invest in real estate in Bangalore", "finances") == "residential_property"
    assert extract_contextual_stage("Close on the house next month") == "stage_transaction"


# ============================================================================
# 6. Horizon Compatibility & Canonical Horizon Selection Tests
# ============================================================================


def test_candidate_milestone_horizon_conflict_prevents_merge():
    """Verify milestone candidates with conflicting horizons are preserved separately."""
    c1 = CandidateGoal(
        title="Run the half marathon in January",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M01"],
    )
    c2 = CandidateGoal(
        title="Run the half marathon this year",
        area="health",
        type="milestone",
        horizon="1yr",
        evidence=["M02"],
    )
    assert are_horizons_compatible("3mo", "1yr", "milestone", "milestone") is False
    assert are_semantically_equivalent_goals(c1, c2) is False

    merged = deduplicate_goals([c1, c2])
    assert len(merged) == 2


def test_habit_compatible_adjacent_horizons_merge():
    """Verify adjacent habit horizons (3mo and 1yr) are compatible and merge."""
    assert are_horizons_compatible("3mo", "1yr", "habit", "habit") is True
    assert are_horizons_compatible("3mo", "3yr", "habit", "habit") is False


def test_canonical_horizon_selection_favors_greater_evidence_count():
    """Verify canonical horizon selection prioritizes the candidate with more evidence."""
    c1 = CandidateGoal(title="Habit", area="health", type="habit", horizon="3mo", evidence=["M01"])
    c2 = CandidateGoal(title="Habit", area="health", type="habit", horizon="1yr", evidence=["M02", "M03"])
    assert select_canonical_horizon(c1, c2) == "1yr"


def test_canonical_horizon_selection_favors_higher_confidence_on_tie():
    """Verify canonical horizon selection breaks evidence count ties using confidence."""
    c1 = CandidateGoal(title="Habit", area="health", type="habit", horizon="3mo", evidence=["M01"], confidence=0.9)
    c2 = CandidateGoal(title="Habit", area="health", type="habit", horizon="1yr", evidence=["M02"], confidence=0.7)
    assert select_canonical_horizon(c1, c2) == "3mo"


def test_canonical_horizon_selection_defaults_to_1yr_for_tied_habits():
    """Verify canonical horizon selection defaults to 1yr when evidence and confidence are tied."""
    c1 = CandidateGoal(title="Habit", area="health", type="habit", horizon="3mo", evidence=["M01"], confidence=0.8)
    c2 = CandidateGoal(title="Habit", area="health", type="habit", horizon="1yr", evidence=["M02"], confidence=0.8)
    assert select_canonical_horizon(c1, c2) == "1yr"


# ============================================================================
# 7. Timestamp & Confidence Validation Tests
# ============================================================================


def test_timestamp_validation_rejects_booleans():
    """Verify boolean True and False are strictly rejected as valid timestamps."""
    assert is_valid_timestamp(True) is False
    assert is_valid_timestamp(False) is False


def test_timestamp_validation_rejects_none_strings_nan_inf_negative():
    """Verify non-numerics, NaN, inf, and negative numbers are rejected."""
    assert is_valid_timestamp(None) is False
    assert is_valid_timestamp("1672531199") is False
    assert is_valid_timestamp(float("nan")) is False
    assert is_valid_timestamp(float("inf")) is False
    assert is_valid_timestamp(-100.0) is False


def test_timestamp_validation_accepts_valid_integers_and_floats():
    """Verify non-negative finite numeric timestamps are accepted."""
    assert is_valid_timestamp(0) is True
    assert is_valid_timestamp(1730000000) is True
    assert is_valid_timestamp(1730000000.5) is True


def test_confidence_validation_rejects_booleans_and_malformed():
    """Verify malformed confidences and booleans default safely to 0.5."""
    assert validate_confidence(True) == 0.5
    assert validate_confidence(False) == 0.5
    assert validate_confidence(None) == 0.5
    assert validate_confidence("0.9") == 0.5
    assert validate_confidence(float("nan")) == 0.5
    assert validate_confidence(1.5) == 0.5
    assert validate_confidence(-0.1) == 0.5


def test_confidence_validation_accepts_valid_float():
    """Verify valid float confidences in [0.0, 1.0] are preserved."""
    assert validate_confidence(0.85) == 0.85
    assert validate_confidence(0.0) == 0.0
    assert validate_confidence(1.0) == 1.0


# ============================================================================
# 8. Seed Goal Reconciliation Tests
# ============================================================================


def test_unmentioned_seed_goal_is_never_emitted():
    """Verify seed goal lacking matching extracted candidates is discarded."""
    seed_goals = [{"title": "Buy a house", "area": "home", "type": "milestone", "horizon": "1yr"}]
    extracted: List[CandidateGoal] = []
    reconciled = reconcile_seed_goals(extracted, seed_goals)
    assert len(reconciled) == 0


def test_mentioned_seed_goal_reconciles_genuine_evidence():
    """Verify matching candidate transfers evidence to authoritative seed goal."""
    seed_goals = [{"title": "Reach 10 clients", "area": "work", "type": "milestone", "horizon": "1yr"}]
    cand = CandidateGoal(
        title="Reach 10 clients by March",
        area="work",
        type="milestone",
        horizon="3mo",
        evidence=["M10", "M11"],
        confidence=0.8,
        source_chats=["C01"],
    )
    reconciled = reconcile_seed_goals([cand], seed_goals)
    assert len(reconciled) == 1
    res = reconciled[0]
    assert res.title == "Reach 10 clients"  # Authoritative seed title
    assert res.horizon == "1yr"  # Authoritative seed horizon adopted
    assert res.evidence == ["M10", "M11"]  # Preserves candidate evidence
    assert res.confidence == 0.8  # No 1.0 inflation
    assert res.is_seed is True


def test_seed_reconciliation_authoritative_horizon():
    """Verify candidate horizon mismatch does not reject seed reconciliation."""
    seed_goals = [{"title": "Run half marathon", "area": "health", "type": "milestone", "horizon": "1yr"}]
    cand = CandidateGoal(
        title="Run the half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M01"],
        confidence=0.75,
    )
    assert are_seed_and_candidate_equivalent(seed_goals[0], cand) is True
    reconciled = reconcile_seed_goals([cand], seed_goals)
    assert len(reconciled) == 1
    assert reconciled[0].horizon == "1yr"
    assert reconciled[0].evidence == ["M01"]
    assert reconciled[0].confidence == 0.75


def test_seed_equivalence_rejects_metric_mismatch():
    """Verify seed goal with 10 clients does not reconcile candidate with 20 clients."""
    seed_item = {"title": "Reach 10 clients", "area": "work", "type": "milestone", "horizon": "1yr"}
    cand = CandidateGoal(
        title="Reach 20 clients",
        area="work",
        type="milestone",
        horizon="1yr",
        evidence=["M01"],
    )
    assert are_seed_and_candidate_equivalent(seed_item, cand) is False


def test_unmatched_candidate_goals_preserved_during_seed_reconciliation():
    """Verify non-seed candidate goals are preserved alongside reconciled seed goals."""
    seed_goals = [{"title": "Reach 10 clients", "area": "work", "type": "milestone", "horizon": "1yr"}]
    c_match = CandidateGoal(
        title="Reach 10 clients",
        area="work",
        type="milestone",
        horizon="1yr",
        evidence=["M01"],
    )
    c_other = CandidateGoal(
        title="Family trip to Coorg",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M02"],
    )
    reconciled = reconcile_seed_goals([c_match, c_other], seed_goals)
    assert len(reconciled) == 2
    seed_titles = [g.title for g in reconciled if g.is_seed]
    non_seed_titles = [g.title for g in reconciled if not g.is_seed]
    assert "Reach 10 clients" in seed_titles
    assert "Family trip to Coorg" in non_seed_titles


# ============================================================================
# 9. Evidence Grounding & Ranking Tests
# ============================================================================


def test_evidence_validation_discards_ungrounded_and_duplicate_ids():
    """Verify ungrounded IDs are dropped and duplicates deduplicated preserving order."""
    dataset_messages = {
        "M01": {"chat_id": "C01"},
        "M02": {"chat_id": "C02"},
    }
    goal = CandidateGoal(
        title="Run half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M01", "M99_FAKE", "M02", "M01"],
    )
    validated = validate_and_filter_goal_evidence(goal, dataset_messages)
    assert validated is not None
    assert validated.evidence == ["M01", "M02"]
    assert set(validated.source_chats) == {"C01", "C02"}


def test_goal_with_zero_verified_evidence_is_discarded():
    """Verify a goal with no valid evidence IDs in the dataset is discarded."""
    dataset_messages = {"M01": {"chat_id": "C01"}}
    goal = CandidateGoal(
        title="Run half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M99_FAKE"],
    )
    assert validate_and_filter_goal_evidence(goal, dataset_messages) is None


def test_evidence_first_ranking_prioritizes_cross_chat_and_multi_msg():
    """Verify ranking prioritizes cross-chat (>=2) and multi-msg (>=2) over raw confidence."""
    dataset_messages = {
        "M01": {"chat_id": "C01"},
        "M02": {"chat_id": "C02"},
        "M03": {"chat_id": "C01"},
    }
    # g_high_conf: single chat, single msg, confidence 0.99
    g_high_conf = CandidateGoal(
        title="Goal A",
        area="self",
        type="milestone",
        horizon="1yr",
        evidence=["M03"],
        confidence=0.99,
        source_chats=["C01"],
    )
    # g_cross_chat: 2 chats, 2 msgs, confidence 0.70
    g_cross_chat = CandidateGoal(
        title="Goal B",
        area="work",
        type="milestone",
        horizon="1yr",
        evidence=["M01", "M02"],
        confidence=0.70,
        source_chats=["C01", "C02"],
    )

    ranked = rank_and_clamp_goals([g_high_conf, g_cross_chat], dataset_messages, max_goals=10)
    assert len(ranked) == 2
    # Goal B ranks first despite lower confidence because of cross-chat support
    assert ranked[0]["title"] == "Goal B"
    assert ranked[1]["title"] == "Goal A"


def test_seed_status_does_not_trump_high_evidence_non_seed_goal():
    """Verify seed goals do not get artificial priority over non-seed goals with higher evidence."""
    dataset_messages = {
        "M01": {"chat_id": "C01"},
        "M02": {"chat_id": "C02"},
        "M03": {"chat_id": "C01"},
    }
    # Seed goal with 1 msg, 1 chat
    g_seed = CandidateGoal(
        title="Goal Seed",
        area="health",
        type="milestone",
        horizon="1yr",
        evidence=["M03"],
        confidence=0.8,
        source_chats=["C01"],
        is_seed=True,
    )
    # Non-seed goal with 2 chats, 2 msgs
    g_non_seed = CandidateGoal(
        title="Goal Organic",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M01", "M02"],
        confidence=0.8,
        source_chats=["C01", "C02"],
        is_seed=False,
    )
    ranked = rank_and_clamp_goals([g_seed, g_non_seed], dataset_messages, max_goals=10)
    assert ranked[0]["title"] == "Goal Organic"
    assert ranked[1]["title"] == "Goal Seed"


def test_schema_emission_strictly_contains_five_fields():
    """Verify ranked goal output dictionary contains strictly the five schema fields."""
    dataset_messages = {"M01": {"chat_id": "C01"}}
    goal = CandidateGoal(
        title="Run half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M01"],
        confidence=0.9,
    )
    ranked = rank_and_clamp_goals([goal], dataset_messages, max_goals=10)
    assert len(ranked) == 1
    assert set(ranked[0].keys()) == {"title", "area", "type", "horizon", "evidence"}


def test_rank_and_clamp_goals_enforces_schema_limit_and_bounds():
    """Verify rank_and_clamp_goals never emits > 10 goals and handles zero/negative safely."""
    # Create 15 distinct validated candidate goals
    dataset_messages = {f"M{i:02d}": {"chat_id": f"C{i:02d}"} for i in range(1, 16)}
    candidates = [
        CandidateGoal(
            title=f"Goal {i:02d}",
            area="work",
            type="milestone",
            horizon="1yr",
            evidence=[f"M{i:02d}"],
            confidence=0.8,
            source_chats=[f"C{i:02d}"],
        )
        for i in range(1, 16)
    ]

    # max_goals = 10
    ranked_10 = rank_and_clamp_goals(candidates, dataset_messages, max_goals=10)
    assert len(ranked_10) == 10

    # max_goals = 5
    ranked_5 = rank_and_clamp_goals(candidates, dataset_messages, max_goals=5)
    assert len(ranked_5) == 5

    # max_goals = 20 (must still clamp to absolute schema maximum of 10)
    ranked_20 = rank_and_clamp_goals(candidates, dataset_messages, max_goals=20)
    assert len(ranked_20) == 10

    # max_goals = 0 (emits empty list)
    ranked_0 = rank_and_clamp_goals(candidates, dataset_messages, max_goals=0)
    assert ranked_0 == []

    # Negative max_goals (must safely emit empty list, never slice backwards)
    ranked_neg1 = rank_and_clamp_goals(candidates, dataset_messages, max_goals=-1)
    assert ranked_neg1 == []

    ranked_neg5 = rank_and_clamp_goals(candidates, dataset_messages, max_goals=-5)
    assert ranked_neg5 == []


# ============================================================================
# 10. Chronological Evidence Ordering Tests
# ============================================================================


def test_chronological_ordering_when_all_timestamps_valid():
    """Case A: All timestamps valid -> sort strictly chronologically by ts ascending, mid ascending."""
    ts_index = {"M01": 200.0, "M02": 100.0, "M03": 300.0}
    ordered = union_evidence(["M01", "M03"], ["M02"], timestamp_index=ts_index)
    assert ordered == ["M02", "M01", "M03"]


def test_deterministic_fallback_when_no_timestamps():
    """Case B: No timestamp index -> sort deterministically by message ID ascending."""
    ordered = union_evidence(["M03", "M01"], ["M02"], timestamp_index=None)
    assert ordered == ["M01", "M02", "M03"]


def test_partial_timestamps_ordering_with_fallback_tail():
    """Case C: Partial timestamps -> known sorted chronologically first, unknown appended at tail."""
    ts_index = {"M01": 200.0, "M03": 100.0}  # M02 unindexed
    ordered = union_evidence(["M01", "M02"], ["M03"], timestamp_index=ts_index)
    assert ordered == ["M03", "M01", "M02"]


def test_invalid_timestamps_treated_as_unknown():
    """Case D: Invalid timestamps (booleans, NaN, inf, negative) treated as unknown tail IDs."""
    ts_index = {
        "M01": 200.0,
        "M02": True,  # boolean
        "M03": float("nan"),
        "M04": -50.0,  # negative
        "M05": 100.0,
    }
    ordered = union_evidence(["M01", "M02", "M03"], ["M04", "M05"], timestamp_index=ts_index)
    # Valid timestamps: M05 (100.0), M01 (200.0)
    # Unknown/invalid tail sorted alphabetically: M02, M03, M04
    assert ordered == ["M05", "M01", "M02", "M03", "M04"]


# ============================================================================
# 11. Immutability & Mutation Safety Tests
# ============================================================================


def test_from_interpreted_goal_does_not_mutate_source():
    """Verify from_interpreted_goal creates new CandidateGoal without mutating InterpretedGoal."""
    orig_msgs = ["M01", "M02"]
    ig = InterpretedGoal(
        title="Run half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence_msgs=orig_msgs,
        is_genuine=True,
        confidence=0.9,
    )
    cg = from_interpreted_goal(ig, chat_id="C01")
    assert cg.evidence == ["M01", "M02"]
    assert cg.evidence is not orig_msgs

    cg.evidence.append("M03")
    assert ig.evidence_msgs == ["M01", "M02"]


def test_deduplicate_goals_does_not_mutate_inputs():
    """Verify deduplicate_goals produces new CandidateGoal instances without mutating originals."""
    c1 = CandidateGoal(
        title="Family trip to Coorg in December",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M01"],
        source_chats=["C01"],
    )
    c2 = CandidateGoal(
        title="We want to visit Coorg in December",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M02"],
        source_chats=["C02"],
    )
    orig_c1_evidence = list(c1.evidence)
    orig_c2_evidence = list(c2.evidence)

    merged = deduplicate_goals([c1, c2])
    assert len(merged) == 1
    assert c1.evidence == orig_c1_evidence
    assert c2.evidence == orig_c2_evidence
    assert c1.source_chats == ["C01"]
    assert c2.source_chats == ["C02"]


# ============================================================================
# 12. Chat-ID Robustness and Validation Tests (Tests A-F)
# ============================================================================


def test_validate_evidence_numeric_chat_id():
    """Test A — Numeric chat ID: Evidence remains valid; numeric chat_id is excluded from source_chats."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": 123, "ts": 100},
    }
    cand = CandidateGoal(
        title="Start running four times a week",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M01"],
    )
    v_cand = validate_and_filter_goal_evidence(cand, dataset_messages)

    assert v_cand is not None
    assert v_cand.evidence == ["M01"]
    assert v_cand.source_chats == []


def test_validate_evidence_list_chat_id():
    """Test B — List chat ID: No exception occurs; evidence remains valid; source_chats is empty."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": ["C01"], "ts": 100},
    }
    cand = CandidateGoal(
        title="Maintain daily meditation",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M01"],
    )
    v_cand = validate_and_filter_goal_evidence(cand, dataset_messages)

    assert v_cand is not None
    assert v_cand.evidence == ["M01"]
    assert v_cand.source_chats == []


def test_validate_evidence_dictionary_chat_id():
    """Test C — Dictionary chat ID: No exception occurs; evidence remains valid; source_chats is empty."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": {"id": "C01"}, "ts": 100},
    }
    cand = CandidateGoal(
        title="Save 20 lakhs for down payment",
        area="home",
        type="milestone",
        horizon="1yr",
        evidence=["M01"],
    )
    v_cand = validate_and_filter_goal_evidence(cand, dataset_messages)

    assert v_cand is not None
    assert v_cand.evidence == ["M01"]
    assert v_cand.source_chats == []


def test_validate_evidence_mixed_valid_and_malformed_chat_ids():
    """Test D — Mixed valid and malformed chat IDs: C02, 123, C01, ['C03'], '', '   '."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": "C02", "ts": 100},
        "M02": {"id": "M02", "chat_id": 123, "ts": 110},
        "M03": {"id": "M03", "chat_id": "C01", "ts": 120},
        "M04": {"id": "M04", "chat_id": ["C03"], "ts": 130},
        "M05": {"id": "M05", "chat_id": "", "ts": 140},
        "M06": {"id": "M06", "chat_id": "   ", "ts": 150},
    }
    cand = CandidateGoal(
        title="Family trip to Coorg",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=["M01", "M02", "M03", "M04", "M05", "M06"],
    )
    v_cand = validate_and_filter_goal_evidence(cand, dataset_messages)

    assert v_cand is not None
    assert v_cand.evidence == ["M01", "M02", "M03", "M04", "M05", "M06"]
    assert v_cand.source_chats == ["C01", "C02"]


def test_validate_evidence_boolean_chat_ids():
    """Test E — Non-string and boolean chat IDs: True and False are excluded from source_chats."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": True, "ts": 100},
        "M02": {"id": "M02", "chat_id": False, "ts": 110},
    }
    cand = CandidateGoal(
        title="Run the half marathon",
        area="health",
        type="milestone",
        horizon="1yr",
        evidence=["M01", "M02"],
    )
    v_cand = validate_and_filter_goal_evidence(cand, dataset_messages)

    assert v_cand is not None
    assert v_cand.evidence == ["M01", "M02"]
    assert v_cand.source_chats == []


def test_validate_evidence_valid_string_chat_ids_unnormalized():
    """Test F — Valid string chat ID with surrounding whitespace preserved exactly as supplied."""
    dataset_messages = {
        "M01": {"id": "M01", "chat_id": " C01 ", "ts": 100},
    }
    cand = CandidateGoal(
        title="Buy a 3BHK",
        area="home",
        type="milestone",
        horizon="3yr",
        evidence=["M01"],
    )
    v_cand = validate_and_filter_goal_evidence(cand, dataset_messages)

    assert v_cand is not None
    assert v_cand.evidence == ["M01"]
    assert v_cand.source_chats == [" C01 "]

