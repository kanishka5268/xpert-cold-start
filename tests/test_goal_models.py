"""Unit tests for goal extraction data models (GoalMetrics, GoalExtractionConfig, CandidateGoal)."""

from dataclasses import FrozenInstanceError, is_dataclass
import pytest

from src.goals import (
    CandidateGoal,
    GoalExtractionConfig,
    GoalMetrics,
)
from src.goals.models import (
    CandidateGoal as ModelCandidateGoal,
    GoalExtractionConfig as ModelGoalExtractionConfig,
    GoalMetrics as ModelGoalMetrics,
)


def test_package_exports():
    """Verify clean export of all three model classes from src.goals."""
    assert CandidateGoal is ModelCandidateGoal
    assert GoalExtractionConfig is ModelGoalExtractionConfig
    assert GoalMetrics is ModelGoalMetrics
    assert is_dataclass(CandidateGoal)
    assert is_dataclass(GoalExtractionConfig)
    assert is_dataclass(GoalMetrics)


def test_goal_metrics_default_construction():
    """Verify default GoalMetrics instantiation produces empty sets for all 5 categories."""
    metrics = GoalMetrics()
    assert metrics.numbers == set()
    assert metrics.quantities_with_units == set()
    assert metrics.percentages == set()
    assert metrics.distances == set()
    assert metrics.ambiguous_metrics == set()
    assert metrics.is_empty() is True


def test_goal_metrics_explicit_population():
    """Verify GoalMetrics retains explicitly populated metric collections."""
    metrics = GoalMetrics(
        numbers={10.0, 25.5},
        quantities_with_units={"20l", "5cr"},
        percentages={"50%", "12.5%"},
        distances={"10k", "half_marathon"},
        ambiguous_metrics={"10k"},
    )
    assert metrics.numbers == {10.0, 25.5}
    assert metrics.quantities_with_units == {"20l", "5cr"}
    assert metrics.percentages == {"50%", "12.5%"}
    assert metrics.distances == {"10k", "half_marathon"}
    assert metrics.ambiguous_metrics == {"10k"}
    assert metrics.is_empty() is False


def test_goal_metrics_immutability():
    """Verify GoalMetrics is a frozen dataclass and prevents field re-assignment."""
    metrics = GoalMetrics()
    with pytest.raises(FrozenInstanceError):
        metrics.numbers = {5.0}  # type: ignore[misc]


def test_goal_metrics_is_empty_evaluation():
    """Verify is_empty() returns True only when all five collections are empty."""
    # Completely empty
    assert GoalMetrics().is_empty() is True

    # Populated across single dimensions
    assert GoalMetrics(numbers={10.0}).is_empty() is False
    assert GoalMetrics(quantities_with_units={"10k"}).is_empty() is False
    assert GoalMetrics(percentages={"20%"}).is_empty() is False
    assert GoalMetrics(distances={"5k"}).is_empty() is False
    assert GoalMetrics(ambiguous_metrics={"10k"}).is_empty() is False

    # Populated across multiple dimensions
    assert GoalMetrics(
        numbers={1.0},
        quantities_with_units={"20l"},
        distances={"10k"},
    ).is_empty() is False


def test_independent_mutable_default_collections():
    """Verify independent default collection instances across separate model objects."""
    # GoalMetrics instances
    m1 = GoalMetrics()
    m2 = GoalMetrics()
    assert m1.numbers is not m2.numbers
    assert m1.quantities_with_units is not m2.quantities_with_units
    assert m1.percentages is not m2.percentages
    assert m1.distances is not m2.distances
    assert m1.ambiguous_metrics is not m2.ambiguous_metrics

    # CandidateGoal instances
    g1 = CandidateGoal(
        title="Run half marathon",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M01"],
    )
    g2 = CandidateGoal(
        title="Save 20L",
        area="finances",
        type="milestone",
        horizon="1yr",
        evidence=["M02"],
    )
    assert g1.source_chats is not g2.source_chats


def test_goal_extraction_config_defaults_and_custom_values():
    """Verify GoalExtractionConfig defaults and custom overrides."""
    # Defaults
    default_config = GoalExtractionConfig()
    assert default_config.max_goals == 10
    assert default_config.default_habit_horizon == "3mo"
    assert default_config.default_milestone_horizon == "1yr"

    # Custom
    custom_config = GoalExtractionConfig(
        max_goals=5,
        default_habit_horizon="1yr",
        default_milestone_horizon="3yr",
    )
    assert custom_config.max_goals == 5
    assert custom_config.default_habit_horizon == "1yr"
    assert custom_config.default_milestone_horizon == "3yr"

    # Frozen
    with pytest.raises(FrozenInstanceError):
        default_config.max_goals = 20  # type: ignore[misc]


def test_candidate_goal_construction_and_defaults():
    """Verify CandidateGoal construction with required fields and optional defaults."""
    # Required fields only
    goal = CandidateGoal(
        title="Maintain four runs a week",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M10", "M11"],
    )
    assert goal.title == "Maintain four runs a week"
    assert goal.area == "health"
    assert goal.type == "habit"
    assert goal.horizon == "3mo"
    assert goal.evidence == ["M10", "M11"]
    assert goal.confidence is None
    assert goal.source_chats == []
    assert goal.is_seed is False

    # Fully populated
    full_goal = CandidateGoal(
        title="Reach 10 clients",
        area="work",
        type="milestone",
        horizon="1yr",
        evidence=["M20", "M21", "M22"],
        confidence=0.95,
        source_chats=["C01", "C02"],
        is_seed=True,
    )
    assert full_goal.confidence == 0.95
    assert full_goal.source_chats == ["C01", "C02"]
    assert full_goal.is_seed is True


def test_candidate_goal_evidence_count():
    """Verify evidence_count property tracks the total number of evidence IDs."""
    goal = CandidateGoal(
        title="Family trip to Coorg",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=[],
    )
    assert goal.evidence_count == 0

    goal.evidence.extend(["M01", "M02", "M03"])
    assert goal.evidence_count == 3


def test_candidate_goal_chat_count_unique_and_duplicates():
    """Verify chat_count property computes unique source chats, handling duplicates correctly."""
    goal = CandidateGoal(
        title="Start daily meditation",
        area="health",
        type="habit",
        horizon="3mo",
        evidence=["M01"],
        source_chats=[],
    )
    assert goal.chat_count == 0

    # Unique chats
    goal.source_chats = ["C01", "C02", "C03"]
    assert goal.chat_count == 3

    # Duplicate chats
    goal.source_chats = ["C01", "C02", "C01", "C03", "C02", "C01"]
    assert goal.chat_count == 3


def test_candidate_goal_to_schema_dict_exact_fields():
    """Verify to_schema_dict() emits strictly the five official schema fields."""
    goal = CandidateGoal(
        title="Run half marathon in January",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence=["M10", "M11"],
        confidence=0.88,
        source_chats=["C01", "C02"],
        is_seed=True,
    )
    schema_dict = goal.to_schema_dict()

    # Exact field set
    assert set(schema_dict.keys()) == {"title", "area", "type", "horizon", "evidence"}
    assert len(schema_dict) == 5

    # Field values
    assert schema_dict["title"] == "Run half marathon in January"
    assert schema_dict["area"] == "health"
    assert schema_dict["type"] == "milestone"
    assert schema_dict["horizon"] == "3mo"
    assert schema_dict["evidence"] == ["M10", "M11"]

    # Internal pipeline metadata must not leak into schema dict
    assert "confidence" not in schema_dict
    assert "source_chats" not in schema_dict
    assert "is_seed" not in schema_dict
    assert "evidence_count" not in schema_dict
    assert "chat_count" not in schema_dict


def test_candidate_goal_to_schema_dict_evidence_independence():
    """Verify to_schema_dict() returns an independent copy of the evidence list."""
    original_evidence = ["M01", "M02"]
    goal = CandidateGoal(
        title="Family trip to Coorg",
        area="experiences",
        type="experience",
        horizon="3mo",
        evidence=original_evidence,
    )

    schema_dict = goal.to_schema_dict()
    assert schema_dict["evidence"] == original_evidence
    assert schema_dict["evidence"] is not goal.evidence

    # Mutating schema_dict evidence does not mutate candidate goal
    schema_dict["evidence"].append("M03")
    assert goal.evidence == ["M01", "M02"]
    assert goal.evidence_count == 2

    # Mutating candidate goal evidence does not mutate previously emitted schema_dict
    goal.evidence.append("M99")
    assert schema_dict["evidence"] == ["M01", "M02", "M03"]


def test_candidate_goal_to_schema_dict_mutation_safety():
    """Verify mutating the returned dictionary does not mutate the CandidateGoal instance."""
    goal = CandidateGoal(
        title="Save 20L for down payment",
        area="finances",
        type="milestone",
        horizon="1yr",
        evidence=["M50"],
        confidence=0.9,
    )

    schema_dict = goal.to_schema_dict()
    schema_dict["title"] = "Completely Mutated Title"
    schema_dict["area"] = "work"
    schema_dict["type"] = "habit"
    schema_dict["horizon"] = "5yr"
    schema_dict["extra_key"] = "leaked_data"
    del schema_dict["evidence"]

    # Original CandidateGoal remains intact
    assert goal.title == "Save 20L for down payment"
    assert goal.area == "finances"
    assert goal.type == "milestone"
    assert goal.horizon == "1yr"
    assert goal.evidence == ["M50"]
    assert not hasattr(goal, "extra_key")

