"""Goal extraction, semantic deduplication, and schema emission package."""

from src.goals.extractor import (
    GoalExtractionDiagnostics,
    GoalExtractionResult,
    build_timestamp_index,
    convert_interpreted_goals_to_candidates,
    extract_goals,
    extract_seed_goals,
    interpret_goal_evidence,
)
from src.goals.models import CandidateGoal, GoalExtractionConfig, GoalMetrics

__all__ = [
    "CandidateGoal",
    "GoalExtractionConfig",
    "GoalExtractionDiagnostics",
    "GoalExtractionResult",
    "GoalMetrics",
    "build_timestamp_index",
    "convert_interpreted_goals_to_candidates",
    "extract_goals",
    "extract_seed_goals",
    "interpret_goal_evidence",
]

