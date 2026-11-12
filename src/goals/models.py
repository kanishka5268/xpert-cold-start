"""Data models for goal post-processing, deduplication, and schema emission."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass(frozen=True)
class GoalMetrics:
    """Structured representation of extracted quantitative targets, distances, and ambiguous metrics."""

    numbers: Set[float] = field(default_factory=set)
    quantities_with_units: Set[str] = field(default_factory=set)
    percentages: Set[str] = field(default_factory=set)
    distances: Set[str] = field(default_factory=set)
    ambiguous_metrics: Set[str] = field(default_factory=set)

    def is_empty(self) -> bool:
        """Return True only when all five metric collections are empty."""
        return (
            not self.numbers
            and not self.quantities_with_units
            and not self.percentages
            and not self.distances
            and not self.ambiguous_metrics
        )


@dataclass(frozen=True)
class GoalExtractionConfig:
    """Configuration parameters for goal deduplication and ranking."""

    max_goals: int = 10
    default_habit_horizon: str = "3mo"
    default_milestone_horizon: str = "1yr"


@dataclass
class CandidateGoal:
    """Internal goal model used during synthesis, semantic deduplication, and schema emission.

    Decouples raw InterpretedGoal from mutation while maintaining full provenance.
    """

    title: str
    area: str
    type: str
    horizon: str
    evidence: List[str]
    confidence: Optional[float] = None
    source_chats: List[str] = field(default_factory=list)
    is_seed: bool = False

    @property
    def evidence_count(self) -> int:
        """Return the number of supporting evidence IDs."""
        return len(self.evidence)

    @property
    def chat_count(self) -> int:
        """Return the number of unique source chat IDs."""
        return len(set(self.source_chats))

    def to_schema_dict(self) -> Dict[str, Any]:
        """Convert to official schema.json goal object (strictly 5 fields).

        Returns a new dictionary with a shallow copy of the evidence list to prevent
        external mutation of the candidate goal's internal state.
        """
        return {
            "title": self.title,
            "area": self.area,
            "type": self.type,
            "horizon": self.horizon,
            "evidence": list(self.evidence),
        }

