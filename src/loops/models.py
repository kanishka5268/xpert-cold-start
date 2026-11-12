"""Data models for open loop processing, resolution, and schema emission."""

from dataclasses import dataclass, field
from datetime import date, datetime
import math
import re
from typing import Any, Dict, List, Optional, Set


VALID_OPEN_LOOP_TYPES: Set[str] = {
    "commitment",
    "question",
    "followup",
    "deadline",
}

VALID_OPEN_LOOP_STATUSES: Set[str] = {
    "open",
    "done",
    "dropped",
}

VALID_HELP_ACTIONS: Set[str] = {
    "assist",
    "remind",
    "leave",
}


@dataclass(frozen=True)
class HelpAction:
    """Structured container for the 'help' decision on an open loop.

    Enforces strict typing and real datetime parsing without fabricated placeholders.
    Strict UTC Policy: remind_at must be an explicit UTC instant ending strictly in 'Z' or '+00:00'.
    Rejects naive datetimes, '-00:00' (unknown local offset), and non-zero offsets.
    """

    action: str  # 'assist' | 'remind' | 'leave'
    next_step: Optional[str] = None
    needs_yes: Optional[bool] = None
    remind_at: Optional[str] = None
    reason: Optional[str] = None

    def validate(self) -> bool:
        """Validate action-specific fields strictly using real parsing without fabrications."""
        if not isinstance(self.action, str) or self.action not in VALID_HELP_ACTIONS:
            return False

        if self.action == "assist":
            if self.remind_at is not None or self.reason is not None:
                return False
            if not isinstance(self.next_step, str) or not self.next_step.strip():
                return False
            if "\r" in self.next_step or "\n" in self.next_step:
                return False
            if not isinstance(self.needs_yes, bool):
                return False

        elif self.action == "remind":
            if self.next_step is not None or self.needs_yes is not None or self.reason is not None:
                return False
            if not isinstance(self.remind_at, str) or not self.remind_at:
                return False
            # Reject leading or trailing whitespace without calling .strip() to normalize
            if self.remind_at.strip() != self.remind_at:
                return False
            val = self.remind_at

            # 1. Enforce explicit UTC timezone representation ('Z' or '+00:00' strictly)
            # Reject '-00:00' (which represents an unknown local offset per RFC 3339),
            # reject naive datetimes, and reject non-zero offsets like '+05:30'.
            if val.endswith("-00:00"):
                return False
            if not (val.endswith("Z") or val.endswith("+00:00")):
                return False

            # 2. Strict ISO-8601 calendar and clock parsing via datetime.fromisoformat()
            try:
                dt = datetime.fromisoformat(val)
            except (ValueError, TypeError):
                return False

            # 3. Require both date and time components
            if "T" not in val and " " not in val:
                return False

            # 4. Enforce timezone presence and exact zero UTC offset
            if dt.tzinfo is None:
                return False
            offset = dt.utcoffset()
            if offset is None or offset.total_seconds() != 0:
                return False

            # 5. Astronomical calendar year bounds
            if dt.year < 1970 or dt.year > 2100:
                return False

        elif self.action == "leave":
            if self.next_step is not None or self.needs_yes is not None or self.remind_at is not None:
                return False
            if not isinstance(self.reason, str) or not self.reason.strip():
                return False

        return True

    def to_schema_dict(self) -> Dict[str, Any]:
        """Emit strictly valid schema.json fields for help object.

        Raises ValueError if validation fails, preventing fabricated emission.
        """
        if not self.validate():
            raise ValueError(f"Invalid HelpAction object cannot be serialized: {self!r}")

        res: Dict[str, Any] = {"action": self.action}
        if self.action == "assist":
            res["next_step"] = self.next_step.strip()  # type: ignore[union-attr]
            res["needs_yes"] = self.needs_yes
        elif self.action == "remind":
            res["remind_at"] = self.remind_at  # Preserved byte-for-byte, no .strip()
        elif self.action == "leave":
            res["reason"] = self.reason.strip()  # type: ignore[union-attr]
        return res


@dataclass
class CandidateOpenLoop:
    """Internal open loop domain model decoupled from LLM raw records.

    Maintains intermediate provenance (due_anchor_msg, source_msgs_all) retained
    internally for verification and omitted from final schema emission.
    """

    title: str
    type: str
    owner: str  # Strictly 'ME' or authentic participant ID (e.g. 'P01')
    chat_id: str
    source_msg: str
    due: Optional[str] = None  # 'YYYY-MM-DD' or None
    due_anchor_msg: Optional[str] = None  # Message ID that supplied due date expression
    status: str = "open"  # 'open' | 'done' | 'dropped'
    closed_by: Optional[str] = None  # Message ID demonstrating closure
    help: Optional[HelpAction] = None
    confidence: float = 0.8
    source_msgs_all: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate model field types, enums, identifier exactness, and status-dependent schema invariants.

        Strict Identity Invariant:
        Identity fields (owner, chat_id, source_msg, closed_by) must be non-empty strings
        without leading or trailing whitespace. Identifiers with whitespace are rejected rather
        than silently repaired or stripped.

        Note: Contextual authenticity checks (verifying participant ID existence,
        chat membership, message existence, and timestamp ordering) are performed
        in the pipeline's final resolver validation stage because the domain model
        alone does not have access to the ConversationContext dataset.
        """
        # 1. Base required field validation (display text title may have whitespace trimmed)
        if not isinstance(self.title, str) or not self.title.strip():
            return False
        if not isinstance(self.type, str) or self.type not in VALID_OPEN_LOOP_TYPES:
            return False
        if not isinstance(self.status, str) or self.status not in VALID_OPEN_LOOP_STATUSES:
            return False

        # 2. Strict identity field validation: must be exact non-empty strings with NO leading/trailing whitespace
        if not isinstance(self.owner, str) or not self.owner or self.owner.strip() != self.owner:
            return False
        if not isinstance(self.chat_id, str) or not self.chat_id or self.chat_id.strip() != self.chat_id:
            return False
        if not isinstance(self.source_msg, str) or not self.source_msg or self.source_msg.strip() != self.source_msg:
            return False

        # 3. Due date validation: YYYY-MM-DD calendar date or None
        if self.due is not None:
            if not isinstance(self.due, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", self.due):
                return False
            try:
                date.fromisoformat(self.due)
            except ValueError:
                return False

        # 4. Status-dependent schema invariants (no silent conversions to None)
        if self.status == "open":
            if self.closed_by is not None:
                return False
            if not isinstance(self.help, HelpAction) or not self.help.validate():
                return False
        elif self.status in ("done", "dropped"):
            # closed_by must be an exact non-empty string with NO leading/trailing whitespace
            if not isinstance(self.closed_by, str) or not self.closed_by or self.closed_by.strip() != self.closed_by:
                return False
            if self.help is not None:
                return False

        return True

    def to_schema_dict(self) -> Dict[str, Any]:
        """Convert to official schema.json open_loop object (all nine official schema properties).

        Strict Identity & Validation Invariants:
        - Raises ValueError if validation fails, preventing emission of malformed or invalid objects.
        - Preserves exact byte-for-byte values of identity fields (owner, chat_id, source_msg, closed_by).
        - NEVER calls .strip() to normalize or repair identity fields during serialization.
        - Display text (title) may have surrounding whitespace trimmed.

        Emits: title, type, owner, chat_id, source_msg, due, status, closed_by, help.
        Nullable fields (due, closed_by, help) are explicitly emitted as None/null where required.
        Internal provenance fields (due_anchor_msg, source_msgs_all, confidence) are strictly excluded from output.
        """
        if not self.validate():
            raise ValueError(f"Invalid CandidateOpenLoop cannot be serialized to schema: {self!r}")

        return {
            "title": self.title.strip(),
            "type": self.type,
            "owner": self.owner,
            "chat_id": self.chat_id,
            "source_msg": self.source_msg,
            "due": self.due,
            "status": self.status,
            "closed_by": self.closed_by if self.status in ("done", "dropped") else None,
            "help": self.help.to_schema_dict() if self.status == "open" and self.help else None,
        }


def _validate_cap_limit(field_name: str, value: Any) -> None:
    """Validate that a cap configuration parameter is either None or a strictly positive integer (> 0).

    Rejects:
    - Booleans (bool is a subclass of int in Python)
    - Non-integers (float, str, list, etc.)
    - Non-positive integers (<= 0)
    """
    if value is None:
        return
    if isinstance(value, bool):
        raise TypeError(f"{field_name} cannot be a boolean (got {value!r})")
    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an integer or None, got {type(value).__name__} ({value!r})"
        )
    if value <= 0:
        raise ValueError(f"{field_name} must be a strictly positive integer (> 0), got {value}")


@dataclass(frozen=True)
class OpenLoopExtractionConfig:
    """Configuration parameters for open-loop extraction and decision rules."""

    min_confidence: float = 0.5
    conservative_closure: bool = True
    default_remind_lead_hours: int = 24
    max_loops_per_chat: Optional[int] = None  # Configurable safety valve, disabled by default (None = unlimited)
    max_open_loops: Optional[int] = None  # Optional global cap, disabled by default (None = unlimited)
    max_runtime_seconds: Optional[float] = 1500.0  # Open-loop extraction phase budget (default: 25 min)
    enable_fallback: Optional[bool] = None  # None = auto-detect (runs if offline/no interpreter)
    enable_selective_routing: bool = True  # Three-tier selective LLM routing (Scenario C)
    disjoint_batching: bool = True  # Multi-snippet disjoint evidence batching (Scenario C)

    def __post_init__(self) -> None:
        """Validate configuration cap parameters strictly."""
        _validate_cap_limit("max_loops_per_chat", self.max_loops_per_chat)
        _validate_cap_limit("max_open_loops", self.max_open_loops)
        if self.enable_fallback is not None and not isinstance(self.enable_fallback, bool):
            raise TypeError(f"enable_fallback must be a bool or None, got {type(self.enable_fallback).__name__}")
        if not isinstance(self.enable_selective_routing, bool):
            raise TypeError(f"enable_selective_routing must be a bool, got {type(self.enable_selective_routing).__name__}")
        if not isinstance(self.disjoint_batching, bool):
            raise TypeError(f"disjoint_batching must be a bool, got {type(self.disjoint_batching).__name__}")
        if self.max_runtime_seconds is not None:
            if isinstance(self.max_runtime_seconds, bool) or not isinstance(self.max_runtime_seconds, (int, float)):
                raise TypeError(
                    f"max_runtime_seconds must be a number or None, got {type(self.max_runtime_seconds).__name__} ({self.max_runtime_seconds!r})"
                )
            fval = float(self.max_runtime_seconds)
            if not math.isfinite(fval):
                raise ValueError(
                    f"max_runtime_seconds must be a finite number, got {self.max_runtime_seconds!r}"
                )
            if fval <= 0:
                raise ValueError(
                    f"max_runtime_seconds must be strictly positive (> 0), got {self.max_runtime_seconds}"
                )


@dataclass
class OpenLoopExtractionDiagnostics:
    """Operational counters for monitoring open loop extraction pipeline."""

    candidates_received: int = 0
    candidates_after_group_filter: int = 0
    batches_created: int = 0
    batches_succeeded: int = 0
    batches_failed: int = 0
    batches_cached: int = 0
    batches_unattempted: int = 0
    batches_attempted_live: int = 0
    batches_offline_unavailable: int = 0
    batches_budget_exhausted: int = 0
    batches_unexpected_errors: int = 0
    fallback_candidates_synthesized: int = 0
    successful_candidates_count: int = 0
    failed_candidates_count: int = 0
    unexpected_candidates_count: int = 0
    interpreted_loops_received: int = 0
    genuine_loops_received: int = 0
    candidates_created: int = 0
    candidates_rejected: int = 0
    loops_deduplicated: int = 0
    loops_marked_done: int = 0
    loops_marked_dropped: int = 0
    loops_kept_open: int = 0
    help_assist_count: int = 0
    help_remind_count: int = 0
    help_leave_count: int = 0
    loops_truncated_by_cap: int = 0
    final_loops_emitted: int = 0
    tier1_count: int = 0
    tier2_count: int = 0
    tier3_count: int = 0
    tier3_batches_created: int = 0
    tier3_batches_succeeded: int = 0
    tier3_batches_failed: int = 0
    tier3_cache_hits: int = 0

    @property
    def tier1_candidates(self) -> List[str]:
        if not hasattr(self, "_tier1_candidates"):
            self._tier1_candidates: List[str] = []
        return self._tier1_candidates

    @tier1_candidates.setter
    def tier1_candidates(self, val: Any) -> None:
        self._tier1_candidates = list(val)
        self.tier1_count = len(self._tier1_candidates)

    @property
    def tier2_candidates(self) -> List[str]:
        if not hasattr(self, "_tier2_candidates"):
            self._tier2_candidates: List[str] = []
        return self._tier2_candidates

    @tier2_candidates.setter
    def tier2_candidates(self, val: Any) -> None:
        self._tier2_candidates = list(val)
        self.tier2_count = len(self._tier2_candidates)

    @property
    def tier3_candidates(self) -> List[str]:
        if not hasattr(self, "_tier3_candidates"):
            self._tier3_candidates: List[str] = []
        return self._tier3_candidates

    @tier3_candidates.setter
    def tier3_candidates(self, val: Any) -> None:
        self._tier3_candidates = list(val)
        self.tier3_count = len(self._tier3_candidates)

    @property
    def routing_reasons(self) -> Dict[str, str]:
        if not hasattr(self, "_routing_reasons"):
            self._routing_reasons: Dict[str, str] = {}
        return self._routing_reasons

    @routing_reasons.setter
    def routing_reasons(self, val: Any) -> None:
        self._routing_reasons = dict(val)

    def record_unexpected_error(self, error_record: Dict[str, Any]) -> None:
        """Record an unexpected error record internally without exposing non-int default in __dict__."""
        if not hasattr(self, "_unexpected_errors"):
            self._unexpected_errors: List[Dict[str, Any]] = []
        self._unexpected_errors.append(error_record)

    @property
    def unexpected_errors(self) -> List[Dict[str, Any]]:
        """Return list of unexpected error records recorded during execution."""
        if not hasattr(self, "_unexpected_errors"):
            self._unexpected_errors = []
        return self._unexpected_errors

