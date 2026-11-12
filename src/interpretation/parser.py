"""LLM response parsing and validation module for conversation evidence interpretation."""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.interpretation.batcher import EvidenceBatch
from src.interpretation.records import (
    InterpretedGoal,
    InterpretedOpenLoop,
    InterpretedOrganisation,
    InterpretedPerson,
)
from src.preprocessing.context_builder import ConversationContext

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

VALID_SUGGESTED_ACTIONS: Set[str] = {
    "assist",
    "remind",
    "leave",
}

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

VALID_RELATIONSHIP_CIRCLES: Set[str] = {
    "spouse",
    "child",
    "parent",
    "in-law",
    "sibling",
    "relative",
    "friend",
    "colleague",
    "client",
    "vendor",
    "partner",
    "staff",
}

VALID_ORG_LINK_TYPES: Set[str] = {
    "runs",
    "supports",
}

VALID_ORG_TYPES: Set[str] = {
    "company",
    "residence",
    "vacation home",
    "workspace",
    "society",
    "school",
    "other",
}

VALID_ORG_SIDES: Set[str] = {
    "home",
    "work",
}

VALID_ORG_ROLES: Set[str] = {
    "owner",
    "partner",
    "helper",
    "resident",
    "parent",
    "member",
    "other",
}


class InterpretationResponseError(ValueError):
    """Base exception for all interpretation response parsing and validation errors."""
    pass


class ResponseParsingError(InterpretationResponseError):
    """Raised when raw model output cannot be parsed as valid, unambiguous JSON."""
    pass


class ResponseStructureError(InterpretationResponseError):
    """Raised when JSON top-level shape or required collection arrays are missing or invalid."""
    pass


@dataclass
class ChatParserDiagnostics:
    """Diagnostic tracking for chat interpretation parsing rejections and acceptances."""

    # Open Loops
    open_loops_total_raw: int = 0
    open_loops_non_dict: int = 0
    open_loops_missing_required_fields: int = 0
    open_loops_invalid_enums: int = 0
    open_loops_invalid_or_missing_is_genuine: int = 0
    open_loops_rejected_not_genuine: int = 0
    open_loops_invalid_or_ungrounded_source: int = 0
    open_loops_accepted: int = 0

    # Goals
    goals_total_raw: int = 0
    goals_non_dict: int = 0
    goals_missing_required_fields: int = 0
    goals_invalid_enums: int = 0
    goals_invalid_or_missing_is_genuine: int = 0
    goals_rejected_not_genuine: int = 0
    goals_invalid_or_ungrounded_evidence: int = 0
    goals_horizon_normalized: int = 0
    goals_accepted: int = 0

    # Sanitized rejection details
    rejection_details: List[Dict[str, str]] = field(default_factory=list)

    def record_rejection(self, record_type: str, reason: str, identifier: Optional[str] = None) -> None:
        """Record a sanitized rejection reason without exposing conversation contents or credentials."""
        entry = {"record_type": record_type, "reason": reason}
        if identifier:
            entry["identifier"] = str(identifier)
        self.rejection_details.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic counters to dictionary."""
        return {
            "open_loops": {
                "total_raw": self.open_loops_total_raw,
                "non_dict": self.open_loops_non_dict,
                "missing_required_fields": self.open_loops_missing_required_fields,
                "invalid_enums": self.open_loops_invalid_enums,
                "invalid_or_missing_is_genuine": self.open_loops_invalid_or_missing_is_genuine,
                "rejected_not_genuine": self.open_loops_rejected_not_genuine,
                "invalid_or_ungrounded_source": self.open_loops_invalid_or_ungrounded_source,
                "accepted": self.open_loops_accepted,
            },
            "goals": {
                "total_raw": self.goals_total_raw,
                "non_dict": self.goals_non_dict,
                "missing_required_fields": self.goals_missing_required_fields,
                "invalid_enums": self.goals_invalid_enums,
                "invalid_or_missing_is_genuine": self.goals_invalid_or_missing_is_genuine,
                "rejected_not_genuine": self.goals_rejected_not_genuine,
                "invalid_or_ungrounded_evidence": self.goals_invalid_or_ungrounded_evidence,
                "horizon_normalized": self.goals_horizon_normalized,
                "accepted": self.goals_accepted,
            },
            "rejection_count": len(self.rejection_details),
        }


def _reject_duplicate_keys(pairs: List[Tuple[Any, Any]]) -> Dict[Any, Any]:
    """Object pairs hook that rejects duplicate keys at any nesting level."""
    result: Dict[Any, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def _reject_constants(constant: str) -> None:
    """Parse constant hook that rejects non-standard JSON constants (NaN, Infinity, -Infinity)."""
    raise ValueError(f"Non-standard JSON constant rejected: {constant!r}")


def _strict_json_loads(text: str) -> Any:
    """Parse JSON string strictly rejecting duplicate keys and non-standard constants."""
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constants,
    )


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """
    Extract a single unambiguous JSON dictionary from raw model response text.

    Accepts:
    1. Direct JSON string (starts with '{' and ends with '}').
    2. Exactly one complete Markdown-fenced JSON block (```json ... ``` or ``` ... ```).
    3. One unambiguous JSON object surrounded by explanatory text.

    Rejects:
    - Empty, whitespace-only, or non-string input.
    - Multiple JSON objects or multiple code fences.
    - Unclosed or truncated code fences.
    - Truncated or malformed JSON (no auto-repairing or trailing comma removal).
    - Duplicate keys at any nesting level.
    - Non-standard JSON constants (NaN, Infinity, -Infinity).
    - Top-level JSON that is not a dictionary.

    Args:
        raw_text: Raw string returned by the LLM client.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ResponseParsingError: If text is empty, malformed, truncated, duplicate-keyed, or non-standard.
        ResponseStructureError: If parsed root is not a JSON object (dict).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ResponseParsingError("Empty or non-string response content")

    text = raw_text.strip()

    # 1. Check for Markdown fences
    if "```" in text:
        fence_count = text.count("```")
        if fence_count != 2:
            raise ResponseParsingError(
                f"Ambiguous or truncated code fences: found {fence_count} fence markers (expected exactly 2)"
            )
        first_fence = text.find("```")
        second_fence = text.find("```", first_fence + 3)
        fence_content = text[first_fence + 3 : second_fence].strip()
        if fence_content.lower().startswith("json"):
            fence_content = fence_content[4:].strip()
        text = fence_content

    # 2. Try direct JSON parse first
    try:
        data = _strict_json_loads(text)
    except (json.JSONDecodeError, ValueError) as err:
        if text.startswith("{") and text.endswith("}"):
            raise ResponseParsingError(f"Malformed, duplicate, or non-standard JSON: {err}") from err
    else:
        if not isinstance(data, dict):
            raise ResponseStructureError(f"Expected top-level JSON object (dict), got {type(data).__name__}")
        return data

    # 3. Locate unambiguous JSON object boundaries surrounded by commentary
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        raise ResponseParsingError("No valid JSON object boundaries found in text")

    candidate_json = text[first_brace : last_brace + 1]

    # 4. Strict, standard JSON parse of candidate object (no fuzzy repairs, no trailing comma removal)
    try:
        data = _strict_json_loads(candidate_json)
    except (json.JSONDecodeError, ValueError) as err:
        raise ResponseParsingError(f"Malformed, duplicate, or non-standard JSON: {err}") from err

    # 5. Enforce dictionary root
    if not isinstance(data, dict):
        raise ResponseStructureError(f"Expected top-level JSON object (dict), got {type(data).__name__}")

    return data


def is_name_grounded_in_text(name: str, corpus: str) -> bool:
    """
    Verify whether an organisation name appears as an authentic entity phrase
    in the prompt text using case-insensitive word boundaries, preventing
    accidental substring matches in unrelated words.

    Args:
        name: Organisation name candidate.
        corpus: Searchable text corpus from prompt inputs.

    Returns:
        True if name matches as a distinct word/phrase in corpus; False otherwise.
    """
    if not isinstance(name, str) or not name.strip() or not isinstance(corpus, str) or not corpus:
        return False
    pattern = r"\b" + re.escape(name.strip()) + r"\b"
    return bool(re.search(pattern, corpus, flags=re.IGNORECASE))


def _is_valid_timestamp(ts: Any) -> bool:
    """Return True if ts is a real numeric timestamp (int or finite float, excluding bool)."""
    if isinstance(ts, bool) or ts is None:
        return False
    if isinstance(ts, int):
        return True
    if isinstance(ts, float):
        return math.isfinite(ts)
    return False


def _parse_valid_due_date(val: Any) -> Optional[str]:
    """Validate that val is a string matching YYYY-MM-DD and is a real calendar date."""
    if not isinstance(val, str):
        return None
    val = val.strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", val):
        return None
    try:
        d = date.fromisoformat(val)
        return d.isoformat()
    except ValueError:
        return None


def _validate_confidence(val: Any) -> Optional[float]:
    """Validate that confidence is a float or int between 0.0 and 1.0 (never clamped)."""
    if val is None:
        return None
    if isinstance(val, bool):  # In Python, bool is subclass of int, so reject bool
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
        if 0.0 <= fval <= 1.0:
            return fval
    return None


MONTH_NAME_TO_NUM: Dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

MONTH_REGEX = re.compile(
    r"\b(?:in|by|for|before|around|until)?\s*(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)\b",
    re.IGNORECASE,
)


def normalize_goal_horizon(
    raw_horizon: Any,
    title: str = "",
    goal_type: str = "milestone",
    evidence_msgs: Optional[Sequence[str]] = None,
    context: Optional[ConversationContext] = None,
    batch_messages_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    anchor_timestamp: Optional[float] = None,
) -> Tuple[str, bool]:
    """Deterministically normalize goal horizon to one of ('3mo', '1yr', '3yr', '5yr').

    Returns (normalized_horizon, was_normalized).
    """
    if isinstance(raw_horizon, str) and raw_horizon.strip().lower() in VALID_GOAL_HORIZONS:
        return raw_horizon.strip().lower(), False

    # Check mapping for common invalid enums
    if isinstance(raw_horizon, str):
        cleaned_raw = raw_horizon.strip().lower()
        if cleaned_raw in ("6mo", "6m", "half_year"):
            return "1yr", True
        if cleaned_raw in ("2yr", "2y", "24mo"):
            return "3yr", True
        if cleaned_raw in ("4yr", "4y", "5y+"):
            return "5yr", True

    # Assemble contextual text and anchor timestamp from evidence messages
    evidence_text_parts: List[str] = []
    inferred_anchor_ts: Optional[float] = anchor_timestamp

    if evidence_msgs:
        for mid in evidence_msgs:
            msg = None
            if batch_messages_by_id and mid in batch_messages_by_id:
                msg = batch_messages_by_id[mid]
            elif context and hasattr(context, "dataset") and isinstance(context.dataset, dict):
                msg = context.dataset.get("messages_by_id", {}).get(mid)
            if msg and isinstance(msg, dict):
                body = msg.get("body") or msg.get("text") or ""
                if body:
                    evidence_text_parts.append(str(body))
                ts = msg.get("ts")
                if isinstance(ts, (int, float)) and ts > 0 and inferred_anchor_ts is None:
                    inferred_anchor_ts = float(ts)

    evidence_text = " ".join(evidence_text_parts)
    combined_text = f"{title} {evidence_text}".lower()

    # 1. Multi-year explicit targets (3yr or 5yr)
    if re.search(r"\b(?:5|five|4|four)\s+years?\b", combined_text) or "retire" in combined_text:
        return "5yr", True
    if re.search(r"\b(?:3|three|2|two)\s+years?\b|\bmulti-?year\b", combined_text):
        return "3yr", True

    # 2. Check explicit target months relative to anchor timestamp
    m_match = MONTH_REGEX.search(combined_text)
    if m_match:
        target_month_name = m_match.group(1).lower()
        target_month_num = MONTH_NAME_TO_NUM.get(target_month_name)
        if target_month_num is not None:
            if inferred_anchor_ts and inferred_anchor_ts > 0:
                try:
                    dt = datetime.fromtimestamp(inferred_anchor_ts, tz=timezone.utc)
                    current_month = dt.month
                    diff = (target_month_num - current_month) % 12
                    if diff == 0:
                        diff = 12
                    if diff <= 3:
                        return "3mo", True
                    elif diff <= 12:
                        return "1yr", True
                except (ValueError, OverflowError, OSError):
                    pass
            # Fallback when anchor timestamp is not available
            if target_month_num in (1, 2, 12):
                return "3mo", True
            else:
                return "1yr", True

    # 3. Near-term temporal expressions
    if re.search(r"\b(?:next\s+month|this\s+month|upcoming|couple\s+of\s+months|soon|q[1-2]|quarter|90\s+days?|3\s+months?)\b", combined_text):
        return "3mo", True

    # 4. Long-term / annual expressions
    if re.search(r"\b(?:next\s+year|by\s+next\s+year|1\s+year|12\s+months?|annual|annually)\b", combined_text):
        return "1yr", True

    # 5. Type-informed deterministic fallback
    if goal_type == "habit":
        return "3mo", True
    if goal_type == "experience":
        return "3mo", True
    if goal_type == "milestone":
        return "1yr", True

    return "1yr", True


def parse_chat_interpretation_response(
    raw_text: str,
    batch: EvidenceBatch,
    context: Optional[ConversationContext] = None,
    diagnostics: Optional[ChatParserDiagnostics] = None,
) -> Tuple[List[InterpretedOpenLoop], List[InterpretedGoal]]:
    """
    Parse and validate LLM output for a chat-level batch into typed records.

    Enforces that:
    - Root JSON contains BOTH 'open_loops' and 'goals', each an explicit list.
      (Missing key or wrong type raises ResponseStructureError; empty lists [] are valid).
    - Open loop source_msg belongs strictly to batch.candidate_message_ids AND batch.messages.
    - If candidate source_msg is absent from batch.messages, reject that open-loop record.
    - Open loop chat_id matches batch.chat_id.
    - Open loop closed_by belongs strictly to batch.messages with ts(closed_by) >= ts(source_msg).
    - Both source and closing timestamps must be valid numbers for closed_by to be preserved.
    - Goal evidence_msgs belong strictly to batch.messages.
    - Required enums match permitted values exactly without substitution.
    - Records with is_genuine is False are discarded; non-boolean values are rejected.
    - Invalid items are discarded independently without failing valid items in the batch.

    Args:
        raw_text: Raw string returned by the LLM.
        batch: The EvidenceBatch instance submitted to the prompt.
        context: Optional ConversationContext for participant ID validation.

    Returns:
        Tuple of (valid_genuine_open_loops, valid_genuine_goals).

    Raises:
        ResponseParsingError: If raw_text is not valid, unambiguous JSON.
        ResponseStructureError: If root structure or required collection arrays are invalid.
        ValueError: If batch is None or not an EvidenceBatch.
    """
    if not isinstance(batch, EvidenceBatch):
        raise ValueError(f"Expected EvidenceBatch instance, got {type(batch).__name__}")

    data = extract_json_payload(raw_text)

    # 1. Required top-level collection keys
    if "open_loops" not in data or not isinstance(data["open_loops"], list):
        raise ResponseStructureError("Response missing required 'open_loops' list")
    if "goals" not in data or not isinstance(data["goals"], list):
        raise ResponseStructureError("Response missing required 'goals' list")

    # 2. Build prompt-bounded reference maps
    candidate_message_ids = set(batch.candidate_message_ids)
    batch_messages_by_id: Dict[str, Dict[str, Any]] = {
        m["id"]: m for m in batch.messages if isinstance(m, dict) and m.get("id")
    }

    # Snippet boundary map for multi-snippet batches
    cand_snippet_allowed_msgs: Dict[str, Set[str]] = {}
    if getattr(batch, "snippets", None) and len(batch.snippets) > 1:
        for snip in batch.snippets:
            snip_msgs = {m["id"] for m in snip.messages if isinstance(m, dict) and m.get("id")}
            for mid in snip.candidate_message_ids:
                cand_snippet_allowed_msgs[mid] = snip_msgs

    # Participant IDs present in the chat prompt input
    prompt_participant_ids: Set[str] = set()
    for m in batch.messages:
        if isinstance(m, dict) and m.get("sender_id"):
            prompt_participant_ids.add(m["sender_id"])
    for c in batch.candidates:
        if isinstance(c, dict) and c.get("sender_id"):
            prompt_participant_ids.add(c["sender_id"])

    allowed_owners: Set[str] = {"ME"} | prompt_participant_ids
    allowed_targets: Set[str] = {"ME"} | prompt_participant_ids

    # 3. Parse and validate Open Loops
    valid_open_loops: List[InterpretedOpenLoop] = []
    if diagnostics is not None:
        diagnostics.open_loops_total_raw += len(data["open_loops"])

    for item in data["open_loops"]:
        if not isinstance(item, dict):
            if diagnostics is not None:
                diagnostics.open_loops_non_dict += 1
                diagnostics.record_rejection("open_loop", "non_dict_record")
            continue

        source_msg = item.get("source_msg")
        # Candidate source_msg must be in candidate_message_ids AND in batch.messages
        if (
            not isinstance(source_msg, str)
            or source_msg not in candidate_message_ids
            or source_msg not in batch_messages_by_id
        ):
            if diagnostics is not None:
                diagnostics.open_loops_invalid_or_ungrounded_source += 1
                diagnostics.record_rejection("open_loop", "invalid_or_ungrounded_source", str(source_msg) if source_msg else None)
            continue

        chat_id = item.get("chat_id")
        if not isinstance(chat_id, str) or chat_id != batch.chat_id:
            if diagnostics is not None:
                diagnostics.open_loops_missing_required_fields += 1
                diagnostics.record_rejection("open_loop", "invalid_or_mismatched_chat_id", str(source_msg))
            continue

        loop_type = item.get("type")
        if not isinstance(loop_type, str) or loop_type not in VALID_OPEN_LOOP_TYPES:
            if diagnostics is not None:
                diagnostics.open_loops_invalid_enums += 1
                diagnostics.record_rejection("open_loop", "invalid_type_enum", str(source_msg))
            continue

        is_genuine = item.get("is_genuine")
        if not isinstance(is_genuine, bool):
            if diagnostics is not None:
                diagnostics.open_loops_invalid_or_missing_is_genuine += 1
                diagnostics.record_rejection("open_loop", "invalid_or_missing_is_genuine", str(source_msg))
            continue
        if is_genuine is False:
            if diagnostics is not None:
                diagnostics.open_loops_rejected_not_genuine += 1
                diagnostics.record_rejection("open_loop", "rejected_not_genuine", str(source_msg))
            continue  # Correctly discard non-genuine conversational items

        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            if diagnostics is not None:
                diagnostics.open_loops_missing_required_fields += 1
                diagnostics.record_rejection("open_loop", "missing_required_title", str(source_msg))
            continue

        status = item.get("status")
        if not isinstance(status, str) or status not in VALID_OPEN_LOOP_STATUSES:
            if diagnostics is not None:
                diagnostics.open_loops_invalid_enums += 1
                diagnostics.record_rejection("open_loop", "invalid_status_enum", str(source_msg))
            continue

        # Optional fields & validation flags
        flags: List[str] = []

        owner = item.get("owner")
        if owner is not None:
            if not isinstance(owner, str) or owner not in allowed_owners:
                flags.append(f"unsupplied_owner:{owner}")
                owner = None

        target_person = item.get("target_person")
        if target_person is not None:
            if not isinstance(target_person, str) or target_person not in allowed_targets:
                flags.append(f"unsupplied_target_person:{target_person}")
                target_person = None

        snippet_allowed_msgs = cand_snippet_allowed_msgs.get(source_msg)

        closed_by = item.get("closed_by")
        if closed_by is not None:
            if not isinstance(closed_by, str) or not closed_by.strip():
                closed_by = None
            else:
                closed_by = closed_by.strip()
                if closed_by not in batch_messages_by_id:
                    flags.append(f"unsupplied_closed_by:{closed_by}")
                    closed_by = None
                elif snippet_allowed_msgs is not None and closed_by not in snippet_allowed_msgs:
                    flags.append(f"cross_snippet_closed_by:{closed_by}")
                    closed_by = None
                else:
                    closing_msg = batch_messages_by_id[closed_by]
                    source_msg_dict = batch_messages_by_id[source_msg]
                    source_ts = source_msg_dict.get("ts")
                    closing_ts = closing_msg.get("ts")

                    # Do not preserve closed_by unless both source and closing timestamps are valid numbers
                    if not _is_valid_timestamp(source_ts) or not _is_valid_timestamp(closing_ts):
                        flags.append(f"invalid_timestamp_comparison:{closed_by}")
                        closed_by = None
                    elif closing_ts < source_ts:
                        flags.append(f"invalid_closed_by_timestamp:{closed_by}")
                        closed_by = None

        due_date = item.get("due_date")
        if due_date is not None:
            parsed_date = _parse_valid_due_date(due_date)
            if parsed_date is None:
                flags.append(f"invalid_date_format:{due_date}")
                due_date = None
            else:
                due_date = parsed_date

        due_anchor_msg = item.get("due_anchor_msg")
        if due_anchor_msg is not None:
            # Do NOT normalize with .strip(); reject invalid types or whitespace-padded IDs
            if not isinstance(due_anchor_msg, str) or not due_anchor_msg or due_anchor_msg.strip() != due_anchor_msg:
                flags.append(f"invalid_due_anchor_msg:{due_anchor_msg}")
                due_anchor_msg = None
            elif due_anchor_msg not in batch_messages_by_id:
                flags.append(f"unsupplied_due_anchor_msg:{due_anchor_msg}")
                due_anchor_msg = None
            elif snippet_allowed_msgs is not None and due_anchor_msg not in snippet_allowed_msgs:
                flags.append(f"cross_snippet_due_anchor_msg:{due_anchor_msg}")
                due_anchor_msg = None
            else:
                anchor_msg = batch_messages_by_id[due_anchor_msg]
                source_msg_dict = batch_messages_by_id[source_msg]

                # Validate anchor belongs to the same chat as batch and source message
                if anchor_msg.get("chat_id") != batch.chat_id:
                    flags.append(f"cross_chat_due_anchor_msg:{due_anchor_msg}")
                    due_anchor_msg = None
                else:
                    source_ts = source_msg_dict.get("ts")
                    anchor_ts = anchor_msg.get("ts")

                    # Both timestamps must be valid numbers and anchor_ts >= source_ts
                    if not _is_valid_timestamp(source_ts) or not _is_valid_timestamp(anchor_ts):
                        flags.append(f"invalid_timestamp_comparison:{due_anchor_msg}")
                        due_anchor_msg = None
                    elif anchor_ts < source_ts:
                        flags.append(f"invalid_due_anchor_timestamp:{due_anchor_msg}")
                        due_anchor_msg = None

        # Cross-validation:
        # 1. If due_date is provided but anchor is invalid, missing, ungrounded, or cross-chat -> invalidate due_date
        if due_date is not None and due_anchor_msg is None:
            flags.append(f"ungrounded_due_date:{due_date}")
            due_date = None

        # 2. Ensure due_anchor_msg is strictly null whenever due_date is null
        if due_date is None:
            due_anchor_msg = None

        suggested_action = item.get("suggested_action")
        if suggested_action is not None:
            if not isinstance(suggested_action, str) or suggested_action not in VALID_SUGGESTED_ACTIONS:
                flags.append(f"unknown_suggested_action:{suggested_action}")
                suggested_action = None

        confidence = item.get("confidence")
        if confidence is not None:
            validated_conf = _validate_confidence(confidence)
            if validated_conf is None:
                flags.append(f"invalid_confidence:{confidence}")
                confidence = None
            else:
                confidence = validated_conf

        valid_open_loops.append(
            InterpretedOpenLoop(
                source_msg=source_msg,
                chat_id=batch.chat_id,
                type=loop_type,
                is_genuine=True,
                title=title.strip(),
                owner=owner,
                target_person=target_person,
                status=status,
                closed_by=closed_by,
                due_date=due_date,
                suggested_action=suggested_action,
                confidence=confidence,
                validation_flags=flags if flags else None,
                due_anchor_msg=due_anchor_msg,
            )
        )
        if diagnostics is not None:
            diagnostics.open_loops_accepted += 1

    # 4. Parse and validate Goals
    valid_goals: List[InterpretedGoal] = []
    if diagnostics is not None:
        diagnostics.goals_total_raw += len(data["goals"])

    for item in data["goals"]:
        if not isinstance(item, dict):
            if diagnostics is not None:
                diagnostics.goals_non_dict += 1
                diagnostics.record_rejection("goal", "non_dict_record")
            continue

        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            if diagnostics is not None:
                diagnostics.goals_missing_required_fields += 1
                diagnostics.record_rejection("goal", "missing_required_title")
            continue

        area = item.get("area")
        if not isinstance(area, str) or area not in VALID_GOAL_AREAS:
            if diagnostics is not None:
                diagnostics.goals_invalid_enums += 1
                diagnostics.record_rejection("goal", "invalid_area_enum")
            continue

        goal_type = item.get("type")
        if not isinstance(goal_type, str) or goal_type not in VALID_GOAL_TYPES:
            if diagnostics is not None:
                diagnostics.goals_invalid_enums += 1
                diagnostics.record_rejection("goal", "invalid_type_enum")
            continue

        is_genuine = item.get("is_genuine")
        if not isinstance(is_genuine, bool):
            if diagnostics is not None:
                diagnostics.goals_invalid_or_missing_is_genuine += 1
                diagnostics.record_rejection("goal", "invalid_or_missing_is_genuine")
            continue
        if is_genuine is False:
            if diagnostics is not None:
                diagnostics.goals_rejected_not_genuine += 1
                diagnostics.record_rejection("goal", "rejected_not_genuine")
            continue  # Discard non-genuine tactical desires

        evidence_msgs = item.get("evidence_msgs")
        if not isinstance(evidence_msgs, list):
            if diagnostics is not None:
                diagnostics.goals_missing_required_fields += 1
                diagnostics.record_rejection("goal", "missing_evidence_msgs_list")
            continue

        # Filter strictly to messages supplied in the batch
        grounded_evidence = [
            mid for mid in evidence_msgs if isinstance(mid, str) and mid in batch_messages_by_id
        ]
        if not grounded_evidence:
            if diagnostics is not None:
                diagnostics.goals_invalid_or_ungrounded_evidence += 1
                diagnostics.record_rejection("goal", "empty_or_ungrounded_evidence")
            continue  # Goal rejected as ungrounded

        # If multi-snippet batch, ensure evidence messages all belong to the primary snippet
        if getattr(batch, "snippets", None) and len(batch.snippets) > 1:
            first_snip = next(
                (s for s in batch.snippets if any(m.get("id") == grounded_evidence[0] for m in s.messages)),
                None,
            )
            if first_snip is not None:
                first_snip_msg_ids = {m["id"] for m in first_snip.messages if isinstance(m, dict) and m.get("id")}
                grounded_evidence = [mid for mid in grounded_evidence if mid in first_snip_msg_ids]
                if not grounded_evidence:
                    if diagnostics is not None:
                        diagnostics.goals_invalid_or_ungrounded_evidence += 1
                        diagnostics.record_rejection("goal", "cross_snippet_evidence")
                    continue

        # Deterministically normalize null horizon or reject invalid horizon enum
        raw_horizon = item.get("horizon")
        if raw_horizon is None:
            horizon, was_normalized = normalize_goal_horizon(
                raw_horizon=raw_horizon,
                title=title,
                goal_type=goal_type,
                evidence_msgs=grounded_evidence,
                context=context,
                batch_messages_by_id=batch_messages_by_id,
            )
            if diagnostics is not None:
                diagnostics.goals_horizon_normalized += 1
                diagnostics.record_rejection("goal", f"normalized_horizon:{horizon}")
        elif isinstance(raw_horizon, str) and raw_horizon.strip().lower() in VALID_GOAL_HORIZONS:
            horizon = raw_horizon.strip().lower()
        else:
            if diagnostics is not None:
                diagnostics.goals_invalid_enums += 1
                diagnostics.record_rejection("goal", "invalid_horizon_enum")
            continue

        confidence = item.get("confidence")
        if confidence is not None:
            confidence = _validate_confidence(confidence)

        valid_goals.append(
            InterpretedGoal(
                title=title.strip(),
                area=area,
                type=goal_type,
                horizon=horizon,
                evidence_msgs=grounded_evidence,
                is_genuine=True,
                confidence=confidence,
            )
        )
        if diagnostics is not None:
            diagnostics.goals_accepted += 1

    return valid_open_loops, valid_goals


def parse_participant_interpretation_response(
    raw_text: str,
    dossiers: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    chat_metadata: Optional[Dict[str, Any]] = None,
    context: Optional[ConversationContext] = None,
) -> Tuple[List[InterpretedPerson], List[InterpretedOrganisation]]:
    """
    Parse and validate LLM output for participant dossiers into typed records.

    Enforces that:
    - Root JSON contains BOTH 'people' and 'organisations', each an explicit list.
      (Missing key or wrong type raises ResponseStructureError; empty lists [] are valid).
    - Person person_id belongs strictly to the submitted dossiers and is not 'ME'.
    - Person looks_after belongs strictly to participant IDs in the submitted dossiers.
    - Person and organisation evidence_msgs belong strictly to messages in the submitted dossiers.
    - Person associated_org_names are grounded in the submitted dossier text.
    - Organisation associated_chats belong strictly to chat IDs in the submitted dossiers or metadata.
    - Organisation name is non-empty and reliably word-boundary grounded in prompt text.
    - Invalid items are discarded independently without failing valid items in the batch.

    Args:
        raw_text: Raw string returned by the LLM.
        dossiers: Single dossier or sequence of dossiers submitted to the prompt.
        chat_metadata: Optional dictionary describing relevant chat metadata.
        context: Optional ConversationContext for entity existence verification.

    Returns:
        Tuple of (valid_people, valid_organisations).

    Raises:
        ResponseParsingError: If raw_text is not valid, unambiguous JSON.
        ResponseStructureError: If root structure or required collection arrays are invalid.
        ValueError: If dossiers is None or empty.
    """
    if dossiers is None:
        raise ValueError("Dossiers cannot be None")

    if isinstance(dossiers, dict):
        dossiers_list = [dossiers]
    elif isinstance(dossiers, (list, tuple)):
        dossiers_list = list(dossiers)
    else:
        raise ValueError(f"Expected dict or sequence of dicts for dossiers, got {type(dossiers).__name__}")

    if not dossiers_list:
        raise ValueError("Dossiers list cannot be empty")

    data = extract_json_payload(raw_text)

    # 1. Required top-level collection keys
    if "people" not in data or not isinstance(data["people"], list):
        raise ResponseStructureError("Response missing required 'people' list")
    if "organisations" not in data or not isinstance(data["organisations"], list):
        raise ResponseStructureError("Response missing required 'organisations' list")

    # 2. Build prompt-bounded reference sets
    dossier_person_ids: Set[str] = set()
    dossier_message_ids: Set[str] = set()
    dossier_chat_ids: Set[str] = set()
    corpus_parts: List[str] = []

    for d in dossiers_list:
        if not isinstance(d, dict):
            continue
        pid = d.get("person_id") or d.get("participant_id")
        if isinstance(pid, str) and pid.strip():
            dossier_person_ids.add(pid.strip())

        for name in d.get("names", []):
            if isinstance(name, str) and name.strip():
                corpus_parts.append(name.strip())

        for chat in d.get("chats", []):
            if isinstance(chat, dict):
                cid = chat.get("chat_id")
                if isinstance(cid, str) and cid.strip():
                    dossier_chat_ids.add(cid.strip())
                cname = chat.get("name")
                if isinstance(cname, str) and cname.strip():
                    corpus_parts.append(cname.strip())

        for msg in d.get("messages", []):
            if isinstance(msg, dict):
                mid = msg.get("id")
                if isinstance(mid, str) and mid.strip():
                    dossier_message_ids.add(mid.strip())
                body = msg.get("body")
                if isinstance(body, str) and body.strip():
                    corpus_parts.append(body.strip())
                sname = msg.get("sender_name")
                if isinstance(sname, str) and sname.strip():
                    corpus_parts.append(sname.strip())

    if chat_metadata and isinstance(chat_metadata, dict):
        for cid, meta in chat_metadata.items():
            if isinstance(cid, str) and cid.strip():
                dossier_chat_ids.add(cid.strip())
            if isinstance(meta, str) and meta.strip():
                corpus_parts.append(meta.strip())
            elif isinstance(meta, dict):
                for sub_val in meta.values():
                    if isinstance(sub_val, str) and sub_val.strip():
                        corpus_parts.append(sub_val.strip())

    dossier_corpus = " \n ".join(corpus_parts)

    # 3. Parse and validate People
    valid_people: List[InterpretedPerson] = []
    for item in data["people"]:
        if not isinstance(item, dict):
            continue

        person_id = item.get("person_id")
        if not isinstance(person_id, str) or not person_id.strip():
            continue
        person_id = person_id.strip()

        # Cannot be 'ME' and must belong strictly to submitted dossiers
        if person_id == "ME" or person_id not in dossier_person_ids:
            continue

        circle = item.get("circle")
        if circle is not None:
            if isinstance(circle, str) and circle.strip().lower() in VALID_RELATIONSHIP_CIRCLES:
                circle = circle.strip().lower()
            else:
                circle = None

        raw_looks_after = item.get("looks_after")
        if isinstance(raw_looks_after, list):
            # Ground strictly to participant IDs in the submitted dossiers
            looks_after = [
                pid.strip()
                for pid in raw_looks_after
                if isinstance(pid, str) and pid.strip() in dossier_person_ids
            ]
        else:
            looks_after = []

        raw_associated_orgs = item.get("associated_org_names")
        if isinstance(raw_associated_orgs, list):
            # Ground reliably using word-boundary phrase match in dossier corpus
            associated_org_names = [
                org.strip()
                for org in raw_associated_orgs
                if isinstance(org, str) and is_name_grounded_in_text(org.strip(), dossier_corpus)
            ]
        else:
            associated_org_names = []

        org_link_type = item.get("org_link_type")
        if org_link_type is not None:
            if isinstance(org_link_type, str) and org_link_type.strip().lower() in VALID_ORG_LINK_TYPES:
                org_link_type = org_link_type.strip().lower()
            else:
                org_link_type = None

        raw_evidence = item.get("evidence_msgs")
        if isinstance(raw_evidence, list):
            # Ground strictly to messages supplied in the dossiers
            evidence_msgs = [
                mid.strip()
                for mid in raw_evidence
                if isinstance(mid, str) and mid.strip() in dossier_message_ids
            ]
        else:
            evidence_msgs = []

        confidence = item.get("confidence")
        if confidence is not None:
            confidence = _validate_confidence(confidence)

        valid_people.append(
            InterpretedPerson(
                person_id=person_id,
                circle=circle,
                looks_after=looks_after,
                associated_org_names=associated_org_names,
                org_link_type=org_link_type,
                evidence_msgs=evidence_msgs,
                confidence=confidence,
            )
        )

    # 4. Parse and validate Organisations
    valid_orgs: List[InterpretedOrganisation] = []
    for item in data["organisations"]:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        clean_name = name.strip()

        # Reject organisation if name is not reliably grounded via word-boundary phrase match
        if not is_name_grounded_in_text(clean_name, dossier_corpus):
            continue

        org_type = item.get("org_type")
        if org_type is not None:
            if isinstance(org_type, str) and org_type.strip().lower() in VALID_ORG_TYPES:
                org_type = org_type.strip().lower()
            else:
                org_type = None

        side = item.get("side")
        if side is not None:
            if isinstance(side, str) and side.strip().lower() in VALID_ORG_SIDES:
                side = side.strip().lower()
            else:
                side = None

        role = item.get("role")
        if role is not None:
            if isinstance(role, str) and role.strip().lower() in VALID_ORG_ROLES:
                role = role.strip().lower()
            else:
                role = None

        raw_evidence = item.get("evidence_msgs")
        if isinstance(raw_evidence, list):
            evidence_msgs = [
                mid.strip()
                for mid in raw_evidence
                if isinstance(mid, str) and mid.strip() in dossier_message_ids
            ]
        else:
            evidence_msgs = []

        raw_chats = item.get("associated_chats")
        if isinstance(raw_chats, list):
            associated_chats = [
                cid.strip()
                for cid in raw_chats
                if isinstance(cid, str) and cid.strip() in dossier_chat_ids
            ]
        else:
            associated_chats = []

        confidence = item.get("confidence")
        if confidence is not None:
            confidence = _validate_confidence(confidence)

        valid_orgs.append(
            InterpretedOrganisation(
                name=clean_name,
                org_type=org_type,
                side=side,
                role=role,
                evidence_msgs=evidence_msgs,
                associated_chats=associated_chats,
                confidence=confidence,
            )
        )

    return valid_people, valid_orgs
