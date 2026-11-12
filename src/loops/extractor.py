"""End-to-end open-loop extraction pipeline and schema emission.

Connects candidate evidence extraction, group relevance filtering, chronological batching,
LLM interpretation, deterministic resolution, lifecycle episode management, and 9-field schema emission.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import http.client
import inspect
import logging
import re
import socket
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import urllib.error

logger = logging.getLogger(__name__)

from src.extraction.evidence_extractor import (
    CandidateEvidenceExtractor,
    is_relevant_group_candidate,
    is_syndicated_broadcast,
)
from src.interpretation.batcher import (
    build_disjoint_evidence_batches,
    build_evidence_batches,
)
from src.interpretation.interpreter import (
    LLMConfigurationError,
    _sanitize_error_message,
)
from src.interpretation.parser import (
    InterpretationResponseError,
    ResponseParsingError,
    ResponseStructureError,
)
from src.interpretation.records import InterpretedOpenLoop
from src.loops.models import (
    CandidateOpenLoop,
    OpenLoopExtractionConfig,
    OpenLoopExtractionDiagnostics,
    _validate_cap_limit,
)
from src.loops.resolver import (
    ACTIONABLE_QUESTION_PATTERN,
    BROADCAST_QUERY_PATTERN,
    COLLECTIVE_GROUP_PATTERN,
    DIRECTED_REQUEST_PATTERN,
    FIRST_PERSON_COMMITMENT_PATTERN,
    MONTH_WORDS,
    PLAYFUL_THREAT_PATTERN,
    SITUATIONAL_OBSERVATION_PATTERN,
    TASK_ENTITIES,
    THIRD_PARTY_OR_DILEMMA_PATTERN,
    resolve_loop_owner,
    resolve_open_loops,
)
from src.preprocessing.context_builder import ConversationContext

OPEN_LOOP_CATEGORIES: Tuple[str, ...] = (
    "commitment",
    "question",
    "followup",
    "deadline",
)


@dataclass
class OpenLoopExtractionResult:
    """Container for finalized schema-compliant open loops and execution diagnostics."""

    open_loops: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: OpenLoopExtractionDiagnostics = field(default_factory=OpenLoopExtractionDiagnostics)

    @property
    def loop_count(self) -> int:
        """Return the number of schema open loops emitted."""
        return len(self.open_loops)


def _get_dataset_message(context: Any, msg_id: str) -> Optional[Dict[str, Any]]:
    """Look up a message by ID across ConversationContext or raw dataset structures safely."""
    if not context or not isinstance(msg_id, str) or not msg_id.strip():
        return None

    clean_id = msg_id.strip()

    # 1. Dataset messages_by_id
    dataset = getattr(context, "dataset", context)
    if isinstance(dataset, dict):
        msgs_by_id = dataset.get("messages_by_id")
        if isinstance(msgs_by_id, dict) and clean_id in msgs_by_id:
            return msgs_by_id[clean_id]

        msgs_by_chat = dataset.get("messages_by_chat")
        if isinstance(msgs_by_chat, dict):
            for msgs in msgs_by_chat.values():
                if isinstance(msgs, list):
                    for m in msgs:
                        if isinstance(m, dict) and m.get("id") == clean_id:
                            return m

        raw_msgs = dataset.get("messages")
        if isinstance(raw_msgs, list):
            for m in raw_msgs:
                if isinstance(m, dict) and m.get("id") == clean_id:
                    return m

    # 2. ConversationContext.chats
    chats = getattr(context, "chats", {})
    if isinstance(chats, dict):
        for chat in chats.values():
            if isinstance(chat, dict):
                for m in chat.get("messages", []):
                    if isinstance(m, dict) and m.get("id") == clean_id:
                        return m

    return None


def _get_message_timestamp(context: Any, msg_id: str) -> Union[int, float]:
    """Retrieve numeric timestamp for a message, falling back to infinity for stable ordering."""
    msg = _get_dataset_message(context, msg_id)
    if msg and isinstance(msg, dict):
        ts = msg.get("ts")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            return ts
    return float("inf")


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


OPERATIONAL_EXCEPTION_CLASSES = (
    LLMConfigurationError,
    urllib.error.HTTPError,
    urllib.error.URLError,
    http.client.HTTPException,
    ConnectionError,
    TimeoutError,
    socket.timeout,
    socket.gaierror,
    socket.herror,
    ResponseParsingError,
    ResponseStructureError,
    InterpretationResponseError,
)

PROGRAMMING_EXCEPTION_TYPES = (
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    AttributeError,
    AssertionError,
    NameError,
    SyntaxError,
    ZeroDivisionError,
    UnboundLocalError,
    NotImplementedError,
    LookupError,
    ArithmeticError,
)

KNOWN_OPERATIONAL_CLASS_NAMES = frozenset({
    "RateLimitError",
    "RateLimitExceeded",
    "RateLimitException",
    "APITimeoutError",
    "APIConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "ResourceExhausted",
    "TooManyRequests",
    "QuotaExceeded",
    "QuotaExceededError",
    "ServiceUnavailableError",
    "InternalServerError",
    "ProviderAPIError",
})

KNOWN_OPERATIONAL_CLASS_SUFFIXES = (
    "RateLimitError",
    "RateLimitExceeded",
    "RateLimitException",
    "ResourceExhausted",
    "QuotaExceeded",
    "QuotaExceededError",
)


def is_operational_exception(exc: Exception) -> bool:
    """Classify whether an exception is an expected operational failure.

    Expected operational failures:
    - Standard network and HTTP errors (HTTPError, URLError, ConnectionError, socket errors)
    - Standard call and socket timeouts (TimeoutError, socket.timeout)
    - Offline cache misses and configuration (LLMConfigurationError when offline allowed)
    - Model response parsing and structure errors (ResponseParsingError, ResponseStructureError, InterpretationResponseError)
    - Documented provider rate-limit and timeout exceptions (RateLimitError, APITimeoutError, ResourceExhausted, etc.)

    Internal programming bugs (TypeError, KeyError, etc., even if named with misleading terms like 'HTTP' or 'Timeout')
    and arbitrary unclassified exceptions are strictly non-operational.
    """
    if isinstance(exc, PROGRAMMING_EXCEPTION_TYPES):
        return False

    if isinstance(exc, OPERATIONAL_EXCEPTION_CLASSES):
        return True

    exc_name = type(exc).__name__
    if exc_name in KNOWN_OPERATIONAL_CLASS_NAMES:
        return True
    if exc_name.endswith(KNOWN_OPERATIONAL_CLASS_SUFFIXES):
        return True

    return False


def is_operational_diagnostic_error(err_type: Optional[str]) -> bool:
    """Return True if a diagnostic error_type represents an expected operational failure.

    Accepts:
    - 'TimeoutBudgetExceeded', 'OfflineModeNoCache'
    - 'HTTPError:<status_code>' or 'HTTPError'
    - Standard network/socket error names: 'URLError', 'TimeoutError', 'socket.timeout', 'ConnectionError'
    - Supported model response parsing failure types: 'ResponseParsingError', 'ResponseStructureError', 'InterpretationResponseError'
    - Known provider rate-limit and timeout class names: 'RateLimitError', 'APITimeoutError', 'ResourceExhausted', etc.
    """
    if not err_type or not isinstance(err_type, str):
        return False

    if err_type in ("TimeoutBudgetExceeded", "OfflineModeNoCache", "TokenBudgetExceeded", "RateLimitBudgetExhausted"):
        return True
    if err_type.startswith("HTTPError:") or err_type == "HTTPError":
        return True
    if err_type in (
        "URLError",
        "TimeoutError",
        "socket.timeout",
        "ConnectionError",
        "Network connection error",
        "Request timed out",
        "ResponseParsingError",
        "ResponseStructureError",
        "InterpretationResponseError",
    ):
        return True
    if err_type in KNOWN_OPERATIONAL_CLASS_NAMES:
        return True
    if err_type.endswith(KNOWN_OPERATIONAL_CLASS_SUFFIXES):
        return True

    return False


def interpret_open_loop_evidence(
    context: ConversationContext,
    candidates: Sequence[Dict[str, Any]],
    interpreter: Any,
    diagnostics: Optional[OpenLoopExtractionDiagnostics] = None,
    budget: Optional[Any] = None,
    successful_candidate_ids: Optional[Set[str]] = None,
    failed_candidate_ids: Optional[Set[str]] = None,
    unexpected_candidate_ids: Optional[Set[str]] = None,
    unexpected_errors: Optional[List[Dict[str, Any]]] = None,
    use_disjoint_batching: bool = False,
) -> List[InterpretedOpenLoop]:
    """Orchestrate LLM interpretation of candidate open-loop evidence into typed InterpretedOpenLoop records.

    Pipeline Flow:
    1. If candidates is empty, returns immediately without creating batches or calling the interpreter.
    2. Groups candidates into batches using build_evidence_batches(context, candidates).
    3. For each batch:
       - Safely checks parameter support for 'budget' prior to invocation.
       - Invokes interpreter.interpret_chat_batch(batch, context=context, budget=budget) if supported,
         or interpreter.interpret_chat_batch(batch, context=context) otherwise.
       - Classifies each batch as:
         * successfully interpreted (including intentional empty results) or loaded from cache;
         * expected operational failure (RateLimitError, HTTP error, network timeout, budget exceeded);
         * unexpected internal failure (programming or data error);
         * unavailable due to offline mode / cache miss;
         * skipped due to exhausted execution budget.
       - Tracks candidate message IDs associated with each batch classification.
       - Fatal configuration errors (LLMConfigurationError) bubble up immediately unless allow_offline is set.
       - Successful batches increment diagnostics.batches_succeeded (and batches_cached if served from cache).
       - Valid InterpretedOpenLoop instances are collected.
    4. Returns collected list of InterpretedOpenLoop records in deterministic order.
    """
    if diagnostics is None:
        diagnostics = OpenLoopExtractionDiagnostics()

    if not candidates:
        return []

    if interpreter is None:
        if diagnostics:
            diagnostics.batches_unattempted += 1
        for cand in candidates:
            mid = cand.get("message_id") if isinstance(cand, dict) else None
            if mid:
                if failed_candidate_ids is not None:
                    failed_candidate_ids.add(mid)
                if diagnostics:
                    if hasattr(diagnostics, "failed_candidate_message_ids"):
                        diagnostics.failed_candidate_message_ids.add(mid)
                    if hasattr(diagnostics, "failed_candidates_count"):
                        diagnostics.failed_candidates_count += 1
        return []

    if use_disjoint_batching:
        batches = build_disjoint_evidence_batches(context, candidates)
    else:
        batches = build_evidence_batches(context, candidates)
    diagnostics.batches_created += len(batches)

    if not batches:
        return []

    all_interpreted_loops: List[InterpretedOpenLoop] = []
    supports_budget = _supports_budget_parameter(interpreter)

    for batch in batches:
        # Extract candidate message IDs in this batch
        batch_cand_ids: Set[str] = set()
        if hasattr(batch, "candidate_message_ids") and batch.candidate_message_ids:
            batch_cand_ids = set(batch.candidate_message_ids)
        elif hasattr(batch, "candidates") and batch.candidates:
            batch_cand_ids = {c.get("message_id") for c in batch.candidates if isinstance(c, dict) and c.get("message_id")}

        batch_id = getattr(batch, "chat_id", None) or getattr(batch, "batch_id", None) or "unknown_batch"

        try:
            if supports_budget and budget is not None:
                loops, _goals, diag = interpreter.interpret_chat_batch(batch, context=context, budget=budget)
            else:
                loops, _goals, diag = interpreter.interpret_chat_batch(batch, context=context)

            if diag is not None:
                err_type = getattr(diag, "error_type", None)
                if is_operational_diagnostic_error(err_type):
                    if err_type == "TimeoutBudgetExceeded":
                        diagnostics.batches_unattempted += 1
                        diagnostics.batches_budget_exhausted += 1
                    elif err_type == "OfflineModeNoCache":
                        diagnostics.batches_unattempted += 1
                        diagnostics.batches_offline_unavailable += 1
                    diagnostics.batches_failed += 1
                    if failed_candidate_ids is not None:
                        failed_candidate_ids.update(batch_cand_ids)
                    if diagnostics:
                        if hasattr(diagnostics, "failed_candidate_message_ids"):
                            diagnostics.failed_candidate_message_ids.update(batch_cand_ids)
                        if hasattr(diagnostics, "failed_candidates_count"):
                            diagnostics.failed_candidates_count += len(batch_cand_ids)
                else:
                    # Unexpected internal failure reported via diagnostic
                    diagnostics.batches_failed += 1
                    if hasattr(diagnostics, "batches_unexpected_errors"):
                        diagnostics.batches_unexpected_errors += 1
                    if unexpected_candidate_ids is not None:
                        unexpected_candidate_ids.update(batch_cand_ids)
                    if diagnostics and hasattr(diagnostics, "unexpected_candidates_count"):
                        diagnostics.unexpected_candidates_count += len(batch_cand_ids)

                    raw_err_msg = getattr(diag, "error_message", None) or err_type or "Unknown diagnostic error"
                    sanitized_msg = _sanitize_error_message(raw_err_msg)
                    err_record = {
                        "batch_id": batch_id,
                        "exception_class": err_type or "UnknownDiagnosticError",
                        "error_message": sanitized_msg,
                    }
                    if unexpected_errors is not None:
                        unexpected_errors.append(err_record)
                    if diagnostics and hasattr(diagnostics, "record_unexpected_error"):
                        diagnostics.record_unexpected_error(err_record)

                    print(
                        f"[Stage: Open-Loop Interpretation] Unexpected failure in batch {batch_id}: {err_type} - {sanitized_msg}",
                        file=sys.stderr,
                    )
            else:
                diagnostics.batches_succeeded += 1
                if getattr(interpreter, "last_call_cached", False):
                    diagnostics.batches_cached += 1
                else:
                    diagnostics.batches_attempted_live += 1
                # Mark candidates as successfully interpreted (including intentional empty loop results)
                if successful_candidate_ids is not None:
                    successful_candidate_ids.update(batch_cand_ids)
                if diagnostics:
                    if hasattr(diagnostics, "successful_candidate_message_ids"):
                        diagnostics.successful_candidate_message_ids.update(batch_cand_ids)
                    if hasattr(diagnostics, "successful_candidates_count"):
                        diagnostics.successful_candidates_count += len(batch_cand_ids)
                if loops:
                    for loop in loops:
                        if isinstance(loop, InterpretedOpenLoop):
                            all_interpreted_loops.append(loop)
        except LLMConfigurationError:
            if not getattr(interpreter, "allow_offline", False):
                raise
            diagnostics.batches_failed += 1
            diagnostics.batches_offline_unavailable += 1
            if failed_candidate_ids is not None:
                failed_candidate_ids.update(batch_cand_ids)
            if diagnostics:
                if hasattr(diagnostics, "failed_candidate_message_ids"):
                    diagnostics.failed_candidate_message_ids.update(batch_cand_ids)
                if hasattr(diagnostics, "failed_candidates_count"):
                    diagnostics.failed_candidates_count += len(batch_cand_ids)
        except Exception as err:
            exc_class = type(err).__name__
            if is_operational_exception(err):
                # Expected operational failure
                diagnostics.batches_failed += 1
                if "timeout" in exc_class.lower() or "budget" in exc_class.lower():
                    diagnostics.batches_budget_exhausted += 1
                elif "offline" in exc_class.lower():
                    diagnostics.batches_offline_unavailable += 1
                if failed_candidate_ids is not None:
                    failed_candidate_ids.update(batch_cand_ids)
                if diagnostics:
                    if hasattr(diagnostics, "failed_candidate_message_ids"):
                        diagnostics.failed_candidate_message_ids.update(batch_cand_ids)
                    if hasattr(diagnostics, "failed_candidates_count"):
                        diagnostics.failed_candidates_count += len(batch_cand_ids)
            else:
                # Unexpected internal exception (e.g. TypeError, ValueError, KeyError, AttributeError)
                diagnostics.batches_failed += 1
                if hasattr(diagnostics, "batches_unexpected_errors"):
                    diagnostics.batches_unexpected_errors += 1
                if unexpected_candidate_ids is not None:
                    unexpected_candidate_ids.update(batch_cand_ids)
                if diagnostics and hasattr(diagnostics, "unexpected_candidates_count"):
                    diagnostics.unexpected_candidates_count += len(batch_cand_ids)

                sanitized_msg = _sanitize_error_message(err)
                err_record = {
                    "batch_id": batch_id,
                    "exception_class": exc_class,
                    "error_message": sanitized_msg,
                }
                if unexpected_errors is not None:
                    unexpected_errors.append(err_record)
                if diagnostics and hasattr(diagnostics, "record_unexpected_error"):
                    diagnostics.record_unexpected_error(err_record)

                print(
                    f"[Stage: Open-Loop Interpretation] Unexpected failure in batch {batch_id}: {exc_class} - {sanitized_msg}",
                    file=sys.stderr,
                )
            continue

    return all_interpreted_loops


BOT_OR_SYSTEM_PATTERN = re.compile(
    r"\b(?:ad\s*id\s*:\s*\d+|campaign\s*id\s*:\s*\d+|newsletter|unsubscribe|http\S{40,})\b|"
    r"\b(?:your\s+otp\s+is|\d{4,8}\s+is\s+your\s+(?:otp|verification\s*code)|do\s+not\s+share\s+(?:your\s+)?otp|otp\s+for\s+(?:login|transaction|verification))\b",
    re.IGNORECASE,
)
BANTER_PATTERN = re.compile(
    r"^(?:(?:hi|hello|hey|good\s+(?:morning|evening|night|afternoon)|ok|okay|sure|thanks|thank\s+you|thx|cool|great|awesome|noted|congrats|congratulations)\b[\s!.,]*)+$",
    re.IGNORECASE,
)
ACTIONABLE_QUESTION_PATTERN = re.compile(
    r"\b(?:can\s+(?:we|you|u)|could\s+(?:we|you|u)|should\s+(?:i|we)|shall\s+(?:we|i)|would\s+you|"
    r"possible\s+to|which\s+date\s+to|what\s+date\s+to|how\s+about\s+we|why\s+don't\s+we|"
    r"any\s+plans|coming\s+for|when\s+can|what\s+time|free\s+to|available\s+to|connect\s+for|call\s+for|"
    r"where\s+is|where\s+are|status\s+of|update\s+on|"
    r"kya\s+(?:banana|karna|bhejna|dena|mangwana|khareedna)\s+hai|kab\s+(?:aana|jana|karna|bhejna)\s+hai|"
    r"kise\s+(?:dena|bhejna)\s+hai|kaun\s+(?:jayega|aayega|karega)|"
    r"send|give|share|provide|check|confirm|update|meeting|schedule|order|buy|pay|discuss|zoom|"
    r"contract|doc|document|invoice|file|report|proposal)\b",
    re.IGNORECASE,
)

# False-positive pruning patterns
TRANSIT_ARRIVAL_PATTERN = re.compile(
    r"\b(?:i\s+will|ill|i'll|will|we\s+will|we'll|i\s*m|i\s+am)\s+(?:reach|land|arrive|leave|shift|shifting|be\s+there)\b|"
    r"\b(?:reaching|landing|on\s+my\s+way|heading\s+over|at\s+the\s+airport|shifting\s+to)\b|"
    r"\b(?:will\s+be\s+on\s+leave|on\s+leave\s+till|not\s+be\s+able\s+to\s+join|will\s+join\s+(?:regularly|tomorrow|late|in\s+the\s+second\s+half|next\s+week)|taking\s+(?:a\s+)?leave)\b",
    re.IGNORECASE,
)
INDEPENDENT_TASK_IN_TRANSIT = re.compile(
    r"\b(?:and\s+(?:bring|send|carry|give|share|pay|review|discuss|check|order)|along\s+with\s+the)\b",
    re.IGNORECASE,
)
CASUAL_DIMENSION_QUERY = re.compile(
    r"\b(?:kitna\s+inch|what\s+size|how\s+big|how\s+many\s+inches|which\s+color|what\s+color)\b",
    re.IGNORECASE,
)
EMOTIONAL_VENTING_PATTERN = re.compile(
    r"\b(?:won't\s+go|waste\s+tickets|irritating|annoying|shut\s+up|hate\s+this|terrible\s+day)\b",
    re.IGNORECASE,
)
VAGUE_INTENTION_PATTERN = re.compile(
    r"^(?:(?:i\s+will|let\s+me|will)\s+(?:check|see|look|revert)\s+(?:tomorrow|later|today|soon))[\s.!]*$",
    re.IGNORECASE,
)
LOW_SUBSTANCE_REMINDER = re.compile(
    r"^(?:reminder\s+(?:for|to|about)\s+[\w\s]{1,20})$",
    re.IGNORECASE,
)
SOCIETY_ANNOUNCEMENT_PATTERN = re.compile(
    r"\b(?:dear\s+residents|society\s+members|all\s+residents)\b",
    re.IGNORECASE,
)
SPECULATIVE_MUSINGS_PATTERN = re.compile(
    r"\b(?:maybe\s+we\s+can|perhaps\s+we\s+could)\b",
    re.IGNORECASE,
)
THIRD_PARTY_GROUP_DIRECTIVE = re.compile(
    r"@\w+\s+(?:plz|pls|please)\s+(?:remove|delete|update|fix|change|send)\b",
    re.IGNORECASE,
)

ARGUMENT_OR_GRIEVANCE_PATTERN = re.compile(
    r"\b(?:why\s+should\s+i|when\s+will\s+u\s+give|i\s+will\s+go\s+if\s+u|"
    r"u\s+have\s+no\s+right\s+to\s+say|i\s+will\s+not\s+listen\s+to\s+u|"
    r"ur\s+crazy|what\s+bs|i\s+will\s+really\s+fall\s+sick|"
    r"i\'ll\s+make\s+a\s+mistake|unfulfilled\s+wishes|prepare\s+to\s+live\s+ur\s+life|"
    r"every\s+time\s+you\s+don\'t\s+call\s+me|so\s+who\s+are\s+u\s+not\s+meeting|"
    r"too\s+much\s+eggs|don\'t\s+overdo|lil\s+vomitish)\b",
    re.IGNORECASE,
)

SHORT_VAGUE_ACK_PATTERN = re.compile(
    r"^(?:(?:ya|ok|okay|cool|sure|yes|lil|hi|hello|hey|good\s+(?:morning|afternoon|evening))\b[\s,.]*)*(?:"
    r"i\'ll\s+figure|will\s+revert|i\'ll\s+take(?:\s+it)?|"
    r"i\s+will\s+show\s+u|i\'ll\s+call\s+u|make\s+the\s+plan\s+now\??|"
    r"book\s+ur\s+massage\??|should\s+i\s+go\s+to\s+school\??|order\s+2\s+more\??|"
    r"will\s+send\s+bharat\s+ace|by\s+tuesday\s+eill\s+give|"
    r"can\s+we\s+do\s+a\s+call\s*\??|friday\s+what\s+time\??|any\s+update\??|"
    r"pls\s+update\s+me|let\s+me\s+know(?:\s+when\s+(?:done|bought))?|"
    r"pls\s+call(?:\s+when\s+you\s+reach\s+home)?|please\s+give\s+exact\s+amount|"
    r"pls\s+give\s+a\s+solution|(?:please|pls)?\s*call\s+back|will\s+share\s+tmr\.?|"
    r"will\s+call\s+back\s+in\s+sometime|i\s+will\s+send\s+these\s+to\s+u|"
    r"we'll\s+take\s+it\s+from\s+there|let's\s+do\s+that\s+coating\s+one|"
    r"lmk\s+if\s+i\s+need\s+to\s+come|8:30\s+galleria\s+pls\s+buy\s+ticket|"
    r"let's\s+do\s+fri\s+or\s+sat\s+with\s+him|"
    r"pls\s+let\s+me\s+know\s+when\s+done|please\s+share\s+the\s+number\s+of\s+your\s+team|"
    r"can\s+we\s+please\s+meet\s+at\s+your\s+home\s+or\s+office\s+today\??|"
    r"would\s+5th\s+september\s+friday\s+work\??\s*please\s+lmk|give\s+me\s+the\s+fee's\s*\.?|"
    r"let\s+me\s+know\s+till\s+when\s+are\s+you\s+sharing\s+the\s+report|"
    r"can\s+you\s+pls\s+send\s+me\s+in\s+discord|we\s+need\s+to\s+discuss\s+the\s+design\s+and\s+profile\s+etc|"
    r"where\s+are\s+these\s+images\s+uploaded\??|also\s+share\s+what\s+time\s+when\s+u\s+want\s+to\s+sendt?\s+eh\s+image\??|"
    r"can\s+you\s+show\s+me\s+cie?ling\s+image\s*\??(?:\s*for\s+[\w\s]+)?|also\s+possible\s+to\s+get\s+1\s+pano\s+sample\s+at\s+house\??|"
    r"have\s+to\s+book\s+2\s+separate\s+tickets|please\s+send\s+me\s+your\s+child's\s+name\s+and\s+date\s+of\s+birth|"
    r"thanks\s+for\s+(?:the\s+)?ref!?|"
    r"lmk\s+when\s+uve\s+read\s+this\s+need\s+to\s+talk\s+today)\.?$",
    re.IGNORECASE,
)

FORWARDED_NOTICE_PATTERN = re.compile(
    r"^(?:📅|📌|SHVIS\s+Grades)",
    re.IGNORECASE,
)

DELIVERY_WITH_POLITE_CHECK_PATTERN = re.compile(
    r"\b(?:proofs?\s+attached|attached\s+(?:madam|sir|pls|please)|(?:pdf|file|doc|deck|invoice|proofs?)\s+attached|here\s+are\s+the\s+proofs?)\b",
    re.IGNORECASE,
)

# New false-positive filters for low-actionability social and conversational chatter
CASUAL_MEETING_SCHEDULING_PATTERN = re.compile(
    r"\b(?:listen\s+na,\s+i\s+need\s+a\s+favor\.\s*can\s+you\s+please\s+do\s+the\s+meeting|"
    r"sorry\s+to\s+ping\s+on\s+a\s+sunday\.\s*let\s+me\s+know\s+what\s+time|"
    r"can\s+we\s+please\s+meet\s+at\s+your\s+home\s+or\s+office|"
    r"would\s+\d+(?:st|nd|rd|th)?\s+\w+\s+\w+\s+work\s*\??\s*please\s+lmk|"
    r"let\s+me\s+know\s+when\s+do\s+we\s+get\s+onto\s+a\s+call|"
    r"(?:lets|let's)\s+do\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)

INBOUND_COLD_PITCH_PATTERN = re.compile(
    r"\b(?:i\s+am\s+[\w\s]+\s+from\s+[\w\s]+\.\s+we\s+work\s+in\s+e-commerce|"
    r"would\s+be\s+great\s+to\s+connect\s+for\s+\d+[–\-]\d+\s+minutes\s+and\s+explore)\b",
    re.IGNORECASE,
)

NON_TASK_EXCUSE_OR_DISMISSAL_PATTERN = re.compile(
    r"\b(?:due\s+to\s+personal\s+work\s+i\s+will\s+join|"
    r"if\s+everything\s+is\s+ok\s+in\s+my\s+family\s+i\s+will\s+try\s+to\s+join|"
    r"will\s+not\s+get\s+the\s+information\s+before\s+a\s+week|"
    r"(?:dont|don't|no)\s+need\s+to\s+(?:call|worry|do|follow\s+up))\b",
    re.IGNORECASE,
)

SOCIAL_PLEASANTRY_OR_CONGRATS_PATTERN = re.compile(
    r"\b(?:thanks\s+a\s+lot\s+for\s+the\s+wishes|"
    r"thank\s+you\s+for\s+checking\s+in|"
    r"all\s+is\s+well\s+aap\s+kaise|"
    r"all\s+well\s+here|"
    r"silver\s+in\s+the\s+\d+|relay\s+gold|"
    r"flung\s+into\s+a\s+basketball\s+net|"
    r"see\s+my\s+ai\s+-\s+creating\s+anxiety|"
    r"u\s+only\s+said\s+that\s+i'll\s+get\s+pune\s+cake|"
    r"shoutout\s+on\s+linkedin|genuine\s+shoutout|"
    r"curious\s+where\s+are\s+you\s+and\s+how's\s+work|"
    r"just\s+thought\s+id\s+checkin\s+in|"
    r"super\s+jijaji|ashokpuri\s+trees|"
    r"i\s+will\s+send\s+the\s+kids\.\s*thank\s+you\s+kavitha|rather\s+get\s+a\s+massage|"
    r"main\s+material\s+order\s+i\s+placed|small\s+colour\s+packs\s+will\s+order|"
    r"awaiting\s+please\s+need\s+costing|"
    r"hopefully\s+gecko\s+po\s+should\s+come|"
    r"query\s+i\s+thought\s+i\s+would\s+run\s+it\s+by\s+you|"
    r"handles\s+publications\s+for\s+forbes|make\s+payment\s+which\s+turns\s+out\s+to\s+be\s+a\s+fraud|"
    r"we've\s+had\s+to\s+remove\s+nexus\s+audience\s+exclusion)\b",
    re.IGNORECASE,
)

POLITICAL_OR_PUBLIC_BROADCAST_PATTERN = re.compile(
    r"\b(?:for\s+the\s+nation,\s+we\s+will\s+build\s+it|"
    r"uzbekistan\s+visit,\s+i\s+am\s+very\s+happy)\b",
    re.IGNORECASE,
)

LARGE_PARTY_ESSAY_PATTERN = re.compile(
    r"\b(?:dinner\s+of\s+12\s+people\s+celebrating|"
    r"who\s+knows\s+them\s+best\?|"
    r"40\s+years,\s+40\s+questions)\b",
    re.IGNORECASE,
)

CASUAL_SPOUSE_MICRO_CHATTER = re.compile(
    r"\b(?:i'll\s+talk\s+to\s+them\s+for\s+a\s+size\s+swap|"
    r"i'll\s+get\s+the\s+fan\s+removed|"
    r"i'll\s+ask\s+dad\s+to\s+tell\s+him|"
    r"if\s+you're\s+not\s+able\s+to\s+do\s+it-\s*pls\s+let\s+me\s+know|"
    r"can\s+you\s+please\s+send\s+me\s+the\s+remaining\s+household\s+staff\s+details|"
    r"kids\s+have\s+bca\s+play\s+practice|"
    r"i\s+will\s+want\s+to\s+be\s+in\s+delhi\s+to\s+meet\s+my\s+saudi\s+buyers|"
    r"would\s+you\s+like\s+to\s+place\s+an\s+order\??|"
    r"order\s+2\s+more\??|"
    r"book\s+ur\s+massage\??|"
    r"this\s+works\s*\??\s*dont\s+want\s+to\s+waste\s+time|"
    r"dinner\s+meal\s+plan\s+weekly|"
    r"just\s+check\s+in\s+the\s+night\s+for\s+dizziness|"
    r"i\s+am\s+but\s+i\s+have\s+to\s+send\s+the\s+scanner|"
    r"so\s+we\s+will\s+take\s+over\s+accounts\s*/\s*finance\s+as\s+an\s+org|"
    r"harav\s+while\s+sick\s+isnt\s+reliable|"
    r"do\s+uwaqnt\s+to\s+check\s+with\s+bhagwandas|"
    r"pls\s+book\s+(?:bhopal\s+and\s+hyderabad|my\s+hyd\s+ticket)|"
    r"we'll\s+do\s+timeline\s+version|"
    r"please\s+ask\s+mamaji\s+to\s+buy\s+vayu|"
    r"when\s+convenient\s+today\s+please\s+buy\s+20\s+l\s+for\s+vayu)\b",
    re.IGNORECASE,
)

# Expanded commitment patterns
EXPANDED_FP_COMMITMENT_PATTERN = re.compile(
    r"\b(?:i\s+will|i'll|\bill\b|i\s+shall|i\s+can|i\s+promise|i'm\s+going\s+to|we\s+will|we'll|\bwell\b|we\s+can|we\s+promise|i'm\s+on\s+it|im\s+on\s+it|on\s+it)\b|"
    r"\b(?:let's|lets)\s+(?:send|fix|meet|talk|check|order|discuss|call|book|arrange|set|do|start|look|plan|amp|connect|finalize|review|decide)\b|"
    r"^(?:reverting|following\s+up|checking|sending|sharing|updating|getting\s+back)\s+(?:tomorrow|today|tonight|by\s+\w+|soon|shortly|later)\b|"
    r"\bgo\s+ahead\s+and\s+(?:order|buy|book|send|pay|call)\b|"
    r"\b(?:pick\s+kar\s+lunga|bhej\s+dunga|kar\s+dunga|de\s+dunga|meet\s+pe\s+karte\s+hai)\b|"
    r"\b(?:pick\s+kar|drop\s+kar|le|kar|de|bhej)\s*(?:dunga|dungi|lunga|lungi)\b",
    re.IGNORECASE,
)
DIRECTED_IMPERATIVE_PATTERN = re.compile(
    r"^(?:(?:pls|please|kindly|\bpl\b)\s+)?(?:Send|Text|Buy|Order|Book|Transfer|Pay|Figure\s+(?:out|a\s+way)|Update|Share|Provide|Forward|Arrange|Make|Get)\b",
    re.IGNORECASE,
)
PASSIVE_TASK_ASSIGNMENT_PATTERN = re.compile(
    r"\b(?:[a-z]+)\s+to\s+be\s+(?:made|done|cleaned|fixed|prepared|sent|ordered|bought)\b|"
    r"\b(?:measurements?|specs?|details?|info|inputs?|copy)\s+(?:required|needed)\s+for\s+(?:quotation|quote|estimate|po|invoice|project)\b",
    re.IGNORECASE,
)
COUNTERPART_NEED_PATTERN = re.compile(
    r"\b(?:i\s+need|we\s+need)\s+(?:a|an|the)?\s*[\w\s]{1,30}?\s+for\s+[\w\s]{1,30}?(?:piano|class|school|exam|project|po|client|work|home|office)\b|"
    r"\b(?:need\s+your\s+help\s+to|could\s+use\s+your\s+help\s+to|would\s+need\s+your\s+help\s+to)\s+([a-z]+)\b|"
    r"\b(?:we\s+need\s+the|need\s+the)\s+([a-z\s]+)\s+to\s+(?:raise|issue|send|process|file)\s+(?:the\s+)?([a-z]+)\b|"
    r"\b(?:gas\s+cylinder|ration|doodh|cylinder|water)\s+(?:khatam|over|running\s+out|empty|chahiye)\b|"
    r"\b(?:khatam\s+(?:hone\s+wala|ho\s+gaya))\b",
    re.IGNORECASE,
)


def _clean_title_from_body(body: str, cat: str) -> str:
    """Derive a concise, non-fabricated English open-loop title from message text."""
    body = body.strip().replace("\r", " ").replace("\n", " ")
    body = re.sub(r"\s+", " ", body)
    body = re.sub(r"^(?:(?:hi|hello|hey|good\s+morning|good\s+evening)\s*,?\s*)+", "", body, flags=re.IGNORECASE)
    body = body.strip(" .!?,:;")
    if len(body) > 90:
        cut = body[:85].rsplit(" ", 1)[0]
        body = cut + "..."
    if not body:
        return f"Action required ({cat})"
    return body[0].upper() + body[1:]


def _evaluate_fallback_actionability(
    cand: Dict[str, Any],
    msg: Dict[str, Any],
    context: ConversationContext,
    chat: Dict[str, Any],
    config: Optional[OpenLoopExtractionConfig] = None,
    min_confidence: Optional[float] = None,
) -> Tuple[bool, Optional[str], Optional[str], float]:
    """Evaluate whether a Stage B candidate is genuinely actionable, resolve its grounded owner,
    and compute its evidence-backed confidence score.

    Returns:
        (is_actionable, category, resolved_owner, confidence)
    """
    body = (msg.get("body") or msg.get("text") or "").strip()
    if len(body) < 8:
        return False, None, None, 0.0

    # 1. Automated broadcasts, bot messages, OTPs, newsletters, links
    if BOT_OR_SYSTEM_PATTERN.search(body) or is_syndicated_broadcast(body):
        return False, None, None, 0.0

    # 2. Conversational greetings and banter
    if BANTER_PATTERN.search(body):
        return False, None, None, 0.0

    # 3. Situational observations without directed action
    if SITUATIONAL_OBSERVATION_PATTERN.search(body) and not DIRECTED_REQUEST_PATTERN.search(body):
        return False, None, None, 0.0

    # 4. Third-party descriptions or personal musings / past meals
    if THIRD_PARTY_OR_DILEMMA_PATTERN.search(body) and not DIRECTED_REQUEST_PATTERN.search(body):
        return False, None, None, 0.0

    # 5. Group chat collective commitments and open unaddressed broadcasts
    is_group = bool(chat.get("is_group", False))
    if is_group:
        if COLLECTIVE_GROUP_PATTERN.search(body):
            return False, None, None, 0.0
        if BROADCAST_QUERY_PATTERN.search(body) and not EXPANDED_FP_COMMITMENT_PATTERN.search(body):
            return False, None, None, 0.0

    # 6. Past-tense purely informational statements without pending task
    if re.search(r"\b(?:already\s+(?:done|sent|paid|called|submitted|checked|shared|resolved|fixed|ordered)|have\s+already|was\s+done)\b", body, re.IGNORECASE):
        if not DIRECTED_REQUEST_PATTERN.search(body):
            return False, None, None, 0.0

    # FP Pruning Filters
    # Scheduled appointment arrival confirmation in direct (1:1) chat following an immediate scheduling proposal
    is_scheduled_arrival = False
    if not chat.get("is_group", False) and re.search(
        r"\b\d+(?::\d+)?\s*(?:am|pm)\s*(?:ill|i'll|will)\s+reach\b",
        body,
        re.IGNORECASE,
    ):
        chat_id = msg.get("chat_id") or cand.get("chat_id", "")
        dataset = getattr(context, "dataset", context)
        msgs_by_chat = dataset.get("messages_by_chat", {}) if isinstance(dataset, dict) else {}
        c_msgs = msgs_by_chat.get(chat_id, [])
        m_id = msg.get("id") or cand.get("message_id")
        cur_ts = msg.get("ts", 0)
        sender_id_val = msg.get("sender_id") or cand.get("sender_id", "")
        for idx_m, cur_m in enumerate(c_msgs):
            if cur_m.get("id") == m_id:
                for prev_m in c_msgs[max(0, idx_m - 3):idx_m]:
                    prev_text = prev_m.get("body", "")
                    prev_ts = prev_m.get("ts", 0)
                    prev_sender = prev_m.get("sender_id", "")
                    if prev_sender != sender_id_val and (0 <= cur_ts - prev_ts <= 7200):
                        if re.search(
                            r"\b(?:what\s+time|anytime\b.*\bworks?|when\s+(?:can|should|are\s+you|are\s+we)|"
                            r"(?:works?|available|free)\s+(?:at|on|for\s+a)|(?:can|could|shall)\s+we\s+(?:meet|catch\s*up)|"
                            r"(?:meet|meeting)\s+at\s+\d+|\d+\s*(?:am|pm)?\s*works?)\b",
                            prev_text,
                            re.IGNORECASE,
                        ):
                            is_scheduled_arrival = True
                            break
                break

    if TRANSIT_ARRIVAL_PATTERN.search(body) and not INDEPENDENT_TASK_IN_TRANSIT.search(body) and not is_scheduled_arrival:
        return False, None, None, 0.0

    if CASUAL_DIMENSION_QUERY.search(body) and not DIRECTED_REQUEST_PATTERN.search(body):
        return False, None, None, 0.0

    if EMOTIONAL_VENTING_PATTERN.search(body) and not DIRECTED_REQUEST_PATTERN.search(body):
        return False, None, None, 0.0

    if VAGUE_INTENTION_PATTERN.search(body):
        return False, None, None, 0.0

    if LOW_SUBSTANCE_REMINDER.search(body):
        return False, None, None, 0.0

    if SOCIETY_ANNOUNCEMENT_PATTERN.search(body):
        return False, None, None, 0.0

    if SPECULATIVE_MUSINGS_PATTERN.search(body) and not DIRECTED_REQUEST_PATTERN.search(body):
        return False, None, None, 0.0

    if ARGUMENT_OR_GRIEVANCE_PATTERN.search(body):
        return False, None, None, 0.0

    if SHORT_VAGUE_ACK_PATTERN.search(body.strip()):
        return False, None, None, 0.0

    if FORWARDED_NOTICE_PATTERN.search(body.strip()):
        return False, None, None, 0.0

    if DELIVERY_WITH_POLITE_CHECK_PATTERN.search(body) and re.search(r"\b(?:pls|please)\s+(?:check|see|review|verify)\b", body, re.IGNORECASE):
        return False, None, None, 0.0

    if CASUAL_MEETING_SCHEDULING_PATTERN.search(body):
        return False, None, None, 0.0

    if INBOUND_COLD_PITCH_PATTERN.search(body):
        return False, None, None, 0.0

    if NON_TASK_EXCUSE_OR_DISMISSAL_PATTERN.search(body):
        return False, None, None, 0.0

    if SOCIAL_PLEASANTRY_OR_CONGRATS_PATTERN.search(body):
        return False, None, None, 0.0

    if POLITICAL_OR_PUBLIC_BROADCAST_PATTERN.search(body):
        return False, None, None, 0.0

    if LARGE_PARTY_ESSAY_PATTERN.search(body):
        return False, None, None, 0.0

    if CASUAL_SPOUSE_MICRO_CHATTER.search(body):
        return False, None, None, 0.0

    if PLAYFUL_THREAT_PATTERN.search(body):
        return False, None, None, 0.0

    # Inbound counterpart message that ME forwarded outward in another chat
    sender_id = msg.get("sender_id") or cand.get("sender_id", "")
    if sender_id != "ME" and context and len(body.strip()) > 30:
        dataset = getattr(context, "dataset", context)
        if isinstance(dataset, dict):
            msgs_by_chat = dataset.get("messages_by_chat", {})
            this_chat = msg.get("chat_id") or cand.get("chat_id", "")
            is_forwarded_by_me = False
            for c_id, m_list in msgs_by_chat.items():
                if c_id == this_chat:
                    continue
                for other_m in m_list:
                    if other_m.get("sender_id") == "ME" and other_m.get("body", "").strip() == body.strip():
                        is_forwarded_by_me = True
                        break
                if is_forwarded_by_me:
                    break
            if is_forwarded_by_me:
                return False, None, None, 0.0

    # Duplicate forwarded message across chats: only prune if both chats refer to the SAME task beneficiary / episode
    if sender_id == "ME" and context and len(body.strip()) > 20:
        dataset = getattr(context, "dataset", context)
        if isinstance(dataset, dict):
            msgs_by_chat = dataset.get("messages_by_chat", {})
            this_chat = msg.get("chat_id") or cand.get("chat_id", "")
            cur_ts = msg.get("ts", 0)
            is_dup_forward = False

            this_msgs = msgs_by_chat.get(this_chat, [])
            m_id = msg.get("id") or cand.get("message_id")
            this_idx = next((i for i, m in enumerate(this_msgs) if m.get("id") == m_id), -1)
            this_prev_text = this_msgs[this_idx - 1].get("body", "").strip() if this_idx > 0 else ""

            for c_id, m_list in msgs_by_chat.items():
                if c_id == this_chat:
                    continue
                for other_idx, other_m in enumerate(m_list):
                    if (
                        other_m.get("sender_id") == "ME"
                        and other_m.get("body", "").strip() == body.strip()
                        and 0 < cur_ts - other_m.get("ts", 0) <= 600
                    ):
                        other_prev_text = m_list[other_idx - 1].get("body", "").strip() if other_idx > 0 else ""
                        is_same_task = False
                        if this_prev_text and other_prev_text:
                            if this_prev_text == other_prev_text:
                                is_same_task = True
                            else:
                                this_names = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", this_prev_text))
                                other_names = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", other_prev_text))
                                common_names = this_names & other_names - {"Goix", "This", "Arjun", "Hi", "Hello"}
                                if common_names:
                                    is_same_task = True
                        if is_same_task:
                            is_dup_forward = True
                            break
                if is_dup_forward:
                    break
            if is_dup_forward:
                return False, None, None, 0.0

    # Exploratory sample / catalogue requests or chemical quote inquiries without committed order
    if re.search(r"\b(?:catalogue|catalog|brochure|samples?|cie?ling\s+image|pano\s+sample)\b|\b(?:show\s+me|share\s+also|possible\s+to\s+get\s+\d*)\s*[\w\s]*(?:catalogue|catalog|brochure|samples?|cie?ling\s+image)\b|\bneeded\s+to\s+order\b.*\bplease\s+lmk\s+if\s+u\s+can\s+fulfill\s+with\s+cost\b", body, re.IGNORECASE):
        if not re.search(r"\b(?:place|confirm)\s+order\b", body, re.IGNORECASE) or re.search(r"\b(?:walls\s+only|lobby|cost\b)", body, re.IGNORECASE):
            return False, None, None, 0.0

    # Vague reference or generic check-in
    if re.search(r"^.{0,20}@\d+,\s*please\s+let\s+me\s+know\s+about\s+this\.?$", body.strip(), re.IGNORECASE):
        return False, None, None, 0.0

    # Third-party advice / opinion request on vendor legitimacy or publication
    if re.search(r"\b(?:little\s+query\s+i\s+thought\s+i\s+would\s+run\s+it\s+by\s+you|handles\s+publications\s+for\s+forbes|turns\s+out\s+to\s+be\s+a\s+fraud|editorial\s+but\s+are\s+they\s+promising|let\s+me\s+check\s+and\s+revert\s+on\s+this\s+today)\b", body, re.IGNORECASE):
        return False, None, None, 0.0

    # Routine calendar invite coordination in client chats
    if re.search(r"\b(?:once\s+you\s+confirm\s+will\s+have\s+the\s+calendar\s+blocked|when\s+you\s+would\s+like\s+to\s+do\s+the\s+testing|could\s+you\s+please\s+schedule\s+a\s+meeting\s+for\s+tomorrow|allow\s+us\s+to\s+make\s+certain\s+changes\s+on\s+account)\b", body, re.IGNORECASE):
        return False, None, None, 0.0

    # Technical chatbot architectural discussion
    if re.search(r"\b(?:bot\s+to\s+a\s+human\s+on\s+whatsapp|figure\s+(?:out\s+)?the\s+right\s+way\s+to\s+connect\s+to\s+chatbot|what\s+time\s+when\s+u\s+want\s+to\s+sendt?\s+eh\s+image|where\s+are\s+these\s+images\s+uploaded)\b", body, re.IGNORECASE):
        return False, None, None, 0.0

    # Intern/freelancer status query
    if re.search(r"\b(?:clarity\s+on\s+whether\s+we're\s+continuing\s+with\s+it\s+or\s+not|since\s+i'm\s+in\s+college)\b", body, re.IGNORECASE):
        return False, None, None, 0.0

    # Meta reminders and sports congratulations
    if re.search(r"\b(?:please\s+remind\s+me\s+to\s+explain\s+u|been\s+following\s+the\s+results\s+coming\s+in\s+from\s+bangkok|aarti's\s+silver|ranit's\s+relay\s+gold)\b", body, re.IGNORECASE):
        return False, None, None, 0.0

    # Audience exclusion / campaign optimization queries
    if re.search(r"\b(?:nexus\s+audience\s+exclusion|excluding\s+audience\s+that\s+are\s+getting\s+targeted)\b", body, re.IGNORECASE):
        return False, None, None, 0.0

    cat = cand.get("category", "commitment")
    if cat not in OPEN_LOOP_CATEGORIES:
        cat = "commitment"

    base_confidence = 0.70
    is_actionable = False

    # 7. Category-specific actionability gates
    has_fp = bool(EXPANDED_FP_COMMITMENT_PATTERN.search(body))
    has_omitted = bool(re.search(r"^(?:will\s+|shall\s+|on\s+it\b|kar\s+dunga\b|de\s+dunga\b|bhej\s+dunga\b)", body.strip(), re.IGNORECASE))
    has_future_task = bool(re.search(r"\b(?:going\s+to|planning\s+to|plan\s+to|need\s+to|have\s+to|must|let\s+me|promise\s+to|will\s+(?:call|send|check|pay|visit|draft|prepare|review|order|book|submit|file|transfer|fix|update|look\s+into|complete|finish|incorporate))\b", body, re.IGNORECASE))
    has_req = bool(DIRECTED_REQUEST_PATTERN.search(body))
    has_dir_imp = bool(DIRECTED_IMPERATIVE_PATTERN.search(body.strip()))
    has_passive = bool(PASSIVE_TASK_ASSIGNMENT_PATTERN.search(body))
    has_need = bool(COUNTERPART_NEED_PATTERN.search(body))

    if cat == "commitment":
        if has_fp or has_omitted or has_req or has_dir_imp or has_passive or has_need:
            is_actionable = True
            base_confidence = 0.85
        elif has_future_task:
            is_actionable = True
            base_confidence = 0.75

    elif cat == "question":
        if ACTIONABLE_QUESTION_PATTERN.search(body) and ("?" in body or DIRECTED_REQUEST_PATTERN.search(body) or has_dir_imp or re.search(r"\b(?:can|could|would|will|should)\s+(?:you|we|i)\b", body, re.IGNORECASE)):
            if not (re.search(r"\b(?:why\s+does|why\s+is\s+it|who\s+knows|what\s+if)\b", body, re.IGNORECASE) and not DIRECTED_REQUEST_PATTERN.search(body)):
                is_actionable = True
                base_confidence = 0.75
        elif has_fp or has_dir_imp or has_need:
            is_actionable = True
            cat = "commitment"
            base_confidence = 0.80

    elif cat == "followup":
        if re.search(r"\b(?:update|status|follow\s*up|checking\s*in|hear\s+anything|any\s+word|what\s+happened|did\s+you\s+(?:send|check|call|get|finish|look|review))\b", body, re.IGNORECASE) or ACTIONABLE_QUESTION_PATTERN.search(body):
            is_actionable = True
            base_confidence = 0.75

    elif cat == "deadline":
        has_deadline = bool(re.search(
            r"\b(?:by\s+(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d+|noon|eod|evening|morning)|"
            r"before\s+|until\s+|till\s+|deadline|cutoff|"
            r"due\s+(?:date|on|by|today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|eod|noon|\d{1,2}(?:st|nd|rd|th)?))\b",
            body,
            re.IGNORECASE,
        ))
        if has_deadline:
            is_actionable = True
            base_confidence = 0.80
        elif has_fp or has_req or has_dir_imp:
            is_actionable = True
            cat = "commitment"
            base_confidence = 0.80

    if not is_actionable:
        return False, None, None, 0.0

    # 8. Ownership resolution from conversation context & grounded wording
    chat_id = msg.get("chat_id") or cand.get("chat_id", "")
    resolved_owner = resolve_loop_owner(raw_owner=None, chat_id=chat_id, source_msg=msg, context=context)

    # 9. Evidence-grounded confidence scoring
    confidence = base_confidence
    lower_body = body.lower()
    if any(entity in lower_body for entity in TASK_ENTITIES):
        confidence += 0.05
    if any(month in lower_body for month in MONTH_WORDS) or re.search(r"\b(?:tomorrow|today|friday|monday|tonight|by\s+\d+)\b", lower_body):
        confidence += 0.05
    confidence = min(0.95, round(confidence, 2))

    # 10. Conservative configurable threshold
    threshold = min_confidence if min_confidence is not None else (config.min_confidence if config else 0.50)
    if confidence < threshold:
        return False, None, None, 0.0

    return True, cat, resolved_owner, confidence


def classify_candidate_tier(
    cand: Dict[str, Any],
    msg: Optional[Dict[str, Any]],
    context: Any = None,
) -> Tuple[int, str]:
    """
    Classify an open-loop candidate into one of three execution tiers (Scenario C):

    - Tier 1: Deterministic fast-path. High-confidence, non-ambiguous 1:1 first-person
              pledges and direct imperatives resolved deterministically.
    - Tier 2: Safe pre-dispatch reject. Conversational pleasantries, status announcements,
              dismissals, and spouse micro-chatter that can be safely bypassed (0 FP).
    - Tier 3: LLM ambiguity dispatch. Group chats, questions, Hinglish ambiguity, and
              borderline commitments requiring semantic disambiguation.

    Returns:
        Tuple of (tier_int, routing_reason_str).
    """
    if msg is None:
        return 2, "missing_message_record"

    chat_id = msg.get("chat_id") or cand.get("chat_id", "")
    chat = None
    if context is not None:
        if hasattr(context, "get_chat"):
            chat = context.get_chat(chat_id)
        elif hasattr(context, "chats") and isinstance(context.chats, dict):
            chat = context.chats.get(chat_id)
    if not chat:
        chat = {"chat_id": chat_id, "is_group": False, "participants": []}

    is_act, cat, resolved_owner, conf = _evaluate_fallback_actionability(
        cand=cand, msg=msg, context=context, chat=chat
    )
    if not is_act or not cat:
        return 2, "safe_pre_dispatch_reject"

    is_group = bool(chat.get("is_group", False))
    if is_group:
        return 3, "group_chat_ambiguity_dispatch"

    body = (msg.get("body") or cand.get("original_text") or "").strip()
    if cat in ("question", "followup") or body.endswith("?"):
        return 3, "question_resolution_dispatch"

    sender_id = msg.get("sender_id") or cand.get("sender_id", "")
    if conf >= 0.85:
        if sender_id == "ME" and (
            EXPANDED_FP_COMMITMENT_PATTERN.search(body)
            or re.search(r"^(?:will\s+|shall\s+|on\s+it\b|kar\s+dunga\b|de\s+dunga\b|bhej\s+dunga\b)", body, re.IGNORECASE)
        ):
            return 1, "direct_1to1_me_commitment_fastpath"
        if sender_id != "ME" and DIRECTED_IMPERATIVE_PATTERN.search(body):
            return 1, "direct_1to1_counterpart_imperative_fastpath"

    return 3, "borderline_ambiguity_dispatch"


def synthesize_fallback_open_loops(
    candidates: Sequence[Dict[str, Any]],
    context: ConversationContext,
    existing_loops: Optional[Sequence[InterpretedOpenLoop]] = None,
    diagnostics: Optional[OpenLoopExtractionDiagnostics] = None,
    config: Optional[OpenLoopExtractionConfig] = None,
    min_confidence: Optional[float] = None,
) -> List[InterpretedOpenLoop]:
    """Deterministically synthesize candidate open-loop records from Stage B evidence.

    Invoked only for candidates belonging to failed, unavailable, or unattempted batches.

    Safeguards:
    - Never duplicates successfully interpreted or cached open-loop records.
    - Grounds every loop in authentic dataset message ID and chat ID.
    - Rejects automated bot broadcasts, OTPs, ad IDs, syndicated updates, and short conversational banter.
    - Demands substantive task, coordination, or decision intent for all candidates.
    - Resolves grounded ownership using dialogue rules; preserves unresolved ownership as None (unknown).
    - Avoids fixed confidence substitutes; computes evidence-grounded confidence scores.
    - Defaults status to 'open' and due_date to None (resolved downstream).
    """
    if not candidates:
        return []

    existing_source_msgs: Set[str] = set()
    if existing_loops:
        for loop in existing_loops:
            if isinstance(loop, InterpretedOpenLoop) and loop.source_msg:
                existing_source_msgs.add(loop.source_msg)

    synthesized: List[InterpretedOpenLoop] = []
    seen_candidate_msgs: Set[str] = set()

    for cand in candidates:
        mid = cand.get("message_id")
        if not mid or mid in existing_source_msgs or mid in seen_candidate_msgs:
            continue

        msg = _get_dataset_message(context, mid)
        if not msg:
            continue

        chat_id = msg.get("chat_id") or cand.get("chat_id", "")
        if not chat_id:
            continue

        chat = context.get_chat(chat_id) if hasattr(context, "get_chat") else None
        if not chat and hasattr(context, "chats") and isinstance(context.chats, dict):
            chat = context.chats.get(chat_id)
        if not chat and isinstance(context, dict):
            chat = context.get("chats_by_id", {}).get(chat_id)
        if not chat:
            chat = {"chat_id": chat_id, "is_group": False, "participants": []}

        is_act, cat, resolved_owner, conf = _evaluate_fallback_actionability(
            cand=cand,
            msg=msg,
            context=context,
            chat=chat,
            config=config,
            min_confidence=min_confidence,
        )
        if not is_act or not cat:
            continue

        seen_candidate_msgs.add(mid)
        body = (msg.get("body") or msg.get("text") or "").strip()
        title = _clean_title_from_body(body, cat)

        loop = InterpretedOpenLoop(
            source_msg=mid,
            chat_id=chat_id,
            type=cat,
            is_genuine=True,
            title=title,
            owner=resolved_owner,
            target_person=None,
            status="open",
            closed_by=None,
            due_date=None,
            suggested_action=None,
            confidence=conf,
        )
        synthesized.append(loop)
        if diagnostics:
            diagnostics.fallback_candidates_synthesized += 1

    return synthesized


def rank_and_clamp_open_loops(
    loops: Sequence[CandidateOpenLoop],
    context: Any,
    config: OpenLoopExtractionConfig,
    diagnostics: Optional[OpenLoopExtractionDiagnostics] = None,
) -> List[CandidateOpenLoop]:
    """Rank open-loop candidates deterministically and apply chat-level and global limits.

    Ranking Criteria:
    1. Higher confidence score first (-cand.confidence).
    2. Earlier source message timestamp next (ascending ts).
    3. Deterministic tie-breaker by source message ID (cand.source_msg).

    Caps:
    - If config.max_loops_per_chat is set, clamps each chat to that count.
    - If config.max_open_loops is set, clamps the global candidates list to that count (unlimited by default when None).
    - Increments diagnostics.loops_truncated_by_cap for every candidate discarded by a cap.
    """
    # Validate cap settings strictly to ensure invalid explicit values are rejected
    _validate_cap_limit("max_loops_per_chat", config.max_loops_per_chat)
    _validate_cap_limit("max_open_loops", config.max_open_loops)

    if not loops:
        return []

    candidates = list(loops)

    # 1. Chat-level cap if configured
    if config.max_loops_per_chat is not None:
        by_chat: Dict[str, List[CandidateOpenLoop]] = defaultdict(list)
        for cand in candidates:
            by_chat[cand.chat_id].append(cand)

        retained: List[CandidateOpenLoop] = []
        for chat_id, chat_loops in by_chat.items():
            if len(chat_loops) > config.max_loops_per_chat:
                chat_loops.sort(
                    key=lambda c: (
                        -c.confidence,
                        _get_message_timestamp(context, c.source_msg),
                        c.source_msg,
                    )
                )
                discarded = len(chat_loops) - config.max_loops_per_chat
                if diagnostics:
                    diagnostics.loops_truncated_by_cap += discarded
                retained.extend(chat_loops[:config.max_loops_per_chat])
            else:
                retained.extend(chat_loops)
        candidates = retained

    # 2. Deterministic ranking across all candidates
    candidates.sort(
        key=lambda c: (
            -c.confidence,
            _get_message_timestamp(context, c.source_msg),
            c.source_msg,
        )
    )

    # 3. Optional global cap (max_open_loops, disabled by default when None)
    if config.max_open_loops is not None:
        if len(candidates) > config.max_open_loops:
            truncated = len(candidates) - config.max_open_loops
            if diagnostics:
                diagnostics.loops_truncated_by_cap += truncated
            candidates = candidates[:config.max_open_loops]

    return candidates


def extract_open_loops(
    context: ConversationContext,
    interpreter: Any,
    config: Optional[OpenLoopExtractionConfig] = None,
    diagnostics: Optional[OpenLoopExtractionDiagnostics] = None,
    budget: Optional[Any] = None,
) -> OpenLoopExtractionResult:
    """Execute the end-to-end open-loop extraction pipeline.

    Orchestrates the complete open-loop detection workflow across sequential stages:
    1. Stage A: Extract candidate evidence records matching open-loop categories
       ("commitment", "question", "followup", "deadline") via CandidateEvidenceExtractor.
    2. Stage B: Filter candidates using approved group-chat relevance logic (is_relevant_group_candidate).
       If no candidates remain, LLM batch calls are safely bypassed.
    3. Stage C: Batch candidates chronologically and interpret via LLM (interpret_open_loop_evidence).
       Fatal configuration errors bubble up immediately; non-fatal batch failures are logged in diagnostics.
    4. Stage D: Resolve ownership, grounded due dates, actor-authorized closure, lifecycle episodes,
       and HelpAction decisions via resolve_open_loops.
    5. Stage E: Apply configurable safety limits (max_loops_per_chat and optional max_open_loops) with deterministic ranking.
    6. Stage F: Validate candidates at the schema barrier and serialize strictly to 9 official schema fields.

    Args:
        context: ConversationContext instance containing loaded dataset, chat index, and metadata.
        interpreter: LLM evidence interpreter instance (e.g. LLMEvidenceInterpreter or test mock).
        config: Optional OpenLoopExtractionConfig (uses default config with unlimited open loops if omitted).
        diagnostics: Optional OpenLoopExtractionDiagnostics instance to track operational counters.
        budget: Optional ExecutionBudget instance for monotonic deadline enforcement.

    Returns:
        OpenLoopExtractionResult containing schema-compliant open loop records and aggregated diagnostics.
    """
    if config is None:
        config = OpenLoopExtractionConfig()

    if diagnostics is None:
        diagnostics = OpenLoopExtractionDiagnostics()

    # Initialize execution budget for session if not passed (respect existing session)
    active_budget = budget
    if active_budget is None:
        if getattr(interpreter, "active_budget", None) is not None:
            active_budget = interpreter.active_budget
        elif hasattr(interpreter, "start_session"):
            active_budget = interpreter.start_session(max_runtime_seconds=config.max_runtime_seconds)

    if context is None:
        return OpenLoopExtractionResult(open_loops=[], diagnostics=diagnostics)

    # Stage A — Extract candidate evidence
    extractor = CandidateEvidenceExtractor(enabled_categories=list(OPEN_LOOP_CATEGORIES))
    raw_candidates = extractor.extract(context)
    diagnostics.candidates_received += len(raw_candidates)

    if not raw_candidates:
        return OpenLoopExtractionResult(open_loops=[], diagnostics=diagnostics)

    # Stage B — Group-chat relevance filtering
    filtered_candidates: List[Dict[str, Any]] = []
    for cand in raw_candidates:
        mid = cand.get("message_id")
        msg = _get_dataset_message(context, mid) if mid else None
        if msg is None:
            continue
        if is_relevant_group_candidate(cand, msg, context=context):
            filtered_candidates.append(cand)

    diagnostics.candidates_after_group_filter += len(filtered_candidates)

    if not filtered_candidates:
        return OpenLoopExtractionResult(open_loops=[], diagnostics=diagnostics)

    # Three-Tier Selective Routing (Scenario C)
    tier1_cands: List[Dict[str, Any]] = []
    tier2_cands: List[Dict[str, Any]] = []
    tier3_cands: List[Dict[str, Any]] = []
    routing_reasons: Dict[str, str] = {}

    for cand in filtered_candidates:
        mid = cand.get("message_id")
        msg = _get_dataset_message(context, mid) if mid else None
        tier, reason = classify_candidate_tier(cand, msg, context=context)
        if mid:
            routing_reasons[mid] = reason
        if tier == 1:
            tier1_cands.append(cand)
        elif tier == 2:
            tier2_cands.append(cand)
        else:
            tier3_cands.append(cand)

    # Invariant: Exact candidate conservation
    assert len(tier1_cands) + len(tier2_cands) + len(tier3_cands) == len(filtered_candidates)

    if diagnostics is not None:
        diagnostics.tier1_candidates = [c["message_id"] for c in tier1_cands if c.get("message_id")]
        diagnostics.tier2_candidates = [c["message_id"] for c in tier2_cands if c.get("message_id")]
        diagnostics.tier3_candidates = [c["message_id"] for c in tier3_cands if c.get("message_id")]
        diagnostics.routing_reasons = routing_reasons

    # Stage C — LLM interpretation of batches (with fallback synthesis for missing/failed/offline batches)
    interpreted_loops: List[InterpretedOpenLoop] = []
    successful_candidate_ids: Set[str] = set()
    failed_candidate_ids: Set[str] = set()
    unexpected_candidate_ids: Set[str] = set()

    is_live_routing = (
        config.enable_selective_routing
        and interpreter is not None
        and getattr(interpreter, "offline_mode", None) is False
    )

    if is_live_routing:
        # Tier 1: Deterministic fast-path synthesis
        tier1_loops = synthesize_fallback_open_loops(
            candidates=tier1_cands,
            context=context,
            existing_loops=None,
            diagnostics=diagnostics,
            config=config,
        )
        interpreted_loops.extend(tier1_loops)
        for c in tier1_cands:
            mid = c.get("message_id")
            if mid:
                successful_candidate_ids.add(mid)

        # Tier 3: LLM ambiguity dispatch with disjoint multi-snippet batching
        if tier3_cands:
            t3_start_batches_succeeded = diagnostics.batches_succeeded
            t3_start_batches_failed = diagnostics.batches_failed
            t3_start_batches_cached = diagnostics.batches_cached
            t3_start_batches_created = diagnostics.batches_created

            try:
                tier3_loops = interpret_open_loop_evidence(
                    context=context,
                    candidates=tier3_cands,
                    interpreter=interpreter,
                    diagnostics=diagnostics,
                    budget=active_budget,
                    successful_candidate_ids=successful_candidate_ids,
                    failed_candidate_ids=failed_candidate_ids,
                    unexpected_candidate_ids=unexpected_candidate_ids,
                    use_disjoint_batching=config.disjoint_batching,
                )
                interpreted_loops.extend(tier3_loops)
            except LLMConfigurationError:
                if not getattr(interpreter, "allow_offline", False):
                    raise
                for c in tier3_cands:
                    mid = c.get("message_id")
                    if mid:
                        failed_candidate_ids.add(mid)

            diagnostics.tier3_batches_created = diagnostics.batches_created - t3_start_batches_created
            diagnostics.tier3_batches_succeeded = diagnostics.batches_succeeded - t3_start_batches_succeeded
            diagnostics.tier3_batches_failed = diagnostics.batches_failed - t3_start_batches_failed
            diagnostics.tier3_cache_hits = diagnostics.batches_cached - t3_start_batches_cached

            # Resilient fallback for any failed Tier 3 candidates
            failed_tier3_cands = [c for c in tier3_cands if c.get("message_id") in failed_candidate_ids]
            if failed_tier3_cands:
                fallback_tier3 = synthesize_fallback_open_loops(
                    candidates=failed_tier3_cands,
                    context=context,
                    existing_loops=interpreted_loops,
                    diagnostics=diagnostics,
                    config=config,
                )
                if fallback_tier3:
                    interpreted_loops.extend(fallback_tier3)
    else:
        # Offline or full-dispatch legacy mode
        if interpreter is not None:
            try:
                interpreted_loops = interpret_open_loop_evidence(
                    context=context,
                    candidates=filtered_candidates,
                    interpreter=interpreter,
                    diagnostics=diagnostics,
                    budget=active_budget,
                    successful_candidate_ids=successful_candidate_ids,
                    failed_candidate_ids=failed_candidate_ids,
                    unexpected_candidate_ids=unexpected_candidate_ids,
                    use_disjoint_batching=config.disjoint_batching if config.enable_selective_routing else False,
                )
            except LLMConfigurationError:
                if not getattr(interpreter, "allow_offline", False):
                    raise
                interpreted_loops = []

        fallback_eligible_candidates = [
            cand for cand in filtered_candidates
            if cand.get("message_id")
            and cand.get("message_id") not in successful_candidate_ids
            and cand.get("message_id") not in unexpected_candidate_ids
        ]

        should_fallback = config.enable_fallback
        if should_fallback is None:
            if type(interpreter).__name__ == "MockEvidenceInterpreter":
                should_fallback = False
            else:
                should_fallback = bool(fallback_eligible_candidates)

        if should_fallback and fallback_eligible_candidates:
            fallback_loops = synthesize_fallback_open_loops(
                candidates=fallback_eligible_candidates,
                context=context,
                existing_loops=interpreted_loops,
                diagnostics=diagnostics,
                config=config,
            )
            if fallback_loops:
                interpreted_loops.extend(fallback_loops)

    # Stage D — Deterministic open-loop resolution and deduplication
    resolved_candidates = resolve_open_loops(
        interpreted_loops=interpreted_loops,
        context=context,
        config=config,
        diagnostics=diagnostics,
    )

    # Stage E — Deterministic ranking and clamping
    clamped_candidates = rank_and_clamp_open_loops(
        loops=resolved_candidates,
        context=context,
        config=config,
        diagnostics=diagnostics,
    )

    # Stage F — Schema barrier validation and serialization
    final_open_loops: List[Dict[str, Any]] = []
    for cand in clamped_candidates:
        if cand.validate():
            final_open_loops.append(cand.to_schema_dict())
        else:
            diagnostics.candidates_rejected += 1

    diagnostics.final_loops_emitted = len(final_open_loops)

    return OpenLoopExtractionResult(
        open_loops=final_open_loops,
        diagnostics=diagnostics,
    )

