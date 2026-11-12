"""LLM Evidence Interpreter module orchestrating conversation evidence interpretation."""

from dataclasses import dataclass, field
import email.utils
import math
import os
import random
import re
import socket
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import urllib.error
import urllib.parse

from llm import chat as default_llm_chat
from src.interpretation.batcher import EvidenceBatch, build_evidence_batches
from src.interpretation.cache import ExecutionBudget, InterpretationCache, _validate_numeric
from src.interpretation.parser import (
    ChatParserDiagnostics,
    InterpretationResponseError,
    ResponseParsingError,
    ResponseStructureError,
    parse_chat_interpretation_response,
    parse_participant_interpretation_response,
)
from src.interpretation.prompts import (
    build_chat_interpretation_prompt,
    build_participant_interpretation_prompt,
)
from src.interpretation.records import (
    InterpretedGoal,
    InterpretedOpenLoop,
    InterpretedOrganisation,
    InterpretedPerson,
)
from src.preprocessing.context_builder import ConversationContext


class LLMConfigurationError(RuntimeError):
    """Raised when LLM client or environment configuration is invalid or unauthorized."""
    pass


@dataclass
class BatchFailureDiagnostic:
    """Diagnostic details for an exhausted batch interpretation failure."""
    phase: str                 # "chat" | "participant"
    batch_id: str              # chat_id (e.g. "C04") or comma-separated person IDs (e.g. "P03,P04")
    attempt_count: int         # Total attempts made before exhaustion
    error_type: str            # Exception class or classification (e.g. "HTTPError:429", "ResponseParsingError")
    error_message: str         # Allowlisted, sanitized diagnostic message
    timestamp: float           # Unix epoch timestamp of failure


@dataclass
class InterpretationDiagnostics:
    """Summary and failure records for an interpretation run."""
    chat_batches_total: int = 0
    chat_batches_succeeded: int = 0
    chat_batches_failed: int = 0
    participant_batches_total: int = 0
    participant_batches_succeeded: int = 0
    participant_batches_failed: int = 0
    batches_cached: int = 0
    batches_unattempted: int = 0
    failures: List[BatchFailureDiagnostic] = field(default_factory=list)

    # Observability & diagnostic aggregate counts
    chat_batches_created: int = 0
    cache_hits: int = 0
    live_llm_calls: int = 0
    offline_cache_misses: int = 0
    successful_parsed_batches: int = 0
    failed_batches: int = 0
    raw_goals_returned: int = 0
    goals_accepted: int = 0
    raw_open_loops_returned: int = 0
    open_loops_accepted: int = 0

    @property
    def has_failures(self) -> bool:
        """True if any batch failed during interpretation."""
        return len(self.failures) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic summary to a serializable dictionary without sensitive data."""
        return {
            "chat_batches_created": self.chat_batches_created,
            "cache_hits": self.cache_hits,
            "live_llm_calls": self.live_llm_calls,
            "offline_cache_misses": self.offline_cache_misses,
            "successful_parsed_batches": self.successful_parsed_batches,
            "failed_batches": self.failed_batches,
            "raw_goals_returned": self.raw_goals_returned,
            "goals_accepted": self.goals_accepted,
            "raw_open_loops_returned": self.raw_open_loops_returned,
            "open_loops_accepted": self.open_loops_accepted,
            "chat_batches_total": self.chat_batches_total,
            "chat_batches_succeeded": self.chat_batches_succeeded,
            "chat_batches_failed": self.chat_batches_failed,
            "participant_batches_total": self.participant_batches_total,
            "participant_batches_succeeded": self.participant_batches_succeeded,
            "participant_batches_failed": self.participant_batches_failed,
            "batches_cached": self.batches_cached,
            "batches_unattempted": self.batches_unattempted,
            "failures_count": len(self.failures),
        }


@dataclass
class InterpretationResult:
    """Strongly typed output of the interpretation layer with isolated diagnostics."""
    open_loops: List[InterpretedOpenLoop]
    goals: List[InterpretedGoal]
    people: List[InterpretedPerson]
    organisations: List[InterpretedOrganisation]
    diagnostics: InterpretationDiagnostics = field(default_factory=InterpretationDiagnostics)


URL_PATTERN = re.compile(r"https?://[^\s<>\"'`]+")
SENSITIVE_PARAM_NAMES = frozenset({
    "token",
    "access_token",
    "api_token",
    "auth_token",
    "api_key",
    "apikey",
    "password",
    "secret",
    "key",
})
STANDALONE_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|api[_-]?token|auth[_-]?token|access[_-]?token|token|password|secret|authorization)\b\s*[:=]\s*(?:'[^']*'|\"[^\"]*\"|[^\s,;'\"&]+)"
)


def _has_sensitive_query(url_str: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url_str)
        if not parsed.query:
            return False
        params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        for k, _ in params:
            clean_k = k.lower().replace("-", "_")
            if clean_k in SENSITIVE_PARAM_NAMES:
                return True
        return False
    except Exception:
        return bool(re.search(r"(?i)[?&](?:api[_-]?key|api[_-]?token|auth[_-]?token|access[_-]?token|token|key|password|secret)=", url_str))


def _redact_sensitive_urls(text: str) -> str:
    def _replacer(m: re.Match) -> str:
        raw_url = m.group(0)
        trailing = ""
        while raw_url and raw_url[-1] in ".,;:!?)":
            trailing = raw_url[-1] + trailing
            raw_url = raw_url[:-1]
        if _has_sensitive_query(raw_url):
            return "[REDACTED_URL]" + trailing
        return m.group(0)

    return URL_PATTERN.sub(_replacer, text)


def _sanitize_error_message(err: Optional[Union[Exception, str]]) -> str:
    """
    Produce an allowlisted, sanitized error summary that never leaks API keys,
    passwords, authorization tokens, credential-bearing URLs, or raw sensitive exception strings.
    """
    if err is None:
        return "Unknown error"

    if isinstance(err, str):
        # 1. Bearer tokens
        sanitized = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", err)
        # 2. Standalone key/token/password assignments (including quoted values with spaces)
        sanitized = STANDALONE_CREDENTIAL_PATTERN.sub(r"\1=[REDACTED]", sanitized)
        # 3. OpenAI-style sk- tokens
        sanitized = re.sub(r"\bsk-[a-zA-Z0-9_\-]{8,}\b", "[REDACTED]", sanitized)
        # 4. User-info credentials in URLs (e.g. username:password@host or https://user:pass@host/path)
        sanitized = re.sub(
            r"(?i)\b([a-z][a-z0-9+.-]*://)?([^/\s@:]+):([^/\s@:]+)@([^\s]+)",
            lambda m: f"{m.group(1) or ''}[REDACTED]:[REDACTED]@{m.group(4)}",
            sanitized,
        )
        # 5. Sensitive query parameters in URLs (entire URL redacted, including remaining params and fragments)
        sanitized = _redact_sensitive_urls(sanitized)
        sanitized = sanitized.split("\n")[0].strip()
        return sanitized or "Unknown error"

    # 1. HTTPError: generic message with status code only (no reason, msg, url, or headers)
    if isinstance(err, urllib.error.HTTPError):
        return f"HTTP error {err.code}"

    # 2. URLError: fixed message only (no reason or url)
    if isinstance(err, urllib.error.URLError):
        return "Network connection error"

    # 3. TimeoutError / socket.timeout: generic timeout message
    if isinstance(err, (TimeoutError, socket.timeout)):
        return "Request timed out"

    # 4. Parsing and structure errors: generic category message
    if isinstance(err, ResponseParsingError):
        return "Model output parsing error"
    if isinstance(err, ResponseStructureError):
        return "Model output structure error"
    if isinstance(err, InterpretationResponseError):
        return "Interpretation response error"

    # 5. Unexpected exceptions: expose only the exception class name, never the raw message
    return type(err).__name__


def _extract_retry_delay(err: Optional[Exception]) -> Optional[float]:
    """Extract stated retry-after delay floor from HTTP headers or response body.

    Collects all valid applicable provider reset delays and returns the maximum required delay:
    - Retry-After: integer or float seconds (e.g. '15', '2.5')
    - Retry-After: RFC 7231 / RFC 2822 HTTP date string evaluated against wall-clock time
    - retry-after-ms / Retry-After-Ms: milliseconds integer or float
    - x-ratelimit-reset / X-RateLimit-Reset: relative seconds or epoch timestamp
    - x-ratelimit-reset-requests / X-RateLimit-Reset-Requests: seconds duration
    - x-ratelimit-reset-tokens / X-RateLimit-Reset-Tokens: seconds duration
    - Response body fallback: regex extraction from error message if no valid headers
    """
    if not isinstance(err, urllib.error.HTTPError):
        return None

    delays: List[float] = []
    headers = getattr(err, "headers", None)

    if headers:
        # 1. Retry-After header
        val = headers.get("Retry-After") or headers.get("retry-after")
        if val is not None:
            val_str = str(val).strip()
            parsed_sec: Optional[float] = None
            try:
                sec = float(val_str)
                if math.isfinite(sec) and sec >= 0:
                    parsed_sec = sec
            except (ValueError, TypeError):
                try:
                    dt = email.utils.parsedate_to_datetime(val_str)
                    if dt:
                        diff = dt.timestamp() - time.time()
                        parsed_sec = max(0.0, diff)
                except Exception:
                    pass
            if parsed_sec is not None:
                delays.append(parsed_sec)

        # 2. retry-after-ms header
        val_ms = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")
        if val_ms is not None:
            try:
                ms = float(str(val_ms).strip())
                if math.isfinite(ms) and ms >= 0:
                    delays.append(ms / 1000.0)
            except (ValueError, TypeError):
                pass

        # 3. x-ratelimit-reset-requests
        val_req = headers.get("x-ratelimit-reset-requests") or headers.get("X-RateLimit-Reset-Requests")
        if val_req is not None:
            try:
                sec = float(str(val_req).strip().rstrip("s"))
                if math.isfinite(sec) and sec >= 0:
                    delays.append(sec)
            except (ValueError, TypeError):
                pass

        # 4. x-ratelimit-reset-tokens
        val_tok = headers.get("x-ratelimit-reset-tokens") or headers.get("X-RateLimit-Reset-Tokens")
        if val_tok is not None:
            try:
                sec = float(str(val_tok).strip().rstrip("s"))
                if math.isfinite(sec) and sec >= 0:
                    delays.append(sec)
            except (ValueError, TypeError):
                pass

        # 5. x-ratelimit-reset
        val_reset = headers.get("x-ratelimit-reset") or headers.get("X-RateLimit-Reset")
        if val_reset is not None:
            try:
                sec = float(str(val_reset).strip().rstrip("s"))
                if math.isfinite(sec):
                    if sec > 1e8:  # epoch timestamp
                        delays.append(max(0.0, sec - time.time()))
                    elif sec >= 0:
                        delays.append(sec)
            except (ValueError, TypeError):
                pass

    if delays:
        return max(delays)

    # 6. Fallback: inspect response body if no valid header delay was found
    try:
        fp = getattr(err, "fp", None)
        if fp:
            body = fp.read(4096)
            if body:
                text = body.decode("utf-8", errors="replace")
                m = re.search(r"try again in ([\d\.]+)s", text, re.IGNORECASE)
                if m:
                    sec = float(m.group(1))
                    if math.isfinite(sec) and sec >= 0:
                        return sec
    except Exception:
        pass

    return None


def build_participant_dossiers(
    context: ConversationContext,
    max_participants_per_batch: int = 8,
    messages_per_dossier: int = 8,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Construct participant dossiers and consolidated chat metadata from ConversationContext.

    Args:
        context: ConversationContext instance from Step 3.3.
        max_participants_per_batch: Maximum participant dossiers per batch (default 8).
        messages_per_dossier: Maximum contextual messages per participant (default 8).

    Returns:
        List of (dossier_batch, chat_metadata_for_batch) tuples in deterministic order.

    Raises:
        TypeError: If context is not a ConversationContext instance.
        ValueError: If batch limits are invalid.
    """
    if not isinstance(context, ConversationContext):
        raise TypeError(f"Expected ConversationContext, got {type(context).__name__}")
    if max_participants_per_batch <= 0:
        raise ValueError(f"max_participants_per_batch must be > 0, got {max_participants_per_batch}")
    if messages_per_dossier <= 0:
        raise ValueError(f"messages_per_dossier must be > 0, got {messages_per_dossier}")

    # 1. Filter and sort participant IDs deterministically, strictly excluding 'ME'
    all_pids = sorted(pid for pid in context.participants.keys() if pid != "ME")

    dossiers: List[Dict[str, Any]] = []

    for pid in all_pids:
        p_data = context.get_participant(pid)
        if not p_data:
            continue

        # Construct names from sender_names and direct_chat_names
        seen_names: Set[str] = set()
        names: List[str] = []
        for n in (p_data.get("sender_names", []) + p_data.get("direct_chat_names", [])):
            if isinstance(n, str) and n.strip() and n.strip() not in seen_names:
                seen_names.add(n.strip())
                names.append(n.strip())

        # Obtain associated chat metadata from context.chats
        chats: List[Dict[str, Any]] = []
        for cid in p_data.get("chat_ids", []):
            chat_info = context.get_chat(cid)
            if chat_info:
                chats.append({
                    "chat_id": cid,
                    "name": chat_info.get("name", ""),
                    "is_group": chat_info.get("is_group", False),
                })

        # Representative messages: use context.get_participant_messages()
        # mode="all" retrieves messages in chats where participant is a member,
        # preserving essential context even for silent participants
        raw_msgs = context.get_participant_messages(
            participant_id=pid,
            mode="all",
            limit=messages_per_dossier,
        )

        clean_msgs: List[Dict[str, Any]] = []
        for m in raw_msgs:
            if isinstance(m, dict):
                clean_msgs.append({
                    "id": m.get("id", ""),
                    "chat_id": m.get("chat_id", ""),
                    "sender_id": m.get("sender_id", ""),
                    "sender_name": m.get("sender_name", ""),
                    "ts": m.get("ts", ""),
                    "body": m.get("body", "") if m.get("body") is not None else "",
                })

        dossiers.append({
            "person_id": pid,
            "names": names,
            "chats": chats,
            "messages": clean_msgs,
        })

    # 2. Chunk dossiers into batches with consolidated chat_metadata
    batches: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    for i in range(0, len(dossiers), max_participants_per_batch):
        chunk = dossiers[i : i + max_participants_per_batch]
        chat_metadata: Dict[str, Any] = {}
        for d in chunk:
            for ch in d.get("chats", []):
                cid = ch.get("chat_id")
                if cid and cid not in chat_metadata:
                    chat_info = context.get_chat(cid)
                    if chat_info:
                        chat_metadata[cid] = {
                            "name": chat_info.get("name", ""),
                            "is_group": chat_info.get("is_group", False),
                            "participants": list(chat_info.get("participants", [])),
                        }
        batches.append((chunk, chat_metadata))

    return batches


def _parse_duration(val: Any) -> Optional[float]:
    """Parse duration string (e.g. '2s', '500ms', '1.5', '1m') or float seconds."""
    if val is None:
        return None
    val_str = str(val).strip().lower()
    try:
        if val_str.endswith("ms"):
            return float(val_str[:-2]) / 1000.0
        elif val_str.endswith("s"):
            return float(val_str[:-1])
        elif val_str.endswith("m"):
            return float(val_str[:-1]) * 60.0
        sec = float(val_str)
        if math.isfinite(sec) and sec >= 0:
            if sec > 1e8:  # epoch timestamp
                return max(0.0, sec - time.time())
            return sec
    except (ValueError, TypeError):
        pass
    return None


def _parse_int_header(val: Any) -> Optional[int]:
    """Parse non-negative integer header value."""
    if val is None:
        return None
    try:
        val_clean = str(val).strip().rstrip("s")
        i = int(float(val_clean))
        return i if i >= 0 else None
    except (ValueError, TypeError):
        return None


def estimate_prompt_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    """
    Estimate prompt token count from messages using character heuristics:
    ~4 characters per token + ~4 tokens framing overhead per message.
    """
    total_tokens = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        total_tokens += 4  # framing overhead per message
        content = m.get("content", "")
        if isinstance(content, str):
            total_tokens += math.ceil(len(content) / 4.0)
    total_tokens += 2  # priming tokens
    return max(1, total_tokens)


class ProactiveRateLimiter:
    """
    Proactive rate limiter enforcing RPM (Requests Per Minute) and TPM (Estimated
    Prompt Tokens Per Minute) across a rolling 60-second window. Monotonic-clock
    based with support for provider response header overrides.
    """

    def __init__(
        self,
        max_rpm: int = 30,
        max_tpm: int = 6000,
        window_seconds: float = 60.0,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        if max_rpm <= 0:
            raise ValueError(f"max_rpm must be strictly positive, got {max_rpm}")
        if max_tpm <= 0:
            raise ValueError(f"max_tpm must be strictly positive, got {max_tpm}")
        self.max_rpm = int(max_rpm)
        self.max_tpm = int(max_tpm)
        self.window_seconds = float(window_seconds)
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep

        # Rolling history: list of (monotonic_timestamp, token_count)
        self._history: List[Tuple[float, int]] = []

        # Provider overrides
        self.provider_remaining_requests: Optional[int] = None
        self.provider_requests_reset_at: Optional[float] = None
        self.provider_remaining_tokens: Optional[int] = None
        self.provider_tokens_reset_at: Optional[float] = None

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._history = [entry for entry in self._history if entry[0] > cutoff]
        if self.provider_requests_reset_at is not None and now >= self.provider_requests_reset_at:
            self.provider_remaining_requests = None
            self.provider_requests_reset_at = None
        if self.provider_tokens_reset_at is not None and now >= self.provider_tokens_reset_at:
            self.provider_remaining_tokens = None
            self.provider_tokens_reset_at = None

    def calculate_wait_time(self, estimated_tokens: int) -> Optional[float]:
        """
        Calculate required wait time in seconds before dispatching.
        Returns:
            float >= 0.0: seconds to wait before dispatch.
            None: if request can never be satisfied (e.g. estimated_tokens > max_tpm).
        """
        if estimated_tokens > self.max_tpm:
            return None

        now = self.clock()
        self._prune(now)

        wait_rpm = 0.0
        wait_tpm = 0.0

        # 1. Provider header constraints take precedence if active
        if self.provider_requests_reset_at is not None and now < self.provider_requests_reset_at and self.provider_remaining_requests is not None:
            if self.provider_remaining_requests <= 0:
                wait_rpm = max(0.0, self.provider_requests_reset_at - now)
        else:
            # Local rolling window RPM
            if len(self._history) >= self.max_rpm:
                idx = len(self._history) - self.max_rpm
                oldest_ts = self._history[idx][0]
                wait_rpm = max(0.0, (oldest_ts + self.window_seconds) - now)

        if self.provider_tokens_reset_at is not None and now < self.provider_tokens_reset_at and self.provider_remaining_tokens is not None:
            if self.provider_remaining_tokens < estimated_tokens:
                wait_tpm = max(0.0, self.provider_tokens_reset_at - now)
        else:
            # Local rolling window TPM
            current_tokens = sum(t for _, t in self._history)
            if current_tokens + estimated_tokens > self.max_tpm:
                needed_freed = current_tokens + estimated_tokens - self.max_tpm
                freed = 0
                for ts, tok in self._history:
                    freed += tok
                    if freed >= needed_freed:
                        wait_tpm = max(0.0, (ts + self.window_seconds) - now)
                        break

        total_wait = max(0.0, wait_rpm, wait_tpm)
        return total_wait

    def acquire(
        self,
        estimated_tokens: int,
        budget: Optional[ExecutionBudget] = None,
    ) -> bool:
        """
        Attempt to acquire capacity for a request with estimated_tokens.
        Waits if needed, provided the execution budget permits.
        Returns True if capacity acquired and request may proceed; False otherwise.
        """
        wait_time = self.calculate_wait_time(estimated_tokens)
        if wait_time is None:
            return False

        if wait_time > 0.0:
            if budget is not None:
                if not budget.can_attempt_call(additional_delay=wait_time):
                    return False
                max_pacing = budget.max_pacing_allowed()
                if wait_time > max_pacing:
                    return False
            self.sleeper(wait_time)
            if budget is not None:
                if budget.calculate_effective_timeout() < budget.min_call_timeout:
                    return False

        # Record this request
        now = self.clock()
        self._prune(now)
        self._history.append((now, estimated_tokens))

        if self.provider_remaining_requests is not None and self.provider_remaining_requests > 0:
            self.provider_remaining_requests -= 1
        if self.provider_remaining_tokens is not None and self.provider_remaining_tokens >= estimated_tokens:
            self.provider_remaining_tokens -= estimated_tokens

        return True

    def update_from_headers(self, headers: Optional[Dict[str, Any]]) -> None:
        """Update provider constraints from response headers (sanitized, no secrets)."""
        if not headers:
            return
        now = self.clock()
        h_lower = {str(k).lower(): v for k, v in headers.items()}

        rem_req = h_lower.get("x-ratelimit-remaining-requests")
        if rem_req is not None:
            val = _parse_int_header(rem_req)
            if val is not None:
                self.provider_remaining_requests = val

        reset_req = h_lower.get("x-ratelimit-reset-requests") or h_lower.get("retry-after")
        if reset_req is not None:
            sec = _parse_duration(reset_req)
            if sec is not None:
                self.provider_requests_reset_at = now + sec

        rem_tok = h_lower.get("x-ratelimit-remaining-tokens")
        if rem_tok is not None:
            val = _parse_int_header(rem_tok)
            if val is not None:
                self.provider_remaining_tokens = val

        reset_tok = h_lower.get("x-ratelimit-reset-tokens")
        if reset_tok is not None:
            sec = _parse_duration(reset_tok)
            if sec is not None:
                self.provider_tokens_reset_at = now + sec


class LLMEvidenceInterpreter:
    """
    Orchestrates the interpretation layer by batching conversation evidence,
    querying the LLM with bounded retries and exponential backoff, parsing typed
    intermediate records, and isolating failures with sanitized diagnostics.
    """

    def __init__(
        self,
        chat_fn: Optional[Callable[..., str]] = None,
        model: Optional[str] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0,
        inter_batch_pacing: float = 1.0,
        min_pacing: float = 0.5,
        max_pacing: float = 5.0,
        timeout: Union[int, float] = 120,
        min_call_timeout: float = 1.0,
        max_runtime_seconds: float = 1500.0,
        reserve_seconds: float = 45.0,
        cache_enabled: Optional[bool] = None,
        cache_path: str = ".cache/llm_cache.json",
        max_participants_per_batch: int = 8,
        messages_per_dossier: int = 8,
        allow_offline: bool = False,
        offline_mode: bool = False,
        max_rpm: Optional[int] = None,
        max_tpm: Optional[int] = None,
        rate_limiter: Optional[Any] = None,
    ) -> None:
        """
        Initialize the interpreter.

        Args:
            chat_fn: LLM chat client function. If None, uses llm.chat and validates
                     environment variables (LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL if model is None).
            model: Optional model identifier.
            max_retries: Number of retries after initial attempt (default 2, meaning 3 attempts total).
            retry_delay: Initial backoff delay in seconds (default 1.0).
            backoff_factor: Multiplier for exponential backoff (default 2.0).
            max_backoff: Maximum backoff delay floor in seconds (default 60.0).
            inter_batch_pacing: Initial adaptive pacing delay between network calls (default 1.0s).
            min_pacing: Minimum pacing delay floor (default 0.5s).
            max_pacing: Maximum pacing delay ceiling (default 5.0s).
            timeout: Configured socket timeout in seconds for each LLM call (default 120).
            min_call_timeout: Minimum viable socket timeout floor (default 1.0s).
            max_runtime_seconds: Open-loop extraction phase budget in seconds (default 1500.0s / 25 min).
            reserve_seconds: Monotonic headroom reserved for deduplication and disk I/O (default 45.0s).
            cache_enabled: Whether checkpoint caching is enabled (default True for live client, False for custom/mock client unless explicitly set).
            cache_path: File path for local cache storage (default '.cache/llm_cache.json').
            max_participants_per_batch: Number of participant dossiers per prompt batch (default 8).
            messages_per_dossier: Number of representative messages per dossier (default 8).
        """
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if retry_delay < 0:
            raise ValueError(f"retry_delay must be >= 0, got {retry_delay}")
        if backoff_factor < 1.0:
            raise ValueError(f"backoff_factor must be >= 1.0, got {backoff_factor}")

        self.chat_fn = chat_fn or default_llm_chat
        self.is_custom_client = chat_fn is not None
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.current_pacing = inter_batch_pacing
        self.min_pacing = min_pacing
        self.max_pacing = max_pacing

        # Validate runtime and timeout numeric configuration strictly
        self.timeout = _validate_numeric("timeout", timeout, strictly_positive=True)
        self.min_call_timeout = _validate_numeric("min_call_timeout", min_call_timeout, strictly_positive=True)
        self.max_runtime_seconds = _validate_numeric("max_runtime_seconds", max_runtime_seconds, strictly_positive=True)
        self.reserve_seconds = _validate_numeric("reserve_seconds", reserve_seconds, non_negative=True)
        if self.reserve_seconds >= self.max_runtime_seconds:
            raise ValueError(
                f"reserve_seconds ({self.reserve_seconds}) must be strictly smaller than max_runtime_seconds ({self.max_runtime_seconds})"
            )

        if cache_enabled is None:
            self.cache_enabled = not self.is_custom_client
        else:
            self.cache_enabled = cache_enabled
        self.cache = InterpretationCache(cache_path=cache_path, enabled=self.cache_enabled)
        self.active_budget: Optional[ExecutionBudget] = None
        self._last_api_call_monotonic: Optional[float] = None
        self._consecutive_successes: int = 0
        self.last_call_cached: bool = False
        self._live_call_count: int = 0
        self.diagnostics: InterpretationDiagnostics = InterpretationDiagnostics()
        self.max_participants_per_batch = max_participants_per_batch
        self.messages_per_dossier = messages_per_dossier

        if max_rpm is not None:
            self.max_rpm = int(max_rpm)
        else:
            try:
                self.max_rpm = int(os.environ.get("LLM_MAX_RPM", "30"))
            except (ValueError, TypeError):
                self.max_rpm = 30

        if max_tpm is not None:
            self.max_tpm = int(max_tpm)
        else:
            try:
                self.max_tpm = int(os.environ.get("LLM_MAX_TPM", "6000"))
            except (ValueError, TypeError):
                self.max_tpm = 6000

        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        else:
            self.rate_limiter = ProactiveRateLimiter(max_rpm=self.max_rpm, max_tpm=self.max_tpm)

        env_offline = (
            os.environ.get("OFFLINE_MODE", "").strip().lower() in ("1", "true", "yes")
            or os.environ.get("LLM_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        )
        self.offline_mode = bool(offline_mode or env_offline)
        self.allow_offline = bool(allow_offline or self.offline_mode)

        # When using default llm.chat, validate environment variables unless in offline mode
        if not self.is_custom_client and not self.offline_mode:
            try:
                self._validate_environment()
            except LLMConfigurationError:
                if self.allow_offline:
                    self.offline_mode = True
                else:
                    raise

    def start_session(self, max_runtime_seconds: Optional[float] = None) -> ExecutionBudget:
        """Initialize or reset a single shared monotonic execution budget for an extraction run."""
        if max_runtime_seconds is not None:
            runtime = _validate_numeric("max_runtime_seconds", max_runtime_seconds, strictly_positive=True)
            if self.reserve_seconds >= runtime:
                raise ValueError(
                    f"reserve_seconds ({self.reserve_seconds}) must be strictly smaller than max_runtime_seconds ({runtime})"
                )
        else:
            runtime = self.max_runtime_seconds

        self.active_budget = ExecutionBudget(
            total_seconds=runtime,
            reserve_seconds=self.reserve_seconds,
            call_timeout=float(self.timeout),
            min_call_timeout=self.min_call_timeout,
        )
        return self.active_budget

    def _apply_pacing(self, budget: Optional[ExecutionBudget]) -> bool:
        """Apply adaptive inter-batch pacing delay before an API call.

        Checks monotonic budget headroom before sleeping.
        Returns False if available budget cannot accommodate minimum call timeout and reserve margin.
        """
        if self._last_api_call_monotonic is None or self.current_pacing <= 0:
            return True

        elapsed = time.monotonic() - self._last_api_call_monotonic
        required_sleep = self.current_pacing - elapsed
        if required_sleep <= 0:
            return True

        if budget is not None:
            max_allowed = budget.max_pacing_allowed()
            if max_allowed <= 0:
                return False
            sleep_duration = min(required_sleep, max_allowed)
        else:
            sleep_duration = required_sleep

        time.sleep(sleep_duration)
        return True

    def _validate_environment(self) -> None:
        """Validate required environment variables when using default llm.chat."""
        if not os.environ.get("LLM_BASE_URL"):
            raise LLMConfigurationError("Missing required environment variable: LLM_BASE_URL")
        if not os.environ.get("LLM_API_KEY"):
            raise LLMConfigurationError("Missing required environment variable: LLM_API_KEY")
        if not self.model and not os.environ.get("LLM_MODEL"):
            raise LLMConfigurationError(
                "Missing LLM_MODEL: provide model parameter or set LLM_MODEL environment variable"
            )

    def _execute_with_retry(
        self,
        phase: str,
        batch_id: str,
        invoke_fn: Callable[[float], Any],
        budget: Optional[ExecutionBudget] = None,
        estimated_tokens: int = 0,
    ) -> Tuple[Optional[Any], Optional[BatchFailureDiagnostic]]:
        """
        Execute an LLM invocation and parsing callable with bounded retries, header floors, and budget checks.

        Rules:
        - HTTP 401/403: raise LLMConfigurationError immediately without retry.
        - HTTP 429, 500, 502, 503, 504: retry with exponential backoff and Retry-After floors.
        - Pre-sleep and post-sleep monotonic budget rechecks prevent late dispatches.
        - Dynamic effective timeout passed to invoke_fn bounds socket inactivity.
        - Network errors & timeouts: retry with exponential backoff.
        - Parsing / structure failures: retry with exponential backoff.
        - Permanent 4xx errors: do not retry, record failure diagnostic.
        - Other unhandled errors: do not retry, record failure diagnostic.
        """
        total_attempts = self.max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(total_attempts):
            if attempt > 0:
                retry_floor = _extract_retry_delay(last_error)
                base_delay = self.retry_delay * (self.backoff_factor ** (attempt - 1))
                if retry_floor is not None:
                    # Enforce strict provider floor: never sleep less than retry_floor!
                    delay = max(retry_floor, base_delay) + random.uniform(0.1, 0.4)
                else:
                    delay = min(self.max_backoff, base_delay) + random.uniform(0.1, 0.4)

                # Pre-sleep budget check
                if budget is not None and not budget.can_attempt_call(additional_delay=delay):
                    return None, BatchFailureDiagnostic(
                        phase=phase,
                        batch_id=batch_id,
                        attempt_count=attempt,
                        error_type="TimeoutBudgetExceeded",
                        error_message="Execution budget cannot accommodate retry delay and reserve margin",
                        timestamp=time.time(),
                    )

                time.sleep(delay)

            # Rate limiter acquisition (accounts for both initial dispatch and each retry attempt)
            if self.rate_limiter is not None:
                if not self.rate_limiter.acquire(estimated_tokens, budget=budget):
                    err_type = (
                        "TokenBudgetExceeded"
                        if estimated_tokens > getattr(self.rate_limiter, "max_tpm", 6000)
                        else "TimeoutBudgetExceeded"
                    )
                    err_msg = (
                        f"Estimated prompt tokens ({estimated_tokens}) exceeds maximum TPM capacity ({getattr(self.rate_limiter, 'max_tpm', 6000)})"
                        if estimated_tokens > getattr(self.rate_limiter, "max_tpm", 6000)
                        else "Execution budget exhausted while waiting for rate limit window capacity"
                    )
                    return None, BatchFailureDiagnostic(
                        phase=phase,
                        batch_id=batch_id,
                        attempt_count=attempt,
                        error_type=err_type,
                        error_message=err_msg,
                        timestamp=time.time(),
                    )

            # Mandatory post-pacing / post-retry-sleep budget recheck
            if budget is not None:
                effective_timeout = budget.calculate_effective_timeout()
                if effective_timeout < self.min_call_timeout:
                    return None, BatchFailureDiagnostic(
                        phase=phase,
                        batch_id=batch_id,
                        attempt_count=attempt,
                        error_type="TimeoutBudgetExceeded",
                        error_message="Execution budget exhausted before dispatch",
                        timestamp=time.time(),
                    )
            else:
                effective_timeout = float(self.timeout)

            try:
                try:
                    result = invoke_fn(effective_timeout)
                    self._consecutive_successes += 1
                    if self._consecutive_successes >= 3:
                        self.current_pacing = max(self.min_pacing, self.current_pacing * 0.95)
                    return result, None
                finally:
                    self._last_api_call_monotonic = time.monotonic()
            except urllib.error.HTTPError as err:
                last_error = err
                self._consecutive_successes = 0
                if self.rate_limiter is not None and getattr(err, "headers", None):
                    self.rate_limiter.update_from_headers(dict(err.headers))
                if err.code == 429:
                    self.current_pacing = min(self.max_pacing, self.current_pacing * 1.5)

                # 1. Immediate configuration failure on 401/403
                if err.code in (401, 403):
                    if getattr(self, "allow_offline", False):
                        self.offline_mode = True
                        return None, BatchFailureDiagnostic(
                            phase=phase,
                            batch_id=batch_id,
                            attempt_count=attempt,
                            error_type="OfflineModeNoCache",
                            error_message=f"LLM API authentication failed (HTTP {err.code}); switched to offline mode",
                            timestamp=time.time(),
                        )
                    raise LLMConfigurationError(
                        f"LLM API authentication failed (HTTP {err.code}). Check LLM_API_KEY and LLM_BASE_URL."
                    ) from None

                # 2. Retryable HTTP errors: 429 and 5xx
                if err.code in (429, 500, 502, 503, 504):
                    continue

                # 3. Permanent 4xx errors: do not retry
                break
            except urllib.error.URLError as err:
                last_error = err
                self._consecutive_successes = 0
                if getattr(self, "allow_offline", False):
                    self.offline_mode = True
                    return None, BatchFailureDiagnostic(
                        phase=phase,
                        batch_id=batch_id,
                        attempt_count=attempt,
                        error_type="OfflineModeNoCache",
                        error_message=f"Network connection failed ({err}); switched to offline mode",
                        timestamp=time.time(),
                    )
                # Retryable network/connection error
                continue
            except (TimeoutError, socket.timeout) as err:
                last_error = err
                self._consecutive_successes = 0
                if getattr(self, "allow_offline", False):
                    self.offline_mode = True
                    return None, BatchFailureDiagnostic(
                        phase=phase,
                        batch_id=batch_id,
                        attempt_count=attempt,
                        error_type="OfflineModeNoCache",
                        error_message=f"Network timeout ({err}); switched to offline mode",
                        timestamp=time.time(),
                    )
                # Retryable timeout
                continue
            except (ResponseParsingError, ResponseStructureError) as err:
                last_error = err
                self._consecutive_successes = 0
                # Retryable parsing / structure error
                continue
            except Exception as err:
                last_error = err
                self._consecutive_successes = 0
                # Other unexpected error: do not retry
                break

        # If exhausted all attempts or encountered a non-retryable error
        err_type = (
            f"HTTPError:{last_error.code}"
            if isinstance(last_error, urllib.error.HTTPError)
            else type(last_error).__name__
        )
        diagnostic = BatchFailureDiagnostic(
            phase=phase,
            batch_id=batch_id,
            attempt_count=attempt + 1,
            error_type=err_type,
            error_message=_sanitize_error_message(last_error),
            timestamp=time.time(),
        )
        return None, diagnostic

    def _record_parser_counts(
        self,
        batch_diag: ChatParserDiagnostics,
        target_diag: Optional[ChatParserDiagnostics] = None,
    ) -> None:
        """Accumulate parser rejection/acceptance counters onto self.diagnostics and optional caller target."""
        self.diagnostics.raw_goals_returned += batch_diag.goals_total_raw
        self.diagnostics.goals_accepted += batch_diag.goals_accepted
        self.diagnostics.raw_open_loops_returned += batch_diag.open_loops_total_raw
        self.diagnostics.open_loops_accepted += batch_diag.open_loops_accepted

        if target_diag is not None:
            target_diag.open_loops_total_raw += batch_diag.open_loops_total_raw
            target_diag.open_loops_non_dict += batch_diag.open_loops_non_dict
            target_diag.open_loops_missing_required_fields += batch_diag.open_loops_missing_required_fields
            target_diag.open_loops_invalid_enums += batch_diag.open_loops_invalid_enums
            target_diag.open_loops_invalid_or_missing_is_genuine += batch_diag.open_loops_invalid_or_missing_is_genuine
            target_diag.open_loops_rejected_not_genuine += batch_diag.open_loops_rejected_not_genuine
            target_diag.open_loops_invalid_or_ungrounded_source += batch_diag.open_loops_invalid_or_ungrounded_source
            target_diag.open_loops_accepted += batch_diag.open_loops_accepted

            target_diag.goals_total_raw += batch_diag.goals_total_raw
            target_diag.goals_non_dict += batch_diag.goals_non_dict
            target_diag.goals_missing_required_fields += batch_diag.goals_missing_required_fields
            target_diag.goals_invalid_enums += batch_diag.goals_invalid_enums
            target_diag.goals_invalid_or_missing_is_genuine += batch_diag.goals_invalid_or_missing_is_genuine
            target_diag.goals_rejected_not_genuine += batch_diag.goals_rejected_not_genuine
            target_diag.goals_invalid_or_ungrounded_evidence += batch_diag.goals_invalid_or_ungrounded_evidence
            target_diag.goals_accepted += batch_diag.goals_accepted

            target_diag.rejection_details.extend(batch_diag.rejection_details)

    def interpret_chat_batch(
        self,
        batch: EvidenceBatch,
        context: Optional[ConversationContext] = None,
        budget: Optional[ExecutionBudget] = None,
        parser_diagnostics: Optional[ChatParserDiagnostics] = None,
    ) -> Tuple[List[InterpretedOpenLoop], List[InterpretedGoal], Optional[BatchFailureDiagnostic]]:
        """
        Interpret a single chat evidence batch into open loops and goals.

        Args:
            batch: EvidenceBatch instance.
            context: Optional ConversationContext for referential verification.
            budget: Optional ExecutionBudget instance overriding active session budget.
            parser_diagnostics: Optional ChatParserDiagnostics instance to record parsing details.

        Returns:
            Tuple of (open_loops, goals, optional_failure_diagnostic).
        """
        if not isinstance(batch, EvidenceBatch):
            raise TypeError(f"Expected EvidenceBatch, got {type(batch).__name__}")

        if batch.candidate_count == 0 or batch.message_count == 0:
            self.diagnostics.successful_parsed_batches += 1
            return [], [], None

        self.last_call_cached = False
        active_budget = budget or self.active_budget
        batch_parser_diag = ChatParserDiagnostics()

        messages = build_chat_interpretation_prompt(batch)
        base_url = os.environ.get("LLM_BASE_URL", "")
        model_name = self.model or os.environ.get("LLM_MODEL", "")

        cache_key = self.cache.compute_key(
            phase="chat",
            prompt_version="chat_interpretation_v1.0",
            base_url=base_url,
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )

        # 1. Cache-first lookup & phase parser revalidation
        cached_raw_text = self.cache.get(cache_key)
        if cached_raw_text is not None:
            try:
                loops, goals = parse_chat_interpretation_response(
                    raw_text=cached_raw_text,
                    batch=batch,
                    context=context,
                    diagnostics=batch_parser_diag,
                )
                self.last_call_cached = True
                self.diagnostics.cache_hits += 1
                self.diagnostics.successful_parsed_batches += 1
                self._record_parser_counts(batch_parser_diag, parser_diagnostics)
                return loops, goals, None
            except (ResponseParsingError, ResponseStructureError):
                self.cache.evict(cache_key)
                # Stale or invalid cache entry evicted; fall through to network call

        if self.offline_mode:
            self.diagnostics.offline_cache_misses += 1
            self.diagnostics.failed_batches += 1
            diagnostic = BatchFailureDiagnostic(
                phase="chat",
                batch_id=batch.chat_id,
                attempt_count=0,
                error_type="OfflineModeNoCache",
                error_message="Offline mode active: batch not found in local cache and live API calls disabled",
                timestamp=time.time(),
            )
            return [], [], diagnostic

        # 2. Pre-pacing budget check & pacing delay (network calls only)
        if not self._apply_pacing(active_budget):
            self.diagnostics.failed_batches += 1
            diagnostic = BatchFailureDiagnostic(
                phase="chat",
                batch_id=batch.chat_id,
                attempt_count=0,
                error_type="TimeoutBudgetExceeded",
                error_message="Execution budget exhausted during pacing",
                timestamp=time.time(),
            )
            return [], [], diagnostic

        # 3. Network execution with dynamic effective socket timeout
        estimated_tokens = estimate_prompt_tokens(messages)

        def _invoke(call_timeout: float) -> Tuple[str, List[InterpretedOpenLoop], List[InterpretedGoal]]:
            self._live_call_count += 1
            self.diagnostics.live_llm_calls += 1
            resp_metadata: Optional[Dict[str, Any]] = None
            try:
                chat_resp = self.chat_fn(
                    messages=messages,
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=call_timeout,
                    return_metadata=True,
                )
            except TypeError:
                chat_resp = self.chat_fn(
                    messages=messages,
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=call_timeout,
                )

            if isinstance(chat_resp, tuple) and len(chat_resp) == 2:
                raw_text, resp_metadata = chat_resp
            else:
                raw_text = chat_resp

            if self.rate_limiter is not None and isinstance(resp_metadata, dict):
                headers = resp_metadata.get("headers")
                if headers:
                    self.rate_limiter.update_from_headers(headers)

            loops, goals = parse_chat_interpretation_response(
                raw_text=raw_text,
                batch=batch,
                context=context,
                diagnostics=batch_parser_diag,
            )
            return raw_text, loops, goals

        parsed_result, diagnostic = self._execute_with_retry(
            phase="chat",
            batch_id=batch.chat_id,
            invoke_fn=_invoke,
            budget=active_budget,
            estimated_tokens=estimated_tokens,
        )

        if parsed_result is not None:
            raw_text, loops, goals = parsed_result
            self.diagnostics.successful_parsed_batches += 1
            self._record_parser_counts(batch_parser_diag, parser_diagnostics)
            self.cache.set(cache_key, raw_text, metadata={"model": model_name, "batch_id": batch.chat_id})
            return loops, goals, None

        self.diagnostics.failed_batches += 1
        return [], [], diagnostic

    def interpret_participant_batch(
        self,
        dossiers: Sequence[Dict[str, Any]],
        chat_metadata: Optional[Dict[str, Any]] = None,
        context: Optional[ConversationContext] = None,
        budget: Optional[ExecutionBudget] = None,
    ) -> Tuple[List[InterpretedPerson], List[InterpretedOrganisation], Optional[BatchFailureDiagnostic]]:
        """
        Interpret a single batch of participant dossiers into people and organisations.

        Args:
            dossiers: Sequence of participant dossier dictionaries.
            chat_metadata: Optional dictionary describing relevant chat metadata.
            context: Optional ConversationContext for entity grounding.
            budget: Optional ExecutionBudget instance overriding active session budget.

        Returns:
            Tuple of (people, organisations, optional_failure_diagnostic).
        """
        if not dossiers:
            self.diagnostics.successful_parsed_batches += 1
            return [], [], None

        self.last_call_cached = False
        active_budget = budget or self.active_budget

        pids = [
            d.get("person_id") or d.get("participant_id") or "unknown"
            for d in dossiers
            if isinstance(d, dict)
        ]
        batch_id = ",".join(pids)

        messages = build_participant_interpretation_prompt(
            dossiers=dossiers,
            chat_metadata=chat_metadata,
        )
        base_url = os.environ.get("LLM_BASE_URL", "")
        model_name = self.model or os.environ.get("LLM_MODEL", "")

        cache_key = self.cache.compute_key(
            phase="participant",
            prompt_version="participant_interpretation_v1.0",
            base_url=base_url,
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )

        # 1. Cache-first lookup & phase parser revalidation
        cached_raw_text = self.cache.get(cache_key)
        if cached_raw_text is not None:
            try:
                people, orgs = parse_participant_interpretation_response(
                    raw_text=cached_raw_text,
                    dossiers=dossiers,
                    chat_metadata=chat_metadata,
                    context=context,
                )
                self.last_call_cached = True
                self.diagnostics.cache_hits += 1
                self.diagnostics.successful_parsed_batches += 1
                return people, orgs, None
            except (ResponseParsingError, ResponseStructureError):
                self.cache.evict(cache_key)
                # Stale or invalid cache entry evicted; fall through to network call

        if self.offline_mode:
            self.diagnostics.offline_cache_misses += 1
            self.diagnostics.failed_batches += 1
            diagnostic = BatchFailureDiagnostic(
                phase="participant",
                batch_id=batch_id,
                attempt_count=0,
                error_type="OfflineModeNoCache",
                error_message="Offline mode active: batch not found in local cache and live API calls disabled",
                timestamp=time.time(),
            )
            return [], [], diagnostic

        # 2. Pre-pacing budget check & pacing delay (network calls only)
        if not self._apply_pacing(active_budget):
            self.diagnostics.failed_batches += 1
            diagnostic = BatchFailureDiagnostic(
                phase="participant",
                batch_id=batch_id,
                attempt_count=0,
                error_type="TimeoutBudgetExceeded",
                error_message="Execution budget exhausted during pacing",
                timestamp=time.time(),
            )
            return [], [], diagnostic

        # 3. Network execution with dynamic effective socket timeout
        estimated_tokens = estimate_prompt_tokens(messages)

        def _invoke(call_timeout: float) -> Tuple[str, List[InterpretedPerson], List[InterpretedOrganisation]]:
            self._live_call_count += 1
            self.diagnostics.live_llm_calls += 1
            resp_metadata: Optional[Dict[str, Any]] = None
            try:
                chat_resp = self.chat_fn(
                    messages=messages,
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=call_timeout,
                    return_metadata=True,
                )
            except TypeError:
                chat_resp = self.chat_fn(
                    messages=messages,
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=call_timeout,
                )

            if isinstance(chat_resp, tuple) and len(chat_resp) == 2:
                raw_text, resp_metadata = chat_resp
            else:
                raw_text = chat_resp

            if self.rate_limiter is not None and isinstance(resp_metadata, dict):
                headers = resp_metadata.get("headers")
                if headers:
                    self.rate_limiter.update_from_headers(headers)

            people, orgs = parse_participant_interpretation_response(
                raw_text=raw_text,
                dossiers=dossiers,
                chat_metadata=chat_metadata,
                context=context,
            )
            return raw_text, people, orgs

        parsed_result, diagnostic = self._execute_with_retry(
            phase="participant",
            batch_id=batch_id,
            invoke_fn=_invoke,
            budget=active_budget,
            estimated_tokens=estimated_tokens,
        )

        if parsed_result is not None:
            raw_text, people, orgs = parsed_result
            self.diagnostics.successful_parsed_batches += 1
            self.cache.set(cache_key, raw_text, metadata={"model": model_name, "batch_id": batch_id})
            return people, orgs, None

        self.diagnostics.failed_batches += 1
        return [], [], diagnostic

    def interpret(
        self,
        context: ConversationContext,
        candidates: Sequence[Dict[str, Any]],
        budget: Optional[ExecutionBudget] = None,
    ) -> InterpretationResult:
        """
        Run end-to-end interpretation over conversation context and candidate evidence.

        Args:
            context: ConversationContext from Step 3.3.
            candidates: Candidate evidence records from Step 3.4.1.
            budget: Optional ExecutionBudget instance overriding active session budget.

        Returns:
            InterpretationResult containing all validated intermediate records and diagnostics.
        """
        if not isinstance(context, ConversationContext):
            raise TypeError(f"Expected ConversationContext, got {type(context).__name__}")

        active_budget = budget or self.active_budget or self.start_session()
        diagnostics = InterpretationDiagnostics()
        self.diagnostics = diagnostics
        all_open_loops: List[InterpretedOpenLoop] = []
        all_goals: List[InterpretedGoal] = []
        all_people: List[InterpretedPerson] = []
        all_orgs: List[InterpretedOrganisation] = []

        # ======================================================================
        # Phase 1: Chat-level Interpretation (Open Loops & Goals)
        # ======================================================================
        chat_batches = build_evidence_batches(context, candidates)
        diagnostics.chat_batches_total = len(chat_batches)
        diagnostics.chat_batches_created = len(chat_batches)

        for batch in chat_batches:
            loops, goals, diag = self.interpret_chat_batch(batch, context=context, budget=active_budget)
            if diag is not None:
                if diag.error_type == "TimeoutBudgetExceeded":
                    diagnostics.batches_unattempted += 1
                diagnostics.chat_batches_failed += 1
                diagnostics.failures.append(diag)
            else:
                diagnostics.chat_batches_succeeded += 1
                if getattr(self, "last_call_cached", False):
                    diagnostics.batches_cached += 1
                all_open_loops.extend(loops)
                all_goals.extend(goals)

        # ======================================================================
        # Phase 2: Participant-level Interpretation (People & Organisations)
        # ======================================================================
        dossier_batches = build_participant_dossiers(
            context=context,
            max_participants_per_batch=self.max_participants_per_batch,
            messages_per_dossier=self.messages_per_dossier,
        )
        diagnostics.participant_batches_total = len(dossier_batches)

        for dossiers, chat_meta in dossier_batches:
            people, orgs, diag = self.interpret_participant_batch(
                dossiers=dossiers,
                chat_metadata=chat_meta,
                context=context,
                budget=active_budget,
            )
            if diag is not None:
                if diag.error_type == "TimeoutBudgetExceeded":
                    diagnostics.batches_unattempted += 1
                diagnostics.participant_batches_failed += 1
                diagnostics.failures.append(diag)
            else:
                diagnostics.participant_batches_succeeded += 1
                if getattr(self, "last_call_cached", False):
                    diagnostics.batches_cached += 1
                all_people.extend(people)
                all_orgs.extend(orgs)

        return InterpretationResult(
            open_loops=all_open_loops,
            goals=all_goals,
            people=all_people,
            organisations=all_orgs,
            diagnostics=diagnostics,
        )

