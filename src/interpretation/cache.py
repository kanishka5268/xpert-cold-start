"""Local checkpointing and monotonic execution budgeting for LLM interpretation.

Provides:
- ExecutionBudget: Monotonic deadline tracking, reserve margin preservation, and dynamic socket timeout bounding.
- InterpretationCache: Content-addressed SHA-256 caching of validated raw model responses with atomic persistence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _validate_numeric(name: str, val: Any, *, strictly_positive: bool = False, non_negative: bool = False) -> float:
    """Validate numeric config values, strictly rejecting bools, non-numbers, NaN, inf, and invalid bounds."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{name} must be a numeric int or float, got {type(val).__name__} ({val!r})")
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be a finite number, got {val!r}")
    if strictly_positive and fval <= 0.0:
        raise ValueError(f"{name} must be strictly positive (> 0), got {val!r}")
    if non_negative and fval < 0.0:
        raise ValueError(f"{name} must be non-negative (>= 0), got {val!r}")
    return fval


@dataclass
class ExecutionBudget:
    """Monotonic execution budget manager for multi-batch interpretation runs.

    Uses time.monotonic() to strictly guard against wall-clock shifts, NTP adjustments,
    and leap seconds.
    """

    total_seconds: float = 1500.0          # Total allowed runtime (default 25 min)
    reserve_seconds: float = 45.0          # Headroom reserved for resolver deduplication and output I/O
    call_timeout: float = 120.0            # Configured per-call socket timeout
    min_call_timeout: float = 1.0          # Minimum viable socket timeout floor
    start_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        """Validate numeric runtime and timeout configuration strictly."""
        self.total_seconds = _validate_numeric("total_seconds", self.total_seconds, strictly_positive=True)
        self.call_timeout = _validate_numeric("call_timeout", self.call_timeout, strictly_positive=True)
        self.min_call_timeout = _validate_numeric("min_call_timeout", self.min_call_timeout, strictly_positive=True)
        self.reserve_seconds = _validate_numeric("reserve_seconds", self.reserve_seconds, non_negative=True)
        if self.reserve_seconds >= self.total_seconds:
            raise ValueError(
                f"reserve_seconds ({self.reserve_seconds}) must be strictly smaller than total_seconds ({self.total_seconds})"
            )
        if self.start_monotonic is not None:
            if isinstance(self.start_monotonic, bool) or not isinstance(self.start_monotonic, (int, float)) or not math.isfinite(float(self.start_monotonic)):
                raise TypeError(f"start_monotonic must be a finite number, got {self.start_monotonic!r}")
            self.start_monotonic = float(self.start_monotonic)

    @property
    def remaining_seconds(self) -> float:
        """Return remaining monotonic budget in seconds, floored at 0.0."""
        elapsed = time.monotonic() - self.start_monotonic
        return max(0.0, self.total_seconds - elapsed)

    def can_attempt_call(self, additional_delay: float = 0.0) -> bool:
        """Return True only if remaining budget can accommodate additional delay, minimum call timeout, and reserve."""
        return self.remaining_seconds >= (additional_delay + self.min_call_timeout + self.reserve_seconds)

    def max_pacing_allowed(self) -> float:
        """Return maximum allowable pacing sleep duration without eating into call timeout or reserve."""
        return max(0.0, self.remaining_seconds - (self.min_call_timeout + self.reserve_seconds))

    def calculate_effective_timeout(self) -> float:
        """Calculate dynamic socket timeout bounded by remaining monotonic headroom.

        effective_timeout = min(call_timeout, remaining_seconds - reserve_seconds)

        Returns 0.0 if remaining headroom is strictly less than min_call_timeout.
        """
        available = self.remaining_seconds - self.reserve_seconds
        if available < self.min_call_timeout:
            return 0.0
        return min(self.call_timeout, available)


class InterpretationCache:
    """Local, content-aware checkpoint cache for validated raw model completions.

    Keyed by deterministic SHA-256 hash of all request-affecting parameters.
    Saves and loads strictly local data using atomic filesystem replacement.
    Strictly excludes API keys and authorization secrets.
    """

    def __init__(
        self,
        cache_path: str = ".cache/llm_cache.json",
        enabled: bool = True,
    ) -> None:
        self.cache_path = cache_path
        self.enabled = enabled
        self._data: Dict[str, Dict[str, Any]] = {}

        if self.enabled:
            self._load()

    def _load(self) -> None:
        """Load cache entries from local JSON file if present and valid."""
        if not os.path.exists(self.cache_path):
            self._data = {}
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._data = loaded
                else:
                    self._data = {}
        except Exception as err:
            logger.warning("Failed to load interpretation cache from %s: %s. Starting fresh.", self.cache_path, err)
            self._data = {}

    def compute_key(
        self,
        phase: str,
        prompt_version: str,
        base_url: str,
        model: str,
        temperature: float,
        response_format: Any,
        messages: Any,
        max_tokens: Any = None,
    ) -> str:
        """Compute canonical SHA-256 key from request-affecting parameters."""
        canonical_payload = {
            "base_url": (base_url or "").rstrip("/"),
            "max_tokens": max_tokens,
            "messages": messages,
            "model": model or "",
            "phase": phase,
            "prompt_version": prompt_version,
            "response_format": response_format,
            "temperature": temperature,
        }
        serialized = json.dumps(canonical_payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """Retrieve cached raw response text if key exists and cache is enabled."""
        if not self.enabled:
            return None
        entry = self._data.get(key)
        if isinstance(entry, dict):
            raw = entry.get("raw_text")
            if isinstance(raw, str):
                return raw
        return None

    def set(self, key: str, raw_text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store validated raw response text and persist atomically to disk."""
        if not self.enabled or not isinstance(raw_text, str):
            return

        entry: Dict[str, Any] = {
            "raw_text": raw_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata and isinstance(metadata, dict):
            # Exclude any accidental secret-bearing keys
            safe_meta = {
                k: v
                for k, v in metadata.items()
                if not any(s in k.lower() for s in ("key", "auth", "token", "secret", "bearer"))
            }
            entry.update(safe_meta)

        self._data[key] = entry
        self._persist()

    def evict(self, key: str) -> None:
        """Evict an invalid or stale cache entry and persist changes."""
        if not self.enabled:
            return
        if key in self._data:
            del self._data[key]
            self._persist()

    def _persist(self) -> None:
        """Atomically persist cache dictionary to disk using a temporary file and fsync."""
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        temp_path = f"{self.cache_path}.tmp.{os.getpid()}"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.cache_path)
        except Exception as err:
            logger.warning("Failed to atomically persist cache to %s: %s", self.cache_path, err)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

