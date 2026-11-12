"""Evidence batching module for WhatsApp conversation evidence interpretation."""

from collections import defaultdict
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

VALID_CATEGORIES: Set[str] = {
    "commitment",
    "question",
    "followup",
    "deadline",
    "person_mention",
    "organisation_mention",
    "goal",
}


def _get_chat_messages(context: Any, chat_id: str) -> List[Dict[str, Any]]:
    """
    Safely retrieve chronological messages for a chat from ConversationContext or dataset dict.
    """
    if context is None:
        return []

    # If ConversationContext object with get_chat()
    if hasattr(context, "get_chat"):
        chat = context.get_chat(chat_id)
        if chat and isinstance(chat, dict):
            return chat.get("messages", [])

    # If object with chats dictionary
    if hasattr(context, "chats") and isinstance(context.chats, dict):
        chat = context.chats.get(chat_id)
        if chat and isinstance(chat, dict):
            return chat.get("messages", [])

    # If dataset dictionary
    if isinstance(context, dict):
        if "messages_by_chat" in context and chat_id in context["messages_by_chat"]:
            return context["messages_by_chat"][chat_id]
        if "chats" in context and isinstance(context["chats"], dict) and chat_id in context["chats"]:
            return context["chats"][chat_id].get("messages", [])
        if "dataset" in context and isinstance(context["dataset"], dict):
            return _get_chat_messages(context["dataset"], chat_id)

    return []


def _find_message_chat(context: Any, message_id: str) -> Optional[str]:
    """Find the chat_id of a message in context, or None if not found anywhere."""
    if context is None:
        return None

    # 1. Dataset messages_by_id or messages_by_chat
    if hasattr(context, "dataset") and isinstance(context.dataset, dict):
        msgs_by_id = context.dataset.get("messages_by_id")
        if isinstance(msgs_by_id, dict) and message_id in msgs_by_id:
            return msgs_by_id[message_id].get("chat_id")
        msgs_by_chat = context.dataset.get("messages_by_chat")
        if isinstance(msgs_by_chat, dict):
            for c_id, msgs in msgs_by_chat.items():
                if any(m.get("id") == message_id for m in msgs):
                    return c_id

    # 2. ConversationContext.chats
    if hasattr(context, "chats") and isinstance(context.chats, dict):
        for c_id, chat in context.chats.items():
            if isinstance(chat, dict) and any(m.get("id") == message_id for m in chat.get("messages", [])):
                return c_id

    # 3. Dict context
    if isinstance(context, dict):
        msgs_by_id = context.get("messages_by_id")
        if isinstance(msgs_by_id, dict) and message_id in msgs_by_id:
            return msgs_by_id[message_id].get("chat_id")
        msgs_by_chat = context.get("messages_by_chat")
        if isinstance(msgs_by_chat, dict):
            for c_id, msgs in msgs_by_chat.items():
                if any(m.get("id") == message_id for m in msgs):
                    return c_id
        chats = context.get("chats")
        if isinstance(chats, dict):
            for c_id, chat in chats.items():
                if isinstance(chat, dict) and any(m.get("id") == message_id for m in chat.get("messages", [])):
                    return c_id

    return None


@dataclass
class ConversationSnippet:
    """
    A contiguous slice of conversation context and its associated candidates within a single chat.

    Attributes:
        snippet_id: Unique identifier for the snippet within the batch (e.g. 'C01_s0').
        messages: Chronologically ordered list of message dictionaries in this snippet.
        candidates: List of candidate evidence dictionaries associated with this snippet.
        candidate_message_ids: List of unique candidate message IDs present in this snippet.
        is_oversized: True if the snippet context exceeds message limits.
    """

    snippet_id: str
    messages: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    candidate_message_ids: List[str]
    is_oversized: bool = False

    @property
    def message_count(self) -> int:
        """Total number of messages in this snippet."""
        return len(self.messages)

    @property
    def candidate_count(self) -> int:
        """Total number of candidate evidence records in this snippet."""
        return len(self.candidates)

    def to_dict(self) -> Dict[str, Any]:
        """Convert snippet to a serializable dictionary."""
        return {
            "snippet_id": self.snippet_id,
            "messages": self.messages,
            "candidates": self.candidates,
            "candidate_message_ids": self.candidate_message_ids,
            "is_oversized": self.is_oversized,
            "message_count": self.message_count,
            "candidate_count": self.candidate_count,
        }


@dataclass
class CandidateBatchTrace:
    """Diagnostic trace recording how a candidate was routed through batching and prompt inclusion."""

    message_id: str
    chat_id: str
    category: str
    entered_batch: bool = False
    batch_id: Optional[str] = None
    included_in_prompt: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to a serializable dictionary."""
        return {
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "category": self.category,
            "entered_batch": self.entered_batch,
            "batch_id": self.batch_id,
            "included_in_prompt": self.included_in_prompt,
        }


@dataclass
class EvidenceBatch:
    """
    A batch of conversation context messages and associated candidate evidence records.

    Attributes:
        chat_id: Identifier of the chat this batch belongs to.
        messages: Chronologically ordered list of message dictionaries providing conversation context.
        candidates: List of candidate evidence dictionaries associated with this batch.
        candidate_message_ids: List of unique candidate message IDs present in this batch.
        categories: List of unique candidate categories represented in this batch.
        is_oversized: True if the context window exceeds max_messages_per_batch.
        snippets: List of ConversationSnippet instances composing this batch.
    """

    chat_id: str
    messages: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    candidate_message_ids: List[str]
    categories: List[str]
    is_oversized: bool = False
    snippets: List[ConversationSnippet] = field(default_factory=list)
    batch_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Ensure snippets is populated even for legacy single-window batches."""
        if not self.batch_id:
            self.batch_id = f"batch_{self.chat_id}"
        if not self.snippets and self.messages:
            cand_ids = list(self.candidate_message_ids)
            self.snippets = [
                ConversationSnippet(
                    snippet_id=f"{self.chat_id}_s0",
                    messages=self.messages,
                    candidates=self.candidates,
                    candidate_message_ids=cand_ids,
                    is_oversized=self.is_oversized,
                )
            ]

    @property
    def message_count(self) -> int:
        """Total number of context messages in this batch."""
        return len(self.messages)

    @property
    def candidate_count(self) -> int:
        """Total number of candidate evidence records in this batch."""
        return len(self.candidates)

    @property
    def snippet_count(self) -> int:
        """Total number of conversation snippets in this batch."""
        return len(self.snippets)

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to a serializable dictionary."""
        return {
            "batch_id": self.batch_id,
            "chat_id": self.chat_id,
            "messages": self.messages,
            "candidates": self.candidates,
            "candidate_message_ids": self.candidate_message_ids,
            "categories": self.categories,
            "is_oversized": self.is_oversized,
            "message_count": self.message_count,
            "candidate_count": self.candidate_count,
            "snippets": [s.to_dict() if hasattr(s, "to_dict") else s for s in self.snippets],
            "snippet_count": self.snippet_count,
        }


def trace_candidates_in_batches(
    batches: Sequence[EvidenceBatch],
    candidates: Sequence[Dict[str, Any]],
) -> List[CandidateBatchTrace]:
    """
    Trace candidate evidence records against a sequence of evidence batches.

    For each candidate, identifies:
    - message_id
    - chat_id
    - category
    - entered_batch: True if candidate was placed in an EvidenceBatch
    - batch_id: ID of the batch containing the candidate
    - included_in_prompt: True if candidate is present in the batch and will be serialized in the prompt
    """
    batch_by_cand_id: Dict[str, EvidenceBatch] = {}
    for b in batches:
        b_id = getattr(b, "batch_id", None) or f"batch_{b.chat_id}"
        for mid in b.candidate_message_ids:
            batch_by_cand_id[mid] = b

    traces: List[CandidateBatchTrace] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        mid = str(cand.get("message_id", ""))
        cid = str(cand.get("chat_id", ""))
        cat = str(cand.get("category", ""))

        b = batch_by_cand_id.get(mid)
        if b is not None and b.chat_id == cid:
            b_id = getattr(b, "batch_id", None) or f"batch_{b.chat_id}"
            traces.append(
                CandidateBatchTrace(
                    message_id=mid,
                    chat_id=cid,
                    category=cat,
                    entered_batch=True,
                    batch_id=b_id,
                    included_in_prompt=True,
                )
            )
        else:
            traces.append(
                CandidateBatchTrace(
                    message_id=mid,
                    chat_id=cid,
                    category=cat,
                    entered_batch=False,
                    batch_id=None,
                    included_in_prompt=False,
                )
            )
    return traces

class EvidenceBatcher:
    """
    Groups candidate evidence records by chat, retrieves surrounding conversation context,
    merges overlapping context windows, and splits batches according to configurable limits.
    """

    def __init__(
        self,
        window_before: int = 2,
        window_after: int = 4,
        max_candidates_per_batch: int = 8,
        max_messages_per_batch: int = 25,
        oversized_strategy: str = "keep_intact",
        allow_disjoint: bool = False,
    ):
        """
        Initialize the evidence batcher with explicit limits.

        Args:
            window_before: Number of messages before each candidate to include in context.
            window_after: Number of messages after each candidate to include in context.
            max_candidates_per_batch: Maximum number of candidate records allowed in a single batch.
            max_messages_per_batch: Maximum number of context messages allowed in a single batch.
            oversized_strategy: Strategy when a single candidate window exceeds max_messages_per_batch:
                                - 'keep_intact': Keep the full required context window, mark is_oversized=True.
                                - 'trim_to_limit': Trim window to max_messages_per_batch centered around candidate,
                                                   guaranteeing candidate message is present, mark is_oversized=True.
            allow_disjoint: If True, non-overlapping candidate snippets within the same chat can be packed
                            together into multi-snippet batches. Defaults to False for contiguous batching.
        """
        if window_before < 0:
            raise ValueError(f"window_before must be >= 0, got {window_before}")
        if window_after < 0:
            raise ValueError(f"window_after must be >= 0, got {window_after}")
        if max_candidates_per_batch <= 0:
            raise ValueError(f"max_candidates_per_batch must be > 0, got {max_candidates_per_batch}")
        if max_messages_per_batch <= 0:
            raise ValueError(f"max_messages_per_batch must be > 0, got {max_messages_per_batch}")
        if oversized_strategy not in ("keep_intact", "trim_to_limit"):
            raise ValueError(f"Unknown oversized_strategy: {oversized_strategy}")

        self.window_before = window_before
        self.window_after = window_after
        self.max_candidates_per_batch = max_candidates_per_batch
        self.max_messages_per_batch = max_messages_per_batch
        self.oversized_strategy = oversized_strategy
        self.allow_disjoint = bool(allow_disjoint)
        self.last_traces: List[CandidateBatchTrace] = []

    def _validate_candidates(self, context: Any, candidates: Sequence[Dict[str, Any]]) -> None:
        """
        Validate that every candidate record has a non-empty chat_id, message_id,
        valid category, and references a real source message in the specified chat.

        Raises:
            ValueError: If any candidate record is invalid or references a nonexistent
                        or cross-chat message.
        """
        chat_msg_sets: Dict[str, Set[str]] = {}

        for cand in candidates:
            if not isinstance(cand, dict):
                raise ValueError(f"Candidate record must be a dict, got {type(cand).__name__}")

            cid = cand.get("chat_id")
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError(f"Candidate record missing valid non-empty 'chat_id': {cand!r}")
            if cid != cid.strip():
                raise ValueError(f"Candidate record 'chat_id' contains leading or trailing whitespace: {cid!r}")

            mid = cand.get("message_id")
            if not isinstance(mid, str) or not mid.strip():
                raise ValueError(f"Candidate record missing valid non-empty 'message_id': {cand!r}")
            if mid != mid.strip():
                raise ValueError(f"Candidate record 'message_id' contains leading or trailing whitespace: {mid!r}")

            cat = cand.get("category")
            if not isinstance(cat, str) or not cat.strip():
                raise ValueError(f"Candidate {mid!r} missing valid non-empty 'category': {cand!r}")
            if cat != cat.strip():
                raise ValueError(f"Candidate {mid!r} 'category' contains leading or trailing whitespace: {cat!r}")
            if cat not in VALID_CATEGORIES:
                raise ValueError(
                    f"Candidate {mid!r} has invalid category {cat!r}. Must be one of {sorted(VALID_CATEGORIES)}"
                )

            # Check if source message exists in the specified chat
            if cid not in chat_msg_sets:
                msgs = _get_chat_messages(context, cid)
                chat_msg_sets[cid] = {m["id"] for m in msgs if isinstance(m, dict) and "id" in m}

            if mid not in chat_msg_sets[cid]:
                actual_cid = _find_message_chat(context, mid)
                if actual_cid is not None:
                    raise ValueError(
                        f"Candidate source message {mid!r} belongs to chat {actual_cid!r}, "
                        f"but candidate specifies chat {cid!r}"
                    )
                raise ValueError(
                    f"Candidate source message {mid!r} not found in conversation context (chat {cid!r})"
                )

    def batch(
        self,
        context: Any,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[EvidenceBatch]:
        """
        Batch candidate evidence records into chronologically ordered, chat-isolated batches.

        Args:
            context: ConversationContext instance or loaded dataset dictionary.
            candidates: Sequence of candidate evidence dictionaries from candidate extraction.

        Returns:
            List of EvidenceBatch instances in deterministic order.
        """
        if not candidates:
            return []

        # Validate candidate records strictly before batching
        self._validate_candidates(context, candidates)

        # 1. Group candidates by chat_id
        candidates_by_chat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for cand in candidates:
            cid = cand["chat_id"]
            candidates_by_chat[cid].append(cand)

        # 2. Sort chats deterministically by earliest candidate timestamp and chat_id
        def _chat_sort_key(cid: str) -> Tuple[int, str]:
            cands = candidates_by_chat[cid]
            min_ts = min((c.get("timestamp") or 0 for c in cands), default=0)
            return (min_ts, cid)

        sorted_chat_ids = sorted(candidates_by_chat.keys(), key=_chat_sort_key)

        all_batches: List[EvidenceBatch] = []

        # 3. Process each chat independently to ensure zero cross-chat mixing
        for chat_id in sorted_chat_ids:
            if self.allow_disjoint:
                chat_batches = self._batch_chat_disjoint(
                    context=context,
                    chat_id=chat_id,
                    candidates=candidates_by_chat[chat_id],
                )
            else:
                chat_batches = self._batch_chat(
                    context=context,
                    chat_id=chat_id,
                    candidates=candidates_by_chat[chat_id],
                )
            all_batches.extend(chat_batches)

        self.last_traces = trace_candidates_in_batches(all_batches, candidates)
        return all_batches

    def trace_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        batches: Optional[Sequence[EvidenceBatch]] = None,
    ) -> List[CandidateBatchTrace]:
        """Trace candidate records against provided batches or most recently created batches."""
        if batches is not None:
            self.last_traces = trace_candidates_in_batches(batches, candidates)
        return self.last_traces

    def _batch_chat(
        self,
        context: Any,
        chat_id: str,
        candidates: List[Dict[str, Any]],
    ) -> List[EvidenceBatch]:
        chat_messages = _get_chat_messages(context, chat_id)
        # Ensure chronological message ordering without changing the underlying source dataset
        active_messages = sorted(
            chat_messages,
            key=lambda m: (
                m.get("ts") if isinstance(m.get("ts"), int) and not isinstance(m.get("ts"), bool) else -1,
                m.get("id", ""),
            ),
        )
        msg_index_map = {msg["id"]: i for i, msg in enumerate(active_messages) if "id" in msg}

        # Sort candidates deterministically by message position, timestamp, ID, and category
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                msg_index_map.get(c.get("message_id"), 999999),
                c.get("timestamp") or 0,
                c.get("message_id", ""),
                c.get("category", ""),
            ),
        )

        batches: List[EvidenceBatch] = []
        current_candidates: List[Dict[str, Any]] = []
        current_start: Optional[int] = None
        current_end: Optional[int] = None

        def _finalize(cands: List[Dict[str, Any]], start_i: int, end_i: int):
            if not cands or start_i is None or end_i is None:
                return

            is_oversized = (end_i - start_i + 1) > self.max_messages_per_batch
            if is_oversized and self.oversized_strategy == "trim_to_limit":
                # Ensure the primary candidate message is kept
                cand_idx = msg_index_map.get(cands[0].get("message_id"), start_i)
                half_limit = self.max_messages_per_batch // 2
                trimmed_start = max(start_i, cand_idx - half_limit)
                trimmed_end = trimmed_start + self.max_messages_per_batch - 1
                if trimmed_end > end_i:
                    trimmed_end = end_i
                    trimmed_start = max(start_i, trimmed_end - self.max_messages_per_batch + 1)
                batch_msgs = active_messages[trimmed_start : trimmed_end + 1]
            else:
                batch_msgs = active_messages[start_i : end_i + 1]

            # Preserve chronological candidate message IDs without duplicates
            seen_ids: Set[str] = set()
            cand_msg_ids: List[str] = []
            for c in cands:
                mid = c.get("message_id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    cand_msg_ids.append(mid)

            cats = sorted({c.get("category") for c in cands if c.get("category")})

            # Create independent deep copies of messages and candidates to protect source data
            batches.append(
                EvidenceBatch(
                    chat_id=chat_id,
                    messages=[copy.deepcopy(m) for m in batch_msgs],
                    candidates=[copy.deepcopy(c) for c in cands],
                    candidate_message_ids=list(cand_msg_ids),
                    categories=list(cats),
                    is_oversized=is_oversized,
                )
            )

        for c in sorted_candidates:
            idx = msg_index_map[c["message_id"]]
            cand_start = max(0, idx - self.window_before)
            cand_end = min(len(active_messages) - 1, idx + self.window_after)

            if not current_candidates:
                current_candidates = [c]
                current_start = cand_start
                current_end = cand_end
                continue

            merged_start = min(current_start, cand_start)
            merged_end = max(current_end, cand_end)
            merged_msg_count = merged_end - merged_start + 1

            can_merge = (
                (len(current_candidates) + 1 <= self.max_candidates_per_batch)
                and (cand_start <= current_end)
                and (merged_msg_count <= self.max_messages_per_batch)
            )

            if can_merge:
                current_candidates.append(c)
                current_start = merged_start
                current_end = merged_end
            else:
                _finalize(current_candidates, current_start, current_end)
                current_candidates = [c]
                current_start = cand_start
                current_end = cand_end

        if current_candidates and current_start is not None and current_end is not None:
            _finalize(current_candidates, current_start, current_end)

        return batches

    def _batch_chat_disjoint(
        self,
        context: Any,
        chat_id: str,
        candidates: List[Dict[str, Any]],
    ) -> List[EvidenceBatch]:
        """
        Batch candidates within a chat into disjoint snippets, merging overlapping windows,
        and grouping multiple snippets into batches up to max_candidates_per_batch and max_messages_per_batch.
        """
        chat_messages = _get_chat_messages(context, chat_id)
        active_messages = sorted(
            chat_messages,
            key=lambda m: (
                m.get("ts") if isinstance(m.get("ts"), int) and not isinstance(m.get("ts"), bool) else -1,
                m.get("id", ""),
            ),
        )
        msg_index_map = {msg["id"]: i for i, msg in enumerate(active_messages) if "id" in msg}

        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                msg_index_map.get(c.get("message_id"), 999999),
                c.get("timestamp") or 0,
                c.get("message_id", ""),
                c.get("category", ""),
            ),
        )

        # 1. Group overlapping / adjacent candidate windows into contiguous snippets
        snippets: List[ConversationSnippet] = []
        curr_cands: List[Dict[str, Any]] = []
        curr_start: Optional[int] = None
        curr_end: Optional[int] = None

        def _finalize_snippet(cands: List[Dict[str, Any]], start_i: int, end_i: int, snip_idx: int) -> ConversationSnippet:
            is_oversized = (end_i - start_i + 1) > self.max_messages_per_batch
            if is_oversized and self.oversized_strategy == "trim_to_limit":
                cand_idx = msg_index_map.get(cands[0].get("message_id"), start_i)
                half_limit = self.max_messages_per_batch // 2
                trimmed_start = max(start_i, cand_idx - half_limit)
                trimmed_end = trimmed_start + self.max_messages_per_batch - 1
                if trimmed_end > end_i:
                    trimmed_end = end_i
                    trimmed_start = max(start_i, trimmed_end - self.max_messages_per_batch + 1)
                snip_msgs = active_messages[trimmed_start : trimmed_end + 1]
            else:
                snip_msgs = active_messages[start_i : end_i + 1]

            seen_ids: Set[str] = set()
            cand_msg_ids: List[str] = []
            for c in cands:
                mid = c.get("message_id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    cand_msg_ids.append(mid)

            return ConversationSnippet(
                snippet_id=f"{chat_id}_s{snip_idx}",
                messages=[copy.deepcopy(m) for m in snip_msgs],
                candidates=[copy.deepcopy(c) for c in cands],
                candidate_message_ids=list(cand_msg_ids),
                is_oversized=is_oversized,
            )

        snip_counter = 0
        for c in sorted_candidates:
            idx = msg_index_map[c["message_id"]]
            cand_start = max(0, idx - self.window_before)
            cand_end = min(len(active_messages) - 1, idx + self.window_after)

            if not curr_cands:
                curr_cands = [c]
                curr_start = cand_start
                curr_end = cand_end
                continue

            merged_start = min(curr_start, cand_start)
            merged_end = max(curr_end, cand_end)
            merged_msg_count = merged_end - merged_start + 1

            can_merge = (
                (len(curr_cands) + 1 <= self.max_candidates_per_batch)
                and (cand_start <= curr_end)
                and (merged_msg_count <= self.max_messages_per_batch)
            )

            if can_merge:
                curr_cands.append(c)
                curr_start = merged_start
                curr_end = merged_end
            else:
                snippets.append(_finalize_snippet(curr_cands, curr_start, curr_end, snip_counter))
                snip_counter += 1
                curr_cands = [c]
                curr_start = cand_start
                curr_end = cand_end

        if curr_cands and curr_start is not None and curr_end is not None:
            snippets.append(_finalize_snippet(curr_cands, curr_start, curr_end, snip_counter))

        if not snippets:
            return []

        # 2. Pack disjoint snippets into batches up to batch capacity limits
        batches: List[EvidenceBatch] = []
        cur_snips: List[ConversationSnippet] = []
        cur_cand_count = 0
        cur_msg_count = 0

        def _finalize_batch(snips: List[ConversationSnippet]):
            if not snips:
                return
            batch_msgs: List[Dict[str, Any]] = []
            batch_cands: List[Dict[str, Any]] = []
            batch_cand_ids: List[str] = []
            seen_ids: Set[str] = set()

            for s in snips:
                batch_msgs.extend(s.messages)
                batch_cands.extend(s.candidates)
                for mid in s.candidate_message_ids:
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        batch_cand_ids.append(mid)

            cats = sorted({c.get("category") for c in batch_cands if c.get("category")})
            is_oversized = any(s.is_oversized for s in snips) or (len(batch_msgs) > self.max_messages_per_batch)

            batches.append(
                EvidenceBatch(
                    chat_id=chat_id,
                    messages=batch_msgs,
                    candidates=batch_cands,
                    candidate_message_ids=batch_cand_ids,
                    categories=cats,
                    is_oversized=is_oversized,
                    snippets=snips,
                )
            )

        for snip in snippets:
            if not cur_snips:
                cur_snips = [snip]
                cur_cand_count = snip.candidate_count
                cur_msg_count = snip.message_count
                continue

            can_pack = (
                (cur_cand_count + snip.candidate_count <= self.max_candidates_per_batch)
                and (cur_msg_count + snip.message_count <= self.max_messages_per_batch)
            )

            if can_pack:
                cur_snips.append(snip)
                cur_cand_count += snip.candidate_count
                cur_msg_count += snip.message_count
            else:
                _finalize_batch(cur_snips)
                cur_snips = [snip]
                cur_cand_count = snip.candidate_count
                cur_msg_count = snip.message_count

        if cur_snips:
            _finalize_batch(cur_snips)

        return batches


def build_evidence_batches(
    context: Any,
    candidates: Sequence[Dict[str, Any]],
    window_before: int = 2,
    window_after: int = 4,
    max_candidates_per_batch: int = 8,
    max_messages_per_batch: int = 25,
    oversized_strategy: str = "keep_intact",
) -> List[EvidenceBatch]:
    """
    Convenience function to batch candidate evidence records into chronologically ordered batches.

    Args:
        context: ConversationContext instance or loaded dataset dictionary.
        candidates: Sequence of candidate evidence dictionaries.
        window_before: Messages before each candidate to include in context.
        window_after: Messages after each candidate to include in context.
        max_candidates_per_batch: Maximum candidates allowed in a single batch.
        max_messages_per_batch: Maximum messages allowed in a single batch.
        oversized_strategy: Strategy ('keep_intact' or 'trim_to_limit') for oversized windows.

    Returns:
        List of EvidenceBatch instances.
    """
    batcher = EvidenceBatcher(
        window_before=window_before,
        window_after=window_after,
        max_candidates_per_batch=max_candidates_per_batch,
        max_messages_per_batch=max_messages_per_batch,
        oversized_strategy=oversized_strategy,
    )
    return batcher.batch(context, candidates)


def build_disjoint_evidence_batches(
    context: Any,
    candidates: Sequence[Dict[str, Any]],
    window_before: int = 2,
    window_after: int = 4,
    max_candidates_per_batch: int = 16,
    max_messages_per_batch: int = 50,
    oversized_strategy: str = "keep_intact",
) -> List[EvidenceBatch]:
    """
    Convenience function to batch candidate evidence records into disjoint multi-snippet batches.

    Groups non-overlapping candidate context windows within the same chat into discrete
    ConversationSnippet instances within a single prompt payload, up to max_candidates_per_batch
    and max_messages_per_batch limits.

    Args:
        context: ConversationContext instance or loaded dataset dictionary.
        candidates: Sequence of candidate evidence dictionaries.
        window_before: Messages before each candidate to include in context.
        window_after: Messages after each candidate to include in context.
        max_candidates_per_batch: Maximum candidates allowed in a single batch (default 16).
        max_messages_per_batch: Maximum messages allowed in a single batch (default 50).
        oversized_strategy: Strategy ('keep_intact' or 'trim_to_limit') for oversized windows.

    Returns:
        List of EvidenceBatch instances with multi-snippet packaging.
    """
    batcher = EvidenceBatcher(
        window_before=window_before,
        window_after=window_after,
        max_candidates_per_batch=max_candidates_per_batch,
        max_messages_per_batch=max_messages_per_batch,
        oversized_strategy=oversized_strategy,
        allow_disjoint=True,
    )
    return batcher.batch(context, candidates)
