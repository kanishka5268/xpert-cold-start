"""Tests for Scenario C live LLM optimization: disjoint batching, 3-tier selective routing, and parser safeguards."""

import copy
from typing import Any, Dict, List, Optional
import pytest

from src.interpretation.batcher import (
    ConversationSnippet,
    EvidenceBatch,
    EvidenceBatcher,
    build_disjoint_evidence_batches,
    build_evidence_batches,
)
from src.interpretation.cache import ExecutionBudget
from src.interpretation.interpreter import (
    BatchFailureDiagnostic,
    LLMEvidenceInterpreter,
    ProactiveRateLimiter,
    estimate_prompt_tokens,
)
from src.interpretation.parser import (
    parse_chat_interpretation_response,
)
from src.interpretation.prompts import (
    CHAT_SYSTEM_PROMPT,
    format_chat_user_message,
)
from src.interpretation.records import InterpretedOpenLoop
from src.loops.extractor import (
    classify_candidate_tier,
    extract_open_loops,
    synthesize_fallback_open_loops,
)
from src.loops.models import (
    OpenLoopExtractionConfig,
    OpenLoopExtractionDiagnostics,
)
from src.preprocessing.context_builder import ConversationContext, build_conversation_context


def _create_test_context(
    chat_messages: Dict[str, List[Dict[str, Any]]],
    chats_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    participants_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    timezone_str: str = "Asia/Kolkata",
) -> ConversationContext:
    """Build a valid ConversationContext from per-chat messages and optional metadata."""
    all_msgs: List[Dict[str, Any]] = []
    messages_by_chat: Dict[str, List[Dict[str, Any]]] = {}
    messages_by_id: Dict[str, Dict[str, Any]] = {}
    chats_by_id: Dict[str, Dict[str, Any]] = {}

    chats_meta = chats_metadata or {}
    for cid, msgs in chat_messages.items():
        meta = chats_meta.get(cid, {})
        chats_by_id[cid] = {
            "id": cid,
            "name": meta.get("name", f"Chat {cid}"),
            "is_group": meta.get("is_group", False),
            "participants": meta.get("participants", ["ME", "P01"]),
            "messages": msgs,
        }
        messages_by_chat[cid] = msgs
        for m in msgs:
            all_msgs.append(m)
            messages_by_id[m["id"]] = m

    all_msgs.sort(key=lambda m: m.get("ts", 0) if isinstance(m.get("ts"), (int, float)) else 0)

    participants = participants_metadata or {
        "ME": {"name": "Arjun", "is_silent": False, "sender_names": ["Arjun", "Me"], "direct_chat_names": ["Arjun"], "names": ["Arjun"]},
        "P01": {"name": "Neha", "is_silent": False, "sender_names": ["Neha"], "direct_chat_names": ["Neha"], "names": ["Neha"]},
        "P02": {"name": "Rahul", "is_silent": False, "sender_names": ["Rahul"], "direct_chat_names": ["Rahul"], "names": ["Rahul"]},
    }

    dataset: Dict[str, Any] = {
        "chats": list(chats_by_id.values()),
        "chats_by_id": chats_by_id,
        "messages": all_msgs,
        "messages_by_chat": messages_by_chat,
        "messages_by_id": messages_by_id,
        "timezone": timezone_str,
        "participants": participants,
    }
    return build_conversation_context(dataset)


class MockContext:
    """Minimal mock for ConversationContext."""

    def __init__(self, chats: Dict[str, Dict[str, Any]]):
        self.chats = chats

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        return self.chats.get(chat_id)

    def is_group_chat(self, chat_id: str) -> bool:
        chat = self.chats.get(chat_id, {})
        return bool(chat.get("is_group", False))


@pytest.fixture
def multi_snippet_context():
    """Create a context with non-contiguous messages in a single chat C01."""
    messages = [
        {"id": f"M{i:03d}", "chat_id": "C01", "sender_id": "P01" if i % 2 == 0 else "ME", "ts": 1000 + i * 60, "body": f"Message {i}"}
        for i in range(100)
    ]
    # Set explicit realistic bodies for specific indices
    messages[10]["body"] = "I will send the revised proposal by Friday afternoon"
    messages[50]["body"] = "Please review the budget spreadsheet today"
    messages[90]["body"] = "Can we connect tomorrow for a quick update?"

    return MockContext({"C01": {"id": "C01", "is_group": False, "messages": messages}})


# ==============================================================================
# 1. Disjoint Multi-Snippet Batching Tests
# ==============================================================================

def test_conversation_snippet_dataclass():
    """Verify ConversationSnippet properties and serialization."""
    snip = ConversationSnippet(
        snippet_id="C01_s0",
        messages=[{"id": "M001", "body": "hello"}],
        candidates=[{"message_id": "M001", "category": "commitment"}],
        candidate_message_ids=["M001"],
        is_oversized=False,
    )
    assert snip.message_count == 1
    assert snip.candidate_count == 1
    d = snip.to_dict()
    assert d["snippet_id"] == "C01_s0"
    assert d["candidate_message_ids"] == ["M001"]


def test_evidence_batch_snippets_property():
    """Verify EvidenceBatch automatically wraps messages in a single snippet for legacy batches."""
    batch = EvidenceBatch(
        chat_id="C01",
        messages=[{"id": "M001", "body": "test"}],
        candidates=[{"message_id": "M001", "category": "commitment"}],
        candidate_message_ids=["M001"],
        categories=["commitment"],
    )
    assert batch.snippet_count == 1
    assert len(batch.snippets) == 1
    assert batch.snippets[0].snippet_id == "C01_s0"
    assert batch.to_dict()["snippet_count"] == 1


def test_disjoint_batching_groups_non_contiguous_candidates(multi_snippet_context):
    """Verify that distant candidates in the same chat are packed into disjoint snippets in one batch."""
    candidates = [
        {"message_id": "M010", "chat_id": "C01", "category": "commitment", "timestamp": 1600},
        {"message_id": "M050", "chat_id": "C01", "category": "commitment", "timestamp": 4000},
    ]
    # In contiguous batching, distant candidates produce 2 batches
    contiguous_batches = build_evidence_batches(multi_snippet_context, candidates, window_before=2, window_after=2)
    assert len(contiguous_batches) == 2

    # In disjoint batching, distant candidates are packed into 1 batch with 2 snippets
    disjoint_batches = build_disjoint_evidence_batches(
        multi_snippet_context,
        candidates,
        window_before=2,
        window_after=2,
        max_candidates_per_batch=16,
        max_messages_per_batch=50,
    )
    assert len(disjoint_batches) == 1
    batch = disjoint_batches[0]
    assert batch.chat_id == "C01"
    assert batch.snippet_count == 2
    assert batch.candidate_count == 2
    assert "M010" in batch.snippets[0].candidate_message_ids
    assert "M050" in batch.snippets[1].candidate_message_ids


def test_disjoint_batching_zero_cross_chat_mixing():
    """Verify that candidates from different chats are never merged into the same batch."""
    c1_msgs = [{"id": f"C1_M{i}", "chat_id": "C01", "sender_id": "ME", "ts": 100 + i, "body": f"C1 msg {i}"} for i in range(10)]
    c2_msgs = [{"id": f"C2_M{i}", "chat_id": "C02", "sender_id": "P02", "ts": 200 + i, "body": f"C2 msg {i}"} for i in range(10)]
    ctx = MockContext({"C01": {"id": "C01", "messages": c1_msgs}, "C02": {"id": "C02", "messages": c2_msgs}})

    candidates = [
        {"message_id": "C1_M2", "chat_id": "C01", "category": "commitment", "timestamp": 102},
        {"message_id": "C2_M2", "chat_id": "C02", "category": "question", "timestamp": 202},
    ]
    batches = build_disjoint_evidence_batches(ctx, candidates)
    assert len(batches) == 2
    assert {b.chat_id for b in batches} == {"C01", "C02"}
    for b in batches:
        assert all(m["chat_id"] == b.chat_id for m in b.messages)


def test_disjoint_batching_empty_candidates(multi_snippet_context):
    """Verify that empty candidates list returns empty batches."""
    batches = build_disjoint_evidence_batches(multi_snippet_context, [])
    assert batches == []


# ==============================================================================
# 2. Multi-Snippet Prompt Formatting Tests
# ==============================================================================

def test_prompt_formatting_single_snippet_backward_compatible():
    """Verify that a single-snippet batch does not contain SNIPPET_COUNT header."""
    batch = EvidenceBatch(
        chat_id="C01",
        messages=[{"id": "M001", "sender_id": "ME", "ts": 100, "body": "hello"}],
        candidates=[{"message_id": "M001", "category": "commitment", "timestamp": 100, "original_text": "hello"}],
        candidate_message_ids=["M001"],
        categories=["commitment"],
    )
    formatted = format_chat_user_message(batch)
    assert "SNIPPET_COUNT" not in formatted
    assert '"candidates":' in formatted
    assert '"messages":' in formatted


def test_prompt_formatting_multi_snippet_contains_snippet_structure():
    """Verify that a multi-snippet batch includes SNIPPET_COUNT and 'snippets' in untrusted payload."""
    s1 = ConversationSnippet(
        snippet_id="C01_s0",
        messages=[{"id": "M010", "sender_id": "ME", "ts": 100, "body": "task 1"}],
        candidates=[{"message_id": "M010", "category": "commitment", "timestamp": 100, "original_text": "task 1"}],
        candidate_message_ids=["M010"],
    )
    s2 = ConversationSnippet(
        snippet_id="C01_s1",
        messages=[{"id": "M050", "sender_id": "P01", "ts": 500, "body": "task 2"}],
        candidates=[{"message_id": "M050", "category": "commitment", "timestamp": 500, "original_text": "task 2"}],
        candidate_message_ids=["M050"],
    )
    batch = EvidenceBatch(
        chat_id="C01",
        messages=[{"id": "M010", "sender_id": "ME", "ts": 100, "body": "task 1"}, {"id": "M050", "sender_id": "P01", "ts": 500, "body": "task 2"}],
        candidates=[
            {"message_id": "M010", "category": "commitment", "timestamp": 100, "original_text": "task 1"},
            {"message_id": "M050", "category": "commitment", "timestamp": 500, "original_text": "task 2"},
        ],
        candidate_message_ids=["M010", "M050"],
        categories=["commitment"],
        snippets=[s1, s2],
    )
    formatted = format_chat_user_message(batch)
    assert "SNIPPET_COUNT: 2" in formatted
    assert '"snippets":' in formatted
    assert '"snippet_id": "C01_s0"' in formatted
    assert '"snippet_id": "C01_s1"' in formatted


# ==============================================================================
# 3. Parser Cross-Snippet Safeguard Tests
# ==============================================================================

def test_parser_rejects_cross_snippet_closed_by():
    """Verify parser strips closed_by if it points to a message from a different snippet."""
    s1 = ConversationSnippet(
        snippet_id="C01_s0",
        messages=[
            {"id": "M010", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "will do"},
            {"id": "M011", "chat_id": "C01", "sender_id": "P01", "ts": 110, "body": "ok"},
        ],
        candidates=[{"message_id": "M010", "category": "commitment", "sender_id": "ME"}],
        candidate_message_ids=["M010"],
    )
    s2 = ConversationSnippet(
        snippet_id="C01_s1",
        messages=[
            {"id": "M050", "chat_id": "C01", "sender_id": "P01", "ts": 500, "body": "unrelated done"},
        ],
        candidates=[{"message_id": "M050", "category": "commitment", "sender_id": "P01"}],
        candidate_message_ids=["M050"],
    )
    batch = EvidenceBatch(
        chat_id="C01",
        messages=s1.messages + s2.messages,
        candidates=s1.candidates + s2.candidates,
        candidate_message_ids=["M010", "M050"],
        categories=["commitment"],
        snippets=[s1, s2],
    )

    # LLM incorrectly points closed_by for M010 to M050 (from snippet 2)
    raw_response = '''{
        "open_loops": [
            {
                "source_msg": "M010",
                "chat_id": "C01",
                "type": "commitment",
                "is_genuine": true,
                "title": "Send report",
                "owner": "ME",
                "target_person": "P01",
                "status": "done",
                "closed_by": "M050"
            }
        ],
        "goals": []
    }'''
    loops, goals = parse_chat_interpretation_response(raw_response, batch)
    assert len(loops) == 1
    assert loops[0].source_msg == "M010"
    # closed_by must be stripped to None due to cross-snippet reference!
    assert loops[0].closed_by is None
    assert loops[0].validation_flags is not None
    assert any("cross_snippet_closed_by:M050" in f for f in loops[0].validation_flags)


def test_parser_accepts_same_snippet_closed_by():
    """Verify parser preserves closed_by when it belongs to the same snippet."""
    s1 = ConversationSnippet(
        snippet_id="C01_s0",
        messages=[
            {"id": "M010", "chat_id": "C01", "sender_id": "ME", "ts": 100, "body": "will do"},
            {"id": "M011", "chat_id": "C01", "sender_id": "ME", "ts": 110, "body": "done"},
        ],
        candidates=[{"message_id": "M010", "category": "commitment", "sender_id": "ME"}],
        candidate_message_ids=["M010"],
    )
    batch = EvidenceBatch(
        chat_id="C01",
        messages=s1.messages,
        candidates=s1.candidates,
        candidate_message_ids=["M010"],
        categories=["commitment"],
        snippets=[s1],
    )
    raw_response = '''{
        "open_loops": [
            {
                "source_msg": "M010",
                "chat_id": "C01",
                "type": "commitment",
                "is_genuine": true,
                "title": "Send report",
                "owner": "ME",
                "target_person": null,
                "status": "done",
                "closed_by": "M011"
            }
        ],
        "goals": []
    }'''
    loops, goals = parse_chat_interpretation_response(raw_response, batch)
    assert len(loops) == 1
    assert loops[0].closed_by == "M011"


# ==============================================================================
# 4. Three-Tier Selective Routing Tests
# ==============================================================================

def test_tier2_pre_dispatch_rejects_pleasantries():
    """Verify that pleasantries, dismissals, and non-task chatter are routed to Tier 2."""
    cand = {"message_id": "M001", "chat_id": "C01", "category": "commitment"}
    pleasantry_msg = {"id": "M001", "chat_id": "C01", "sender_id": "P01", "body": "Thank you for checking in"}
    tier, reason = classify_candidate_tier(cand, pleasantry_msg)
    assert tier == 2
    assert "safe_pre_dispatch_reject" in reason


def test_tier1_deterministic_fastpath_1to1():
    """Verify direct 1:1 first-person commitments with high confidence route to Tier 1."""
    cand = {"message_id": "M002", "chat_id": "C01", "category": "commitment"}
    commitment_msg = {"id": "M002", "chat_id": "C01", "sender_id": "ME", "body": "I will send the invoice by Friday afternoon"}
    ctx = MockContext({"C01": {"id": "C01", "is_group": False, "messages": [commitment_msg]}})
    tier, reason = classify_candidate_tier(cand, commitment_msg, context=ctx)
    assert tier == 1
    assert "fastpath" in reason


def test_tier3_group_chat_ambiguity_dispatch():
    """Verify candidates in group chats are routed to Tier 3 for context disambiguation."""
    cand = {"message_id": "M003", "chat_id": "C02", "category": "commitment"}
    group_msg = {"id": "M003", "chat_id": "C02", "sender_id": "ME", "body": "I will take care of the booking"}
    ctx = MockContext({"C02": {"id": "C02", "is_group": True, "messages": [group_msg]}})
    tier, reason = classify_candidate_tier(cand, group_msg, context=ctx)
    assert tier == 3
    assert "group_chat" in reason


def test_candidate_conservation_invariant():
    """Verify candidate conservation holds: |C| = |T1| + |T2| + |T3| without overlap."""
    msgs = [
        {"id": "M1", "chat_id": "C1", "sender_id": "ME", "ts": 100, "body": "I will send the report tomorrow"},
        {"id": "M2", "chat_id": "C1", "sender_id": "P1", "ts": 200, "body": "Thank you very much"},
        {"id": "M3", "chat_id": "C2", "sender_id": "P2", "ts": 300, "body": "Can you review the deck today?"},
    ]
    ctx = _create_test_context(
        {"C1": msgs[:2], "C2": msgs[2:]},
        chats_metadata={"C1": {"is_group": False}, "C2": {"is_group": True}},
    )
    cands = [
        {"message_id": "M1", "chat_id": "C1", "category": "commitment"},
        {"message_id": "M2", "chat_id": "C1", "category": "commitment"},
        {"message_id": "M3", "chat_id": "C2", "category": "question"},
    ]
    t1, t2, t3 = [], [], []
    for c in cands:
        m = next(m for m in msgs if m["id"] == c["message_id"])
        tier, reason = classify_candidate_tier(c, m, context=ctx)
        if tier == 1:
            t1.append(c)
        elif tier == 2:
            t2.append(c)
        else:
            t3.append(c)

    assert len(t1) + len(t2) + len(t3) == len(cands)
    assert {c["message_id"] for c in t1} == {"M1"}
    assert {c["message_id"] for c in t2} == {"M2"}
    assert {c["message_id"] for c in t3} == {"M3"}


# ==============================================================================
# 5. Pipeline Execution Mode Tests
# ==============================================================================

class MockScenarioCInterpreter:
    """Mock interpreter supporting Scenario C offline and live routing modes."""

    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self.allow_offline = True
        self.batch_calls: List[Any] = []

    def interpret_chat_batch(self, batch: Any, context: Any = None, budget: Any = None):
        self.batch_calls.append(batch)
        if self.offline_mode:
            return [], [], BatchFailureDiagnostic(
                phase="chat",
                batch_id=batch.chat_id,
                attempt_count=1,
                error_type="OfflineModeNoCache",
                error_message="Offline mode active",
                timestamp=100.0,
            )
        loops = [
            InterpretedOpenLoop(
                source_msg=cid,
                chat_id=batch.chat_id,
                type="commitment",
                is_genuine=True,
                title=f"Action on {cid}",
                owner="ME",
                target_person=None,
                status="open",
                closed_by=None,
                due_date=None,
                suggested_action="remind",
            )
            for cid in batch.candidate_message_ids
        ]
        return loops, [], None


def test_live_routing_bypasses_tier2_and_dispatches_tier3():
    """Verify that in live routing mode, Tier 2 candidates are bypassed and Tier 3 batches are dispatched."""
    messages = [
        {"id": "M1", "chat_id": "C1", "sender_id": "ME", "ts": 100, "body": "I will send the contract by Friday"},
        {"id": "M2", "chat_id": "C1", "sender_id": "P1", "ts": 200, "body": "Thank you for checking in"},
        {"id": "M3", "chat_id": "C2", "sender_id": "P2", "ts": 300, "body": "Can you check the invoice status?"},
    ]
    context = _create_test_context(
        {"C1": messages[:2], "C2": messages[2:]},
        chats_metadata={"C1": {"is_group": False}, "C2": {"is_group": True}},
    )

    interpreter = MockScenarioCInterpreter(offline_mode=False)
    config = OpenLoopExtractionConfig(enable_selective_routing=True, disjoint_batching=True)
    diag = OpenLoopExtractionDiagnostics()

    result = extract_open_loops(context=context, interpreter=interpreter, config=config, diagnostics=diag)

    # Candidate accounting
    assert diag.tier1_count + diag.tier2_count + diag.tier3_count == diag.candidates_after_group_filter
    assert "M2" in diag.tier2_candidates
    # Only Tier 3 batch was dispatched to interpreter
    assert len(interpreter.batch_calls) == diag.tier3_batches_created
    assert all(b.chat_id == "C2" for b in interpreter.batch_calls)
    assert result.loop_count > 0


def test_offline_mode_preserves_deterministic_fallback():
    """Verify offline mode executes fallback without making live interpreter calls."""
    messages = [
        {"id": "M1", "chat_id": "C1", "sender_id": "ME", "ts": 100, "body": "I will send the contract by Friday"},
        {"id": "M2", "chat_id": "C1", "sender_id": "P1", "ts": 200, "body": "Thank you for checking in"},
    ]
    context = _create_test_context(
        {"C1": messages},
        chats_metadata={"C1": {"is_group": False}},
    )

    interpreter = MockScenarioCInterpreter(offline_mode=True)
    config = OpenLoopExtractionConfig()
    diag = OpenLoopExtractionDiagnostics()

    result = extract_open_loops(context=context, interpreter=interpreter, config=config, diagnostics=diag)

    # In offline mode, interpreter reports offline unavailable and fallback executes
    assert diag.batches_offline_unavailable == 1
    assert result.loop_count >= 1
    assert result.open_loops[0]["source_msg"] == "M1"



# ==============================================================================
# 6. Snippet-Independent Interpretation Tests (P1 Correction)
# ==============================================================================

def test_chat_system_prompt_contains_snippet_independence_instructions():
    """Verify CHAT_SYSTEM_PROMPT contains mandatory snippet-independence rules."""
    assert "11. SNIPPET-INDEPENDENT INTERPRETATION:" in CHAT_SYSTEM_PROMPT
    assert "ONLY its own associated snippet" in CHAT_SYSTEM_PROMPT
    assert "Never use messages from another snippet" in CHAT_SYSTEM_PROMPT
    assert "cannot be marked done because of a message appearing in another snippet" in CHAT_SYSTEM_PROMPT
    assert "Do not transfer contextual assumptions, identities, commitments, or deadlines between snippets" in CHAT_SYSTEM_PROMPT
    assert "If evidence within the candidate's own snippet is insufficient" in CHAT_SYSTEM_PROMPT


def test_snippet_association_and_independent_representation():
    """Verify that multiple snippets retain their independent candidate mappings in prompt formatting."""
    s1 = ConversationSnippet(
        snippet_id="C01_s0",
        messages=[
            {"id": "M01", "sender_id": "ME", "ts": 100, "body": "I will review the design tomorrow"},
            {"id": "M02", "sender_id": "P01", "ts": 150, "body": "Great, thanks"},
        ],
        candidates=[{"message_id": "M01", "category": "commitment", "timestamp": 100, "original_text": "I will review the design tomorrow"}],
        candidate_message_ids=["M01"],
    )
    s2 = ConversationSnippet(
        snippet_id="C01_s1",
        messages=[
            {"id": "M50", "sender_id": "P01", "ts": 500, "body": "Did you finish the review?"},
            {"id": "M51", "sender_id": "ME", "ts": 550, "body": "Yes done"},
        ],
        candidates=[{"message_id": "M50", "category": "question", "timestamp": 500, "original_text": "Did you finish the review?"}],
        candidate_message_ids=["M50"],
    )
    batch = EvidenceBatch(
        chat_id="C01",
        messages=s1.messages + s2.messages,
        candidates=s1.candidates + s2.candidates,
        candidate_message_ids=["M01", "M50"],
        categories=["commitment", "question"],
        snippets=[s1, s2],
    )
    prompt_text = format_chat_user_message(batch)
    assert "SNIPPET_COUNT: 2" in prompt_text
    assert '"snippet_id": "C01_s0"' in prompt_text
    assert '"snippet_id": "C01_s1"' in prompt_text
    assert '"snippet_id": "C01_s0"' in prompt_text


# ==============================================================================
# 7. Proactive Rate Limiting (RPM / TPM) Tests (P1 Correction)
# ==============================================================================

class MockClock:
    """Deterministic monotonic clock for rate-limiter unit testing without wall-clock sleep."""
    def __init__(self, start: float = 1000.0):
        self.current_time = start

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


def test_rate_limiter_rpm_enforcement():
    """Verify 30 RPM limit enforces waiting after 30 requests within 60s."""
    clock = MockClock(1000.0)
    sleeps: List[float] = []
    limiter = ProactiveRateLimiter(max_rpm=30, max_tpm=60000, window_seconds=60.0, clock=clock, sleeper=sleeps.append)

    # First 30 requests should succeed immediately without sleep
    for _ in range(30):
        assert limiter.acquire(estimated_tokens=10) is True
        clock.advance(1.0)  # 1s per request, total elapsed 30s

    assert len(sleeps) == 0

    # 31st request: window has 30 requests. Oldest request was at t=1000.0.
    # Current time = 1030.0. Oldest expires at 1060.0.
    # Must wait 1060.0 - 1030.0 = 30.0s.
    wait_time = limiter.calculate_wait_time(estimated_tokens=10)
    assert wait_time is not None and abs(wait_time - 30.0) < 1e-3

    # Acquire will trigger sleeper(30.0)
    def advancing_sleeper(s: float):
        sleeps.append(s)
        clock.advance(s)

    limiter.sleeper = advancing_sleeper
    assert limiter.acquire(estimated_tokens=10) is True
    assert len(sleeps) == 1
    assert abs(sleeps[0] - 30.0) < 1e-3


def test_rate_limiter_tpm_enforcement():
    """Verify TPM limit enforces waiting when token consumption reaches threshold."""
    clock = MockClock(1000.0)
    sleeps: List[float] = []
    limiter = ProactiveRateLimiter(max_rpm=100, max_tpm=6000, window_seconds=60.0, clock=clock, sleeper=sleeps.append)

    # Dispatch 3 requests of 1,800 tokens each (total 5,400 tokens)
    for _ in range(3):
        assert limiter.acquire(estimated_tokens=1800) is True
        clock.advance(5.0)  # requests at t=1000, 1005, 1010. Current t=1015.

    # 4th request needing 1,000 tokens would push total to 6,400 (> 6,000).
    # To free 400 tokens, oldest request (t=1000, 1800 tok) must expire at t=1060.
    # Required wait = 1060 - 1015 = 45s.
    wait_time = limiter.calculate_wait_time(estimated_tokens=1000)
    assert wait_time is not None and abs(wait_time - 45.0) < 1e-3


def test_rate_limiter_rolling_window_expiration():
    """Verify that requests age out and restore quota after window_seconds."""
    clock = MockClock(1000.0)
    limiter = ProactiveRateLimiter(max_rpm=2, max_tpm=1000, window_seconds=60.0, clock=clock)

    assert limiter.acquire(estimated_tokens=200) is True  # t=1000
    clock.advance(10.0)
    assert limiter.acquire(estimated_tokens=200) is True  # t=1010
    clock.advance(10.0)  # current t=1020

    # Capacity full (2 requests in window)
    assert limiter.calculate_wait_time(200) > 0.0

    # Advance clock past t=1060 (first request expires)
    clock.advance(41.0)  # current t=1061 (> 1060)
    assert limiter.calculate_wait_time(200) == 0.0


def test_rate_limiter_individual_request_exceeds_token_budget():
    """Verify requests that individually exceed max_tpm are safely rejected."""
    clock = MockClock(1000.0)
    limiter = ProactiveRateLimiter(max_rpm=30, max_tpm=6000, window_seconds=60.0, clock=clock)

    # 7,000 tokens exceeds 6,000 max_tpm
    wait_time = limiter.calculate_wait_time(7000)
    assert wait_time is None
    assert limiter.acquire(7000) is False


def test_rate_limiter_provider_header_overrides():
    """Verify provider response headers take precedence over local quota estimates."""
    clock = MockClock(1000.0)
    limiter = ProactiveRateLimiter(max_rpm=30, max_tpm=6000, window_seconds=60.0, clock=clock)

    # Local window is empty, but provider reports 0 remaining requests reset in 5s
    headers = {
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "5s",
    }
    limiter.update_from_headers(headers)

    wait = limiter.calculate_wait_time(100)
    assert wait is not None and abs(wait - 5.0) < 1e-3

    # Advance clock by 5s -> provider limit resets
    clock.advance(5.01)
    assert limiter.calculate_wait_time(100) == 0.0


def test_rate_limiter_retry_attempts_counted_towards_rpm():
    """Verify retry attempts count as outbound requests against rate limiter."""
    clock = MockClock(1000.0)
    limiter = ProactiveRateLimiter(
        max_rpm=30,
        max_tpm=30000,
        window_seconds=60.0,
        clock=clock,
        sleeper=clock.advance,
    )

    attempts = 0
    def failing_chat_fn(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        import urllib.error
        raise urllib.error.HTTPError("http://test", 500, "Internal Server Error", {}, None)

    interpreter = LLMEvidenceInterpreter(
        chat_fn=failing_chat_fn,
        max_retries=2,
        retry_delay=0.001,
        rate_limiter=limiter,
        cache_enabled=False,
    )
    batch = EvidenceBatch(
        chat_id="C01",
        messages=[{"id": "M01", "sender_id": "ME", "ts": 100, "body": "test"}],
        candidates=[{"message_id": "M01", "category": "commitment", "timestamp": 100, "original_text": "test"}],
        candidate_message_ids=["M01"],
        categories=["commitment"],
    )

    loops, goals, diag = interpreter.interpret_chat_batch(batch)
    assert diag is not None
    # Total attempts = 1 initial + 2 retries = 3
    assert attempts == 3
    # The limiter history must have exactly 3 entries recorded
    assert len(limiter._history) == 3


def test_rate_limiter_shared_accounting_chat_and_participant():
    """Verify chat and participant interpretation share the same rate limiter."""
    limiter = ProactiveRateLimiter(max_rpm=30, max_tpm=6000)
    interpreter = LLMEvidenceInterpreter(
        chat_fn=lambda *args, **kwargs: '{"open_loops": [], "goals": [], "people": [], "organisations": []}',
        rate_limiter=limiter,
        cache_enabled=False,
    )

    chat_batch = EvidenceBatch(
        chat_id="C01",
        messages=[{"id": "M01", "sender_id": "ME", "ts": 100, "body": "test"}],
        candidates=[{"message_id": "M01", "category": "commitment", "timestamp": 100, "original_text": "test"}],
        candidate_message_ids=["M01"],
        categories=["commitment"],
    )
    interpreter.interpret_chat_batch(chat_batch)
    assert len(limiter._history) == 1

    dossiers = [{"person_id": "P01", "names": ["Neha"], "chats": [], "messages": []}]
    interpreter.interpret_participant_batch(dossiers)
    assert len(limiter._history) == 2


def test_rate_limiter_budget_exhaustion_during_waiting():
    """Verify execution budget exhaustion while waiting for rate capacity triggers diagnostic failure."""
    clock = MockClock(1000.0)
    limiter = ProactiveRateLimiter(max_rpm=1, max_tpm=1000, window_seconds=60.0, clock=clock)

    # Use up the single slot
    assert limiter.acquire(100) is True

    # Budget has only 10s remaining (reserve 5s, min call 1s -> max pacing allowed = 4s)
    budget = ExecutionBudget(total_seconds=10.0, reserve_seconds=5.0, call_timeout=10.0, min_call_timeout=1.0)

    # Acquire needing 60s wait will exceed 4s max pacing
    assert limiter.acquire(100, budget=budget) is False


def test_rate_limiter_offline_and_cache_zero_rate_consumption():
    """Verify offline mode and cache hits do not consume rate limit quota."""
    limiter = ProactiveRateLimiter(max_rpm=30, max_tpm=6000)
    interpreter = LLMEvidenceInterpreter(
        chat_fn=lambda *args, **kwargs: '{"open_loops": [], "goals": []}',
        rate_limiter=limiter,
        offline_mode=True,
        cache_enabled=False,
    )
    batch = EvidenceBatch(
        chat_id="C01",
        messages=[{"id": "M01", "sender_id": "ME", "ts": 100, "body": "test"}],
        candidates=[{"message_id": "M01", "category": "commitment", "timestamp": 100, "original_text": "test"}],
        candidate_message_ids=["M01"],
        categories=["commitment"],
    )
    interpreter.interpret_chat_batch(batch)
    assert len(limiter._history) == 0


# ==============================================================================
# 8. Stage A / Stage B / Tier Conservation Diagnostics Tests (P2 Correction)
# ==============================================================================

def test_stage_b_filtering_and_candidate_conservation_fixture():
    """
    Verify Stage A, Stage B, and tier diagnostic accounting:
    - Stage A count equals raw candidate count.
    - Stage B count equals filtered candidate count.
    - Tier counts sum exactly to Stage B count: |T1| + |T2| + |T3| == candidates_after_group_filter.
    - The old incorrect equality (|T1| + |T2| + |T3| == candidates_received) is NOT assumed.
    """
    # Build dataset where:
    # C1 (1:1 chat):
    # - M1 (ME commitment) -> Stage A passes, Stage B passes -> Tier 1
    # - M2 (Social thank you) -> Stage A passes, Stage B passes -> Tier 2
    # - M3 (Question) -> Stage A passes, Stage B passes -> Tier 3
    # C2 (Group chat):
    # - M4 (General group chatter / broadcast from P2 without ME assignment) -> Stage A matches, Stage B filters out
    messages_c1 = [
        {"id": "M1", "chat_id": "C1", "sender_id": "ME", "ts": 100, "body": "I will send the report tomorrow morning"},
        {"id": "M2", "chat_id": "C1", "sender_id": "P1", "ts": 200, "body": "Thank you for checking in"},
        {"id": "M3", "chat_id": "C1", "sender_id": "P1", "ts": 300, "body": "Can you check the balance?"},
    ]
    messages_c2 = [
        {"id": "M4", "chat_id": "C2", "sender_id": "P2", "ts": 400, "body": "Why is it raining so much today?"},
    ]
    context = _create_test_context(
        {"C1": messages_c1, "C2": messages_c2},
        chats_metadata={"C1": {"is_group": False}, "C2": {"is_group": True, "participants": ["ME", "P1", "P2"]}},
    )

    interpreter = MockScenarioCInterpreter(offline_mode=False)
    config = OpenLoopExtractionConfig(enable_selective_routing=True, disjoint_batching=True)
    diag = OpenLoopExtractionDiagnostics()

    result = extract_open_loops(context=context, interpreter=interpreter, config=config, diagnostics=diag)

    # Stage A caught raw candidates from both C1 and C2
    assert diag.candidates_received >= 4
    # Stage B filtered out M4 from group chat C2
    assert diag.candidates_after_group_filter < diag.candidates_received

    # Mandatory invariant: Tier counts sum exactly to candidates_after_group_filter
    tier_sum = diag.tier1_count + diag.tier2_count + diag.tier3_count
    assert tier_sum == diag.candidates_after_group_filter
    # Proves old incorrect assumption does not hold
    assert tier_sum != diag.candidates_received


def test_stage_b_zero_candidates_and_all_filtered_edge_cases():
    """Verify diagnostic accounting for zero candidates and when Stage B filters 100% of candidates."""
    # 1. Zero candidate case
    diag_empty = OpenLoopExtractionDiagnostics()
    ctx_empty = _create_test_context({"C1": []})
    result_empty = extract_open_loops(context=ctx_empty, interpreter=None, diagnostics=diag_empty)
    assert diag_empty.candidates_received == 0
    assert diag_empty.candidates_after_group_filter == 0
    assert diag_empty.tier1_count + diag_empty.tier2_count + diag_empty.tier3_count == 0

    # 2. All-filtered case: only irrelevant group chat messages
    messages_group = [
        {"id": "MG1", "chat_id": "CG", "sender_id": "P2", "ts": 100, "body": "Why is it so hot outside?"},
    ]
    ctx_all_filtered = _create_test_context(
        {"CG": messages_group},
        chats_metadata={"CG": {"is_group": True, "participants": ["ME", "P1", "P2"]}},
    )
    diag_filtered = OpenLoopExtractionDiagnostics()
    result_filtered = extract_open_loops(context=ctx_all_filtered, interpreter=None, diagnostics=diag_filtered)
    # Stage A caught MG1, but Stage B filtered it out
    assert diag_filtered.candidates_received >= 1
    assert diag_filtered.candidates_after_group_filter == 0
    assert diag_filtered.tier1_count + diag_filtered.tier2_count + diag_filtered.tier3_count == 0
    assert diag_filtered.tier1_count + diag_filtered.tier2_count + diag_filtered.tier3_count == diag_filtered.candidates_after_group_filter
