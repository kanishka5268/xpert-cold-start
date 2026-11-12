"""Unit and integration tests for the LLM Evidence Interpreter module."""

import email.utils
import io
import json
import os
import socket
import time
import urllib.error
import pytest

from src.ingestion.loader import load_dataset
from src.interpretation.batcher import EvidenceBatch, build_evidence_batches
from src.interpretation.cache import ExecutionBudget, InterpretationCache
from src.interpretation.interpreter import (
    BatchFailureDiagnostic,
    InterpretationDiagnostics,
    InterpretationResult,
    LLMConfigurationError,
    LLMEvidenceInterpreter,
    _extract_retry_delay,
    build_participant_dossiers,
)
from src.interpretation.parser import (
    ResponseParsingError,
    ResponseStructureError,
)
from src.interpretation.prompts import build_chat_interpretation_prompt
from src.interpretation.records import (
    InterpretedGoal,
    InterpretedOpenLoop,
    InterpretedOrganisation,
    InterpretedPerson,
)
from src.preprocessing.context_builder import build_conversation_context


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_batch():
    """A realistic EvidenceBatch for chat-level testing."""
    messages = [
        {"id": "M015", "chat_id": "C04", "sender_id": "P04", "sender_name": "Neha", "ts": 1787534800, "body": "Are we taking the Zest client?"},
        {"id": "M016", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535000, "body": "Pitch deck for Halcyon is due Friday. Neha let me think about Zest"},
        {"id": "M017", "chat_id": "C04", "sender_id": "P04", "sender_name": "Neha", "ts": 1787535200, "body": "Okay, let me know by tomorrow"},
        {"id": "M018", "chat_id": "C04", "sender_id": "ME", "sender_name": "ME", "ts": 1787535400, "body": "Will do. Sent the invoice to Brightside"},
    ]
    candidates = [
        {
            "message_id": "M016",
            "chat_id": "C04",
            "sender_id": "ME",
            "sender_name": "ME",
            "timestamp": 1787535000,
            "category": "question",
            "matched_signals": ["let me think"],
            "original_text": "Pitch deck for Halcyon is due Friday. Neha let me think about Zest",
        },
        {
            "message_id": "M018",
            "chat_id": "C04",
            "sender_id": "ME",
            "sender_name": "ME",
            "timestamp": 1787535400,
            "category": "commitment",
            "matched_signals": ["will do"],
            "original_text": "Will do. Sent the invoice to Brightside",
        },
    ]
    return EvidenceBatch(
        chat_id="C04",
        messages=messages,
        candidates=candidates,
        candidate_message_ids=["M016", "M018"],
        categories=["commitment", "question"],
    )


@pytest.fixture
def sample_context(tmp_path):
    """Create a realistic ConversationContext including silent and active participants."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    chats = [
        {"chat_id": "C01", "name": "Direct Neha", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Driver Direct", "is_group": False, "participants": ["ME", "P02"]},
        {"chat_id": "C03", "name": "Family Group", "is_group": True, "participants": ["ME", "P01", "P03"]},
    ]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "ME", "sender_name": "ME", "ts": 1787530000, "kind": "text", "body": "Neha can you send the report?"},
        {"id": "M002", "chat_id": "C01", "sender_id": "P01", "sender_name": "Neha Brightside", "ts": 1787530100, "kind": "text", "body": "Yes, Brightside report will be ready by 5pm."},
        {"id": "M003", "chat_id": "C02", "sender_id": "ME", "sender_name": "ME", "ts": 1787531000, "kind": "text", "body": "Raju please pick up the kids from Das School at 2pm."},
        {"id": "M004", "chat_id": "C03", "sender_id": "ME", "sender_name": "ME", "ts": 1787532000, "kind": "text", "body": "Mummy please take your medicine on time."},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")

    return build_conversation_context(load_dataset(tmp_path))


# ==============================================================================
# 1. Configuration & Auth Failure Tests
# ==============================================================================

def test_init_with_custom_chat_fn_does_not_require_env_vars(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    interpreter = LLMEvidenceInterpreter(chat_fn=lambda **kwargs: "{}")
    assert interpreter.is_custom_client is True
    assert interpreter.max_retries == 2


def test_init_with_default_chat_fn_raises_when_env_vars_missing(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(LLMConfigurationError, match="Missing required environment variable: LLM_BASE_URL"):
        LLMEvidenceInterpreter()

    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    with pytest.raises(LLMConfigurationError, match="Missing required environment variable: LLM_API_KEY"):
        LLMEvidenceInterpreter()


def test_init_with_explicit_model_does_not_require_llm_model_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    interpreter = LLMEvidenceInterpreter(model="llama-3.3-70b-versatile")
    assert interpreter.model == "llama-3.3-70b-versatile"


def test_auth_failure_401_raises_configuration_error_immediately(sample_batch):
    call_count = 0

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    with pytest.raises(LLMConfigurationError, match="LLM API authentication failed \\(HTTP 401\\)"):
        interpreter.interpret_chat_batch(sample_batch)

    # Must fail immediately without retrying
    assert call_count == 1


def test_auth_failure_403_raises_configuration_error_immediately(sample_batch):
    call_count = 0

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    with pytest.raises(LLMConfigurationError, match="LLM API authentication failed \\(HTTP 403\\)"):
        interpreter.interpret_chat_batch(sample_batch)

    assert call_count == 1


def test_sanitization_api_key_never_exposed_in_diagnostics(sample_batch, monkeypatch):
    secret_key = "test_secret_value_for_sanitization_only"
    monkeypatch.setenv("LLM_API_KEY", secret_key)

    def mock_chat(**kwargs):
        raise RuntimeError(f"Failed with credential {secret_key} in authorization header")

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=0)
    _, _, diag = interpreter.interpret_chat_batch(sample_batch)

    assert diag is not None
    assert secret_key not in diag.error_message
    assert diag.error_message == "RuntimeError"
    assert diag.error_type == "RuntimeError"


def test_sanitization_httperror_never_exposes_secrets_in_url_or_reason(sample_batch):
    fake_key = "fake_key_999_secret"
    fake_token = "token_xyz_888"
    fake_user = "user_admin_777"
    fake_query = "secret_query_val"
    sensitive_url = f"https://{fake_user}:{fake_key}@api.groq.com/v1/chat/path_secret?param={fake_query}&token={fake_token}"
    sensitive_reason = f"Internal server error containing secret {fake_key} and token {fake_token}"

    def mock_chat(**kwargs):
        raise urllib.error.HTTPError(
            url=sensitive_url,
            code=429,
            msg=sensitive_reason,
            hdrs={"Authorization": f"Bearer {fake_token}"},
            fp=None,
        )

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=0)
    _, _, diag = interpreter.interpret_chat_batch(sample_batch)

    assert diag is not None
    assert diag.error_type == "HTTPError:429"
    assert diag.error_message == "HTTP error 429"
    assert fake_key not in diag.error_message
    assert fake_token not in diag.error_message
    assert fake_user not in diag.error_message
    assert fake_query not in diag.error_message
    assert "path_secret" not in diag.error_message
    assert "api.groq.com" not in diag.error_message
    assert "Authorization" not in diag.error_message


def test_sanitization_urlerror_never_exposes_secrets_in_reason(sample_batch):
    fake_key = "fake_key_999_secret"
    fake_token = "token_xyz_888"
    sensitive_reason = f"Connection failed to https://admin:{fake_key}@api.groq.com/v1/chat?token={fake_token}"

    def mock_chat(**kwargs):
        raise urllib.error.URLError(reason=sensitive_reason)

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=0)
    _, _, diag = interpreter.interpret_chat_batch(sample_batch)

    assert diag is not None
    assert diag.error_type == "URLError"
    assert diag.error_message == "Network connection error"
    assert fake_key not in diag.error_message
    assert fake_token not in diag.error_message
    assert "admin" not in diag.error_message
    assert "api.groq.com" not in diag.error_message


# ==============================================================================
# 2. Dossier Construction Tests
# ==============================================================================

def test_build_participant_dossiers_excludes_me(sample_context):
    batches = build_participant_dossiers(sample_context)
    all_dossier_pids = [
        d["person_id"]
        for batch, _ in batches
        for d in batch
    ]
    assert "ME" not in all_dossier_pids
    assert set(all_dossier_pids) == {"P01", "P02", "P03"}


def test_build_participant_dossiers_combines_sender_and_direct_chat_names(sample_context):
    batches = build_participant_dossiers(sample_context)
    dossiers_by_pid = {
        d["person_id"]: d
        for batch, _ in batches
        for d in batch
    }
    # P01 is in Direct Neha and sent with name "Neha Brightside"
    p01_names = dossiers_by_pid["P01"]["names"]
    assert "Neha Brightside" in p01_names or "Direct Neha" in p01_names


def test_build_participant_dossiers_preserves_silent_participants_with_messages(sample_context):
    # P02 is the driver (only ME sent messages to P02 in C02, P02 is silent)
    # P03 is Mummy (only ME sent messages to P03 in C03, P03 is silent)
    batches = build_participant_dossiers(sample_context)
    dossiers_by_pid = {
        d["person_id"]: d
        for batch, _ in batches
        for d in batch
    }

    p02_dossier = dossiers_by_pid["P02"]
    assert len(p02_dossier["messages"]) > 0
    assert any("Raju please pick up the kids" in m["body"] for m in p02_dossier["messages"])

    p03_dossier = dossiers_by_pid["P03"]
    assert len(p03_dossier["messages"]) > 0
    assert any("Mummy please take your medicine" in m["body"] for m in p03_dossier["messages"])


def test_build_participant_dossiers_attaches_chat_metadata(sample_context):
    batches = build_participant_dossiers(sample_context)
    for batch, chat_metadata in batches:
        assert isinstance(chat_metadata, dict)
        for cid, meta in chat_metadata.items():
            assert "name" in meta
            assert "is_group" in meta
            assert "participants" in meta


def test_build_participant_dossiers_chunking(sample_context):
    # 3 participants (P01, P02, P03). With max_participants_per_batch=2 -> 2 batches
    batches = build_participant_dossiers(sample_context, max_participants_per_batch=2)
    assert len(batches) == 2
    assert len(batches[0][0]) == 2
    assert len(batches[1][0]) == 1


def test_build_participant_dossiers_invalid_context_raises_type_error():
    with pytest.raises(TypeError, match="Expected ConversationContext"):
        build_participant_dossiers({"chats": {}})


# ==============================================================================
# 3. Retry Rule Tests
# ==============================================================================

def test_permanent_http_400_does_not_retry_records_diagnostic(sample_batch):
    call_count = 0

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=None,
        )

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert loops == []
    assert goals == []
    assert call_count == 1  # No retry on permanent 4xx
    assert diag is not None
    assert diag.error_type == "HTTPError:400"
    assert diag.attempt_count == 1


def test_permanent_http_404_does_not_retry_records_diagnostic(sample_batch):
    call_count = 0

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert call_count == 1
    assert diag is not None
    assert diag.error_type == "HTTPError:404"


def test_retry_http_429_succeeds_on_second_attempt(sample_batch):
    call_count = 0
    valid_payload = json.dumps({
        "open_loops": [
            {
                "source_msg": "M018",
                "chat_id": "C04",
                "type": "commitment",
                "is_genuine": True,
                "title": "Send invoice to Brightside",
                "owner": "ME",
                "target_person": None,
                "status": "open",
                "closed_by": None,
                "due_date": None,
                "suggested_action": "assist",
            }
        ],
        "goals": [],
    })

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise urllib.error.HTTPError(
                url="https://api.groq.com/openai/v1/chat/completions",
                code=429,
                msg="Too Many Requests",
                hdrs={},
                fp=None,
            )
        return valid_payload

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert call_count == 2
    assert len(loops) == 1
    assert loops[0].source_msg == "M018"
    assert diag is None


def test_retry_http_503_succeeds_on_third_attempt(sample_batch):
    call_count = 0
    valid_payload = json.dumps({"open_loops": [], "goals": []})

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise urllib.error.HTTPError(
                url="https://api.groq.com/openai/v1/chat/completions",
                code=503,
                msg="Service Unavailable",
                hdrs={},
                fp=None,
            )
        return valid_payload

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert call_count == 3
    assert diag is None


def test_retry_network_url_error_succeeds_on_retry(sample_batch):
    call_count = 0
    valid_payload = json.dumps({"open_loops": [], "goals": []})

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise urllib.error.URLError("Connection refused")
        return valid_payload

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert call_count == 2
    assert diag is None


def test_retry_timeout_error_succeeds_on_retry(sample_batch):
    call_count = 0
    valid_payload = json.dumps({"open_loops": [], "goals": []})

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("Socket read timed out")
        return valid_payload

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert call_count == 2
    assert diag is None


def test_retry_parsing_error_succeeds_on_retry(sample_batch):
    call_count = 0
    valid_payload = json.dumps({"open_loops": [], "goals": []})

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "This is not valid json {"
        return valid_payload

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert call_count == 2
    assert diag is None


def test_retry_structure_error_succeeds_on_retry(sample_batch):
    call_count = 0
    valid_payload = json.dumps({"open_loops": [], "goals": []})

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({"wrong_top_level": []})  # Missing required open_loops and goals
        return valid_payload

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert call_count == 2
    assert diag is None


def test_exhausted_retries_records_diagnostic_and_continues(sample_batch):
    call_count = 0

    def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=2, retry_delay=0.01)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert call_count == 3  # 1 initial + 2 retries
    assert loops == []
    assert goals == []
    assert diag is not None
    assert diag.phase == "chat"
    assert diag.batch_id == sample_batch.chat_id
    assert diag.attempt_count == 3
    assert diag.error_type == "HTTPError:500"


# ==============================================================================
# 4. Failure Isolation Tests
# ==============================================================================

def test_chat_batch_failure_isolates_and_preserves_other_chat_batches(sample_context):
    # Two candidates in different chats: M001 in C01, M003 in C02
    candidates = [
        {"message_id": "M001", "chat_id": "C01", "sender_id": "ME", "category": "question", "timestamp": 1787530000},
        {"message_id": "M003", "chat_id": "C02", "sender_id": "ME", "category": "commitment", "timestamp": 1787531000},
    ]

    def mock_chat(messages, **kwargs):
        content = " ".join(m.get("content", "") for m in messages)
        # Fail for C01, succeed for C02
        if "CHAT_ID" in content and "C01" in content:
            raise urllib.error.HTTPError(
                url="https://api.groq.com/openai/v1/chat/completions",
                code=502,
                msg="Bad Gateway",
                hdrs={},
                fp=None,
            )
        # Succeed for C02 chat batch
        if "CHAT_ID" in content and "C02" in content:
            return json.dumps({
                "open_loops": [
                    {
                        "source_msg": "M003",
                        "chat_id": "C02",
                        "type": "commitment",
                        "is_genuine": True,
                        "title": "Pick up kids",
                        "owner": "ME",
                        "target_person": None,
                        "status": "open",
                        "closed_by": None,
                        "due_date": None,
                        "suggested_action": "remind",
                    }
                ],
                "goals": [],
            })
        # Participant payload
        return json.dumps({"people": [], "organisations": []})

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=1, retry_delay=0.01)
    result = interpreter.interpret(sample_context, candidates)

    # C01 failed, but C02 succeeded and was preserved!
    assert len(result.open_loops) == 1
    assert result.open_loops[0].source_msg == "M003"
    assert result.diagnostics.chat_batches_total == 2
    assert result.diagnostics.chat_batches_succeeded == 1
    assert result.diagnostics.chat_batches_failed == 1
    assert len(result.diagnostics.failures) == 1
    assert result.diagnostics.failures[0].batch_id == "C01"


def test_participant_batch_failure_isolates_and_preserves_other_participant_batches(sample_context):
    def mock_chat(messages, **kwargs):
        content = " ".join(m.get("content", "") for m in messages)
        # Fail if P01 is in batch, succeed otherwise
        if "P01" in content:
            raise urllib.error.HTTPError(
                url="https://api.groq.com/openai/v1/chat/completions",
                code=504,
                msg="Gateway Timeout",
                hdrs={},
                fp=None,
            )
        # For other participant batch
        return json.dumps({
            "people": [
                {
                    "person_id": "P02",
                    "circle": "staff",
                    "looks_after": [],
                    "associated_org_names": [],
                    "org_link_type": None,
                    "evidence_msgs": ["M003"],
                }
            ],
            "organisations": [],
        })

    # Chunk with 1 participant per batch to produce multiple batches
    interpreter = LLMEvidenceInterpreter(
        chat_fn=mock_chat,
        max_retries=1,
        retry_delay=0.01,
        max_participants_per_batch=1,
    )
    result = interpreter.interpret(sample_context, candidates=[])

    assert result.diagnostics.participant_batches_failed >= 1
    assert result.diagnostics.participant_batches_succeeded >= 1
    # People from the successful batches were preserved
    assert any(p.person_id == "P02" for p in result.people)


def test_chat_phase_failure_does_not_block_participant_phase(sample_context):
    candidate = {"message_id": "M001", "chat_id": "C01", "sender_id": "ME", "category": "question", "timestamp": 1787530000}

    def mock_chat(messages, **kwargs):
        content = " ".join(m.get("content", "") for m in messages)
        # Fail all chat prompts
        if "CHAT_ID" in content:
            raise urllib.error.HTTPError(
                url="https://api.groq.com/openai/v1/chat/completions",
                code=500,
                msg="Internal Error",
                hdrs={},
                fp=None,
            )
        # Succeed for participant prompts
        return json.dumps({
            "people": [
                {
                    "person_id": "P01",
                    "circle": "colleague",
                    "looks_after": [],
                    "associated_org_names": ["Brightside"],
                    "org_link_type": "supports",
                    "evidence_msgs": ["M002"],
                }
            ],
            "organisations": [
                {
                    "name": "Brightside",
                    "org_type": "company",
                    "side": "work",
                    "role": "partner",
                    "evidence_msgs": ["M002"],
                    "associated_chats": ["C01"],
                }
            ],
        })

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, max_retries=1, retry_delay=0.01)
    result = interpreter.interpret(sample_context, [candidate])

    assert len(result.open_loops) == 0
    assert result.diagnostics.chat_batches_failed == 1
    assert len(result.people) == 1
    assert result.people[0].person_id == "P01"
    assert len(result.organisations) == 1
    assert result.organisations[0].name == "Brightside"


# ==============================================================================
# 5. Empty Input & Invariant Tests
# ==============================================================================

def test_interpret_chat_batch_empty_candidates_returns_empty():
    batch = EvidenceBatch(
        chat_id="C01",
        messages=[{"id": "M01", "ts": 1}],
        candidates=[],
        candidate_message_ids=[],
        categories=[],
    )
    called = False

    def mock_chat(**kwargs):
        nonlocal called
        called = True
        return "{}"

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat)
    loops, goals, diag = interpreter.interpret_chat_batch(batch)
    assert loops == []
    assert goals == []
    assert diag is None
    assert called is False


def test_interpret_participant_batch_empty_dossiers_returns_empty():
    called = False

    def mock_chat(**kwargs):
        nonlocal called
        called = True
        return "{}"

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat)
    people, orgs, diag = interpreter.interpret_participant_batch([])
    assert people == []
    assert orgs == []
    assert diag is None
    assert called is False


def test_token_usage_not_duplicated_by_interpreter(monkeypatch, tmp_path, sample_batch):
    # Ensure interpreter does not directly write to or duplicate usage.jsonl
    # Injected chat_fn does not touch usage.jsonl
    called = False

    def mock_chat(**kwargs):
        nonlocal called
        called = True
        return json.dumps({"open_loops": [], "goals": []})

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat)
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert called is True
    assert diag is None


# ==============================================================================
# 6. End-to-End Orchestration Integration Test
# ==============================================================================

def test_end_to_end_interpret_with_mock_client(sample_context):
    candidates = [
        {"message_id": "M001", "chat_id": "C01", "sender_id": "ME", "category": "question", "timestamp": 1787530000},
        {"message_id": "M003", "chat_id": "C02", "sender_id": "ME", "category": "commitment", "timestamp": 1787531000},
    ]

    def mock_chat(messages, **kwargs):
        content = " ".join(m.get("content", "") for m in messages)
        if "CHAT_ID" in content:
            if "C01" in content:
                return json.dumps({
                    "open_loops": [
                        {
                            "source_msg": "M001",
                            "chat_id": "C01",
                            "type": "question",
                            "is_genuine": True,
                            "title": "Send report",
                            "owner": "P01",
                            "target_person": None,
                            "status": "done",
                            "closed_by": "M002",
                            "due_date": None,
                            "suggested_action": None,
                        }
                    ],
                    "goals": [
                        {
                            "title": "Complete client deliverable",
                            "area": "work",
                            "type": "milestone",
                            "horizon": "3mo",
                            "evidence_msgs": ["M001", "M002"],
                            "is_genuine": True,
                        }
                    ],
                })
            else:
                return json.dumps({"open_loops": [], "goals": []})

        if "PARTICIPANT_IDS" in content:
            return json.dumps({
                "people": [
                    {
                        "person_id": "P01",
                        "circle": "colleague",
                        "looks_after": [],
                        "associated_org_names": ["Brightside"],
                        "org_link_type": "supports",
                        "evidence_msgs": ["M002"],
                    }
                ],
                "organisations": [
                    {
                        "name": "Brightside",
                        "org_type": "company",
                        "side": "work",
                        "role": "partner",
                        "evidence_msgs": ["M002"],
                        "associated_chats": ["C01"],
                    }
                ],
            })
        return "{}"

    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat)
    result = interpreter.interpret(sample_context, candidates)

    assert isinstance(result, InterpretationResult)
    assert len(result.open_loops) == 1
    assert result.open_loops[0].source_msg == "M001"
    assert result.open_loops[0].closed_by == "M002"

    assert len(result.goals) == 1
    assert result.goals[0].title == "Complete client deliverable"

    assert len(result.people) == 1
    assert result.people[0].person_id == "P01"

    assert len(result.organisations) == 1
    assert result.organisations[0].name == "Brightside"

    assert result.diagnostics.has_failures is False
    assert result.diagnostics.chat_batches_succeeded == 2
    assert result.diagnostics.participant_batches_succeeded == 1


# ==============================================================================
# Milestone B Tests: Retries, Pacing, Budgeting, and Checkpointing
# ==============================================================================


def test_extract_retry_delay_numeric_headers():
    """Verify numeric Retry-After headers are extracted accurately as float seconds."""
    from http.client import HTTPMessage

    # Integer seconds
    msg = HTTPMessage()
    msg["Retry-After"] = "18"
    err = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg, io.BytesIO(b""))
    assert _extract_retry_delay(err) == 18.0

    # Float seconds
    msg_float = HTTPMessage()
    msg_float["retry-after"] = "2.5"
    err_float = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg_float, io.BytesIO(b""))
    assert _extract_retry_delay(err_float) == 2.5


def test_extract_retry_delay_http_date():
    """Verify RFC 7231 / RFC 2822 HTTP-date Retry-After headers are calculated against wall-clock time."""
    from http.client import HTTPMessage

    future_ts = time.time() + 25.0
    future_date_str = email.utils.formatdate(future_ts, usegmt=True)

    msg = HTTPMessage()
    msg["Retry-After"] = future_date_str
    err = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg, io.BytesIO(b""))
    extracted = _extract_retry_delay(err)
    assert extracted is not None
    # Allow tolerance for test execution time
    assert 22.0 <= extracted <= 28.0


def test_extract_retry_delay_ms_and_reset_headers():
    """Verify millisecond and token reset headers are extracted properly."""
    from http.client import HTTPMessage

    # retry-after-ms
    msg_ms = HTTPMessage()
    msg_ms["retry-after-ms"] = "3500"
    err_ms = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg_ms, io.BytesIO(b""))
    assert _extract_retry_delay(err_ms) == 3.5

    # x-ratelimit-reset-tokens duration
    msg_tok = HTTPMessage()
    msg_tok["x-ratelimit-reset-tokens"] = "14s"
    err_tok = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg_tok, io.BytesIO(b""))
    assert _extract_retry_delay(err_tok) == 14.0

    # x-ratelimit-reset relative
    msg_reset = HTTPMessage()
    msg_reset["x-ratelimit-reset"] = "45"
    err_reset = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg_reset, io.BytesIO(b""))
    assert _extract_retry_delay(err_reset) == 45.0


def test_extract_retry_delay_body_regex_fallback():
    """Verify regex fallback extracts seconds from error body JSON when headers are omitted."""
    from http.client import HTTPMessage

    body = b'{"error": {"message": "Rate limit exceeded. Please try again in 16.4s.", "type": "tokens"}}'
    err = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", HTTPMessage(), io.BytesIO(body))
    assert _extract_retry_delay(err) == 16.4


def test_strict_retry_floor_respected_and_never_premature(sample_batch):
    """Verify retry engine sleeps at least the stated provider floor without premature calls."""
    from http.client import HTTPMessage

    msg = HTTPMessage()
    msg["Retry-After"] = "0.05"
    err = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg, io.BytesIO(b""))

    call_timestamps = []

    def mock_chat(*args, **kwargs):
        call_timestamps.append(time.monotonic())
        if len(call_timestamps) < 2:
            raise err
        return json.dumps({
            "open_loops": [{
                "source_msg": "M018",
                "chat_id": "C04",
                "title": "Sent invoice",
                "type": "commitment",
                "owner": "ME",
                "status": "done",
            }],
            "goals": [],
        })

    interpreter = LLMEvidenceInterpreter(
        chat_fn=mock_chat,
        retry_delay=0.01,
        cache_enabled=False,
    )
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert diag is None
    assert len(call_timestamps) == 2
    # Verify elapsed time between attempts is strictly >= 0.05s (provider floor)
    elapsed_between = call_timestamps[1] - call_timestamps[0]
    assert elapsed_between >= 0.05


def test_budget_exhaustion_during_retry_defers_batch(sample_batch):
    """Verify that if retry delay exceeds remaining budget headroom, the batch is deferred immediately."""
    from http.client import HTTPMessage

    msg = HTTPMessage()
    msg["Retry-After"] = "50.0"
    err = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg, io.BytesIO(b""))

    call_count = 0

    def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise err

    # Budget with 55 seconds (leaving 10s headroom for call 1, but failing 50s retry + 45s reserve)
    budget = ExecutionBudget(total_seconds=55.0, reserve_seconds=45.0, call_timeout=5.0)
    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, cache_enabled=False)

    t0 = time.monotonic()
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch, budget=budget)
    elapsed = time.monotonic() - t0

    assert diag is not None
    assert diag.error_type == "TimeoutBudgetExceeded"
    assert call_count == 1  # Did not retry
    assert elapsed < 1.0  # Did not sleep the 50s


def test_dynamic_effective_timeout_bounded_by_remaining_headroom(sample_batch):
    """Verify effective socket timeout passed to chat_fn is dynamically bounded by remaining headroom."""
    received_timeouts = []

    def mock_chat(*args, **kwargs):
        received_timeouts.append(kwargs.get("timeout"))
        return json.dumps({"open_loops": [], "goals": []})

    # Total 65s budget with 45s reserve leaves 20s available headroom
    budget = ExecutionBudget(total_seconds=65.0, reserve_seconds=45.0, call_timeout=120.0)
    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, cache_enabled=False)

    interpreter.interpret_chat_batch(sample_batch, budget=budget)

    assert len(received_timeouts) == 1
    # Effective timeout should be bounded to approx 20s, NOT 120s!
    assert 18.0 <= received_timeouts[0] <= 20.5


def test_dynamic_effective_timeout_below_minimum_defers_batch(sample_batch):
    """Verify that if remaining headroom is below min_call_timeout (1.0s), batch is deferred."""
    call_count = 0

    def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return "{}"

    # Remaining headroom is exactly 0.5s, less than min_call_timeout 1.0s
    budget = ExecutionBudget(total_seconds=45.5, reserve_seconds=45.0, call_timeout=120.0, min_call_timeout=1.0)
    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, cache_enabled=False)

    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch, budget=budget)

    assert diag is not None
    assert diag.error_type == "TimeoutBudgetExceeded"
    assert call_count == 0


def test_cache_first_succeeds_even_when_network_budget_exhausted(sample_batch, tmp_path):
    """Verify cache hits succeed instantly even when 0s remains in the network budget."""
    cache_file = str(tmp_path / "test_cache.json")

    # Valid chat response with is_genuine: True
    valid_resp = json.dumps({
        "open_loops": [{
            "source_msg": "M018",
            "chat_id": "C04",
            "title": "Sent invoice to Brightside",
            "type": "commitment",
            "is_genuine": True,
            "owner": "ME",
            "status": "done",
        }],
        "goals": [],
    })

    interpreter = LLMEvidenceInterpreter(
        chat_fn=lambda *a, **k: pytest.fail("Should not call API on cache hit"),
        cache_enabled=True,
        cache_path=cache_file,
    )
    # Seed cache
    messages = build_chat_interpretation_prompt(sample_batch)
    key = interpreter.cache.compute_key(
        phase="chat",
        prompt_version="chat_interpretation_v1.0",
        base_url=os.environ.get("LLM_BASE_URL", ""),
        model="",
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    interpreter.cache.set(key, valid_resp)

    # Set budget with 0 seconds remaining via elapsed start_monotonic
    budget = ExecutionBudget(total_seconds=50.0, reserve_seconds=45.0, start_monotonic=time.monotonic() - 100.0)
    assert budget.remaining_seconds == 0.0

    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch, budget=budget)

    assert diag is None
    assert interpreter.last_call_cached is True
    assert len(loops) == 1
    assert loops[0].source_msg == "M018"


def test_invalid_cache_entry_evicted_and_falls_through_to_api(sample_batch, tmp_path):
    """Verify that corrupted cache entries are evicted and fall through to fresh API calls."""
    cache_file = str(tmp_path / "test_cache.json")

    api_called = False

    def mock_chat(*args, **kwargs):
        nonlocal api_called
        api_called = True
        return json.dumps({
            "open_loops": [{
                "source_msg": "M018",
                "chat_id": "C04",
                "title": "Sent invoice to Brightside",
                "type": "commitment",
                "is_genuine": True,
                "owner": "ME",
                "status": "done",
            }],
            "goals": [],
        })

    interpreter = LLMEvidenceInterpreter(
        chat_fn=mock_chat,
        cache_enabled=True,
        cache_path=cache_file,
    )

    # Seed invalid/corrupted JSON in cache
    messages = build_chat_interpretation_prompt(sample_batch)
    key = interpreter.cache.compute_key(
        phase="chat",
        prompt_version="chat_interpretation_v1.0",
        base_url=os.environ.get("LLM_BASE_URL", ""),
        model="",
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    interpreter.cache.set(key, "not valid json at all")

    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)

    assert diag is None
    assert api_called is True
    assert len(loops) == 1
    # Verify corrupted key was evicted and replaced with clean response
    assert interpreter.cache.get(key) is not None
    assert "not valid json" not in interpreter.cache.get(key)


def test_participant_cache_lookup_and_parser_revalidation(sample_context, tmp_path):
    """Verify participant phase caching and revalidation via parse_participant_interpretation_response."""
    cache_file = str(tmp_path / "part_cache.json")
    dossiers_batch = build_participant_dossiers(sample_context, max_participants_per_batch=2)
    dossiers, chat_meta = dossiers_batch[0]

    valid_part_resp = json.dumps({
        "people": [{
            "person_id": dossiers[0]["person_id"],
            "circle": "friend",
            "relationship": "colleague",
            "associated_org_names": ["Brightside"],
            "org_link_type": "supports",
            "evidence_msgs": ["M018"],
        }],
        "organisations": [{
            "name": "Brightside",
            "org_type": "company",
            "side": "work",
            "role": "partner",
            "evidence_msgs": ["M018"],
            "associated_chats": ["C04"],
        }],
    })

    call_count = 0

    def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return valid_part_resp

    interpreter = LLMEvidenceInterpreter(
        chat_fn=mock_chat,
        cache_enabled=True,
        cache_path=cache_file,
    )

    # 1. First call: Cache miss -> calls API
    p1, o1, diag1 = interpreter.interpret_participant_batch(dossiers, chat_metadata=chat_meta, context=sample_context)
    assert diag1 is None
    assert call_count == 1
    assert interpreter.last_call_cached is False
    assert len(p1) == 1
    assert len(o1) == 1

    # 2. Second call: Cache hit -> bypasses API
    p2, o2, diag2 = interpreter.interpret_participant_batch(dossiers, chat_metadata=chat_meta, context=sample_context)
    assert diag2 is None
    assert call_count == 1  # Not incremented!
    assert interpreter.last_call_cached is True
    assert len(p2) == 1
    assert len(o2) == 1


def test_shared_monotonic_budget_across_multiple_batches(sample_batch):
    """Verify a single shared budget is consumed across batches without resetting start time."""
    def mock_chat(*args, **kwargs):
        time.sleep(0.05)  # Simulate small work
        return json.dumps({"open_loops": [], "goals": []})

    budget = ExecutionBudget(total_seconds=500.0)
    interpreter = LLMEvidenceInterpreter(chat_fn=mock_chat, cache_enabled=False)

    rem0 = budget.remaining_seconds
    interpreter.interpret_chat_batch(sample_batch, budget=budget)
    rem1 = budget.remaining_seconds
    interpreter.interpret_chat_batch(sample_batch, budget=budget)
    rem2 = budget.remaining_seconds

    # Remaining time strictly decreases monotonically
    assert rem1 < rem0
    assert rem2 < rem1


def test_cache_atomic_persistence_and_secret_exclusion(tmp_path):
    """Verify cache writes are atomic and strictly exclude API keys and authorization headers."""
    cache_path = str(tmp_path / "subdir" / "cache.json")
    cache = InterpretationCache(cache_path=cache_path, enabled=True)

    metadata = {
        "model": "gpt-4",
        "api_key": "sk-secret-leak",
        "authorization": "Bearer secret-token",
        "batch_id": "C01",
    }
    cache.set("test_key", '{"status": "ok"}', metadata=metadata)

    assert os.path.exists(cache_path)
    with open(cache_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "test_key" in content
    assert "ok" in content
    assert "C01" in content
    # Secrets strictly excluded!
    assert "sk-secret-leak" not in content
    assert "secret-token" not in content


def test_disabled_cache_leaves_disk_clean(sample_batch, tmp_path):
    """Verify disabled cache does not create or write files to disk."""
    cache_file = str(tmp_path / "disabled_cache.json")

    interpreter = LLMEvidenceInterpreter(
        chat_fn=lambda *a, **k: json.dumps({"open_loops": [], "goals": []}),
        cache_enabled=False,
        cache_path=cache_file,
    )

    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)
    assert diag is None
    assert not os.path.exists(cache_file)


def test_backward_compatibility_without_budget_or_cache(sample_batch):
    """Verify legacy callers without budget or custom cache parameters execute cleanly."""
    interpreter = LLMEvidenceInterpreter(
        chat_fn=lambda *a, **k: json.dumps({"open_loops": [], "goals": []}),
        cache_enabled=False,
    )
    loops, goals, diag = interpreter.interpret_chat_batch(sample_batch)
    assert diag is None
    assert isinstance(loops, list)
    assert isinstance(goals, list)


# ==============================================================================
# Milestone B Correction Tests: Callable Inspection, Pacing on Failures,
# Maximum Reset Header Delays, and Strict Numeric Validation
# ==============================================================================

from src.loops.extractor import interpret_open_loop_evidence
from src.loops.models import OpenLoopExtractionConfig, OpenLoopExtractionDiagnostics


def test_interpret_open_loop_evidence_with_interpreter_without_budget(sample_context):
    """Verify interpret_open_loop_evidence cleanly supports test doubles without a budget parameter."""
    candidates = [
        {"message_id": "M001", "chat_id": "C01", "sender_id": "ME", "category": "commitment", "timestamp": 1787530000}
    ]

    class LegacyInterpreter:
        def __init__(self):
            self.calls = []

        def interpret_chat_batch(self, batch, context=None):
            self.calls.append((batch, context))
            loop = InterpretedOpenLoop(
                source_msg="M001",
                chat_id="C01",
                title="Send the report",
                type="commitment",
                is_genuine=True,
                owner="ME",
                target_person=None,
                status="open",
                closed_by=None,
                due_date=None,
                suggested_action="remind",
            )
            return [loop], [], None

    legacy = LegacyInterpreter()
    diag = OpenLoopExtractionDiagnostics()
    budget = ExecutionBudget(total_seconds=500.0)

    loops = interpret_open_loop_evidence(sample_context, candidates, interpreter=legacy, diagnostics=diag, budget=budget)

    assert len(legacy.calls) == 1
    assert len(loops) == 1
    assert loops[0].source_msg == "M001"
    assert diag.batches_succeeded == 1


def test_interpret_open_loop_evidence_internal_typeerror_not_retried(sample_context):
    """Verify that an internal TypeError inside interpret_chat_batch is not retried and called exactly once."""
    candidates = [
        {"message_id": "M001", "chat_id": "C01", "sender_id": "ME", "category": "commitment", "timestamp": 1787530000}
    ]

    call_count = 0

    class BuggyInterpreter:
        def interpret_chat_batch(self, batch, context=None, budget=None):
            nonlocal call_count
            call_count += 1
            # Simulate internal TypeError unrelated to parameter signature
            raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")

    buggy = BuggyInterpreter()
    diag = OpenLoopExtractionDiagnostics()
    budget = ExecutionBudget(total_seconds=500.0)

    loops = interpret_open_loop_evidence(sample_context, candidates, interpreter=buggy, diagnostics=diag, budget=budget)

    assert call_count == 1  # Crucial: exactly 1 call, NEVER retried!
    assert loops == []
    assert diag.batches_failed == 1


def test_adaptive_pacing_observed_after_failed_network_dispatch(sample_batch):
    """Verify that a failed network attempt updates monotonic timestamp and subsequent batch observes pacing."""
    call_timestamps = []
    attempt = 0

    def mock_chat(*args, **kwargs):
        nonlocal attempt
        attempt += 1
        call_timestamps.append(time.monotonic())
        if attempt == 1:
            # Batch 1 fails with HTTP 500
            raise urllib.error.HTTPError(
                url="http://api.test",
                code=500,
                msg="Internal Server Error",
                hdrs={},
                fp=None,
            )
        # Batch 2 succeeds
        return json.dumps({"open_loops": [], "goals": []})

    interpreter = LLMEvidenceInterpreter(
        chat_fn=mock_chat,
        max_retries=0,  # Single attempt per batch to isolate inter-batch pacing
        inter_batch_pacing=0.15,  # 150ms pacing
        cache_enabled=False,
    )

    # Dispatch batch 1 (fails)
    loops1, goals1, diag1 = interpreter.interpret_chat_batch(sample_batch)
    assert diag1 is not None
    assert interpreter._last_api_call_monotonic is not None

    # Dispatch batch 2 (should observe pacing from the failed batch 1)
    loops2, goals2, diag2 = interpreter.interpret_chat_batch(sample_batch)
    assert diag2 is None

    assert len(call_timestamps) == 2
    elapsed_between_calls = call_timestamps[1] - call_timestamps[0]
    # Verify the delay between dispatch 1 and dispatch 2 reflects the 150ms pacing (with reasonable jitter buffer)
    assert elapsed_between_calls >= 0.12


def test_extract_retry_delay_combines_multiple_reset_headers_and_takes_max():
    """Verify that multiple rate-limit reset headers are combined and the maximum delay is respected."""
    from http.client import HTTPMessage

    msg = HTTPMessage()
    msg["Retry-After"] = "4.5"
    msg["x-ratelimit-reset-requests"] = "2.0"
    msg["x-ratelimit-reset-tokens"] = "18.25"
    msg["x-ratelimit-reset"] = "6.0"

    err = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg, io.BytesIO(b""))

    delay = _extract_retry_delay(err)
    assert delay == 18.25  # Max of [4.5, 2.0, 18.25, 6.0]


def test_extract_retry_delay_malformed_headers_do_not_crash():
    """Verify that malformed reset headers are gracefully skipped without crashing."""
    from http.client import HTTPMessage

    msg = HTTPMessage()
    msg["Retry-After"] = "invalid-seconds-string"
    msg["x-ratelimit-reset-tokens"] = "not_a_number"
    msg["x-ratelimit-reset-requests"] = "3.5"

    err = urllib.error.HTTPError("http://api.test", 429, "Too Many Requests", msg, io.BytesIO(b""))

    delay = _extract_retry_delay(err)
    assert delay == 3.5  # Skips malformed and extracts valid 3.5


def test_execution_budget_numeric_validation():
    """Verify ExecutionBudget strictly rejects booleans, NaN, inf, zero, and negative values."""
    # 1. total_seconds validation
    with pytest.raises(TypeError):
        ExecutionBudget(total_seconds=True)
    with pytest.raises(TypeError):
        ExecutionBudget(total_seconds="100")
    with pytest.raises(ValueError):
        ExecutionBudget(total_seconds=float("nan"))
    with pytest.raises(ValueError):
        ExecutionBudget(total_seconds=float("inf"))
    with pytest.raises(ValueError):
        ExecutionBudget(total_seconds=0.0)
    with pytest.raises(ValueError):
        ExecutionBudget(total_seconds=-10.0)

    # 2. call_timeout validation
    with pytest.raises(TypeError):
        ExecutionBudget(call_timeout=False)
    with pytest.raises(ValueError):
        ExecutionBudget(call_timeout=0.0)
    with pytest.raises(ValueError):
        ExecutionBudget(call_timeout=-5.0)
    with pytest.raises(ValueError):
        ExecutionBudget(call_timeout=float("nan"))

    # 3. min_call_timeout validation
    with pytest.raises(TypeError):
        ExecutionBudget(min_call_timeout=True)
    with pytest.raises(ValueError):
        ExecutionBudget(min_call_timeout=0.0)
    with pytest.raises(ValueError):
        ExecutionBudget(min_call_timeout=-1.0)
    with pytest.raises(ValueError):
        ExecutionBudget(min_call_timeout=float("inf"))

    # 4. reserve_seconds validation (non-negative and strictly smaller than total_seconds)
    with pytest.raises(TypeError):
        ExecutionBudget(reserve_seconds=False)
    with pytest.raises(ValueError):
        ExecutionBudget(reserve_seconds=-1.0)
    with pytest.raises(ValueError):
        ExecutionBudget(total_seconds=45.0, reserve_seconds=45.0)  # reserve == total
    with pytest.raises(ValueError):
        ExecutionBudget(total_seconds=40.0, reserve_seconds=50.0)  # reserve > total

    # Valid boundary cases
    b = ExecutionBudget(total_seconds=45.1, reserve_seconds=45.0)
    assert b.total_seconds == 45.1
    assert b.reserve_seconds == 45.0

    b0 = ExecutionBudget(total_seconds=10.0, reserve_seconds=0.0)
    assert b0.reserve_seconds == 0.0


def test_interpreter_numeric_validation():
    """Verify LLMEvidenceInterpreter strictly rejects booleans, NaN, inf, zero, and invalid bounds."""
    # timeout
    with pytest.raises(TypeError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=True)
    with pytest.raises(TypeError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout="120")
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=0)
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=-10.0)
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=float("nan"))
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=float("inf"))

    # min_call_timeout
    with pytest.raises(TypeError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", min_call_timeout=False)
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", min_call_timeout=0.0)

    # max_runtime_seconds
    with pytest.raises(TypeError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", max_runtime_seconds=True)
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", max_runtime_seconds=float("inf"))
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", max_runtime_seconds=0.0)

    # reserve_seconds >= max_runtime_seconds
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", max_runtime_seconds=50.0, reserve_seconds=60.0)

    # start_session validation
    interp = LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", max_runtime_seconds=100.0, reserve_seconds=45.0)
    with pytest.raises(TypeError):
        interp.start_session(max_runtime_seconds=True)
    with pytest.raises(ValueError):
        interp.start_session(max_runtime_seconds=float("nan"))
    with pytest.raises(ValueError):
        interp.start_session(max_runtime_seconds=40.0)  # < reserve_seconds


def test_interpreter_fractional_timeout_preservation():
    """Verify fractional timeout values are preserved without silent integer truncation."""
    # 1. timeout=0.5 remains positive and usable when starting a session
    interp_half = LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=0.5)
    assert interp_half.timeout == 0.5
    budget_half = interp_half.start_session()
    assert budget_half.call_timeout == 0.5
    assert budget_half.call_timeout > 0.0

    # 2. timeout=1.9 is not silently changed to 1
    interp_float = LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=1.9)
    assert interp_float.timeout == 1.9
    assert interp_float.timeout != 1
    budget_float = interp_float.start_session()
    assert budget_float.call_timeout == 1.9
    assert budget_float.call_timeout != 1.0

    # 3. Existing invalid timeout cases continue to be rejected
    with pytest.raises(TypeError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=True)
    with pytest.raises(TypeError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout="0.5")
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=0.0)
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=-0.5)
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=float("nan"))
    with pytest.raises(ValueError):
        LLMEvidenceInterpreter(chat_fn=lambda *a, **k: "{}", timeout=float("inf"))


def test_open_loop_extraction_config_rejects_non_finite_max_runtime():
    """Verify OpenLoopExtractionConfig rejects NaN, positive/negative infinity, booleans, zero, and negative values."""
    with pytest.raises(TypeError):
        OpenLoopExtractionConfig(max_runtime_seconds=True)
    with pytest.raises(TypeError):
        OpenLoopExtractionConfig(max_runtime_seconds=False)
    with pytest.raises(TypeError):
        OpenLoopExtractionConfig(max_runtime_seconds="1500")
    with pytest.raises(ValueError):
        OpenLoopExtractionConfig(max_runtime_seconds=float("nan"))
    with pytest.raises(ValueError):
        OpenLoopExtractionConfig(max_runtime_seconds=float("inf"))
    with pytest.raises(ValueError):
        OpenLoopExtractionConfig(max_runtime_seconds=float("-inf"))
    with pytest.raises(ValueError):
        OpenLoopExtractionConfig(max_runtime_seconds=0.0)
    with pytest.raises(ValueError):
        OpenLoopExtractionConfig(max_runtime_seconds=-100.0)

    # None and strictly positive numbers are valid
    cfg_none = OpenLoopExtractionConfig(max_runtime_seconds=None)
    assert cfg_none.max_runtime_seconds is None
    cfg_pos = OpenLoopExtractionConfig(max_runtime_seconds=120.5)
    assert cfg_pos.max_runtime_seconds == 120.5

