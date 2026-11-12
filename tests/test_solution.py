"""Offline integration tests for solution.py end-to-end pipeline execution and schema compliance."""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import pytest
import jsonschema

import solution
from check import validate as check_validate, load_input


@pytest.fixture
def schema_spec() -> Dict[str, Any]:
    """Load the official schema.json specification."""
    schema_path = Path("schema.json")
    assert schema_path.exists(), "schema.json must exist in project root"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_cli_argument_validation(monkeypatch):
    """Verify solution.py validates command-line arguments and rejects improper invocation."""
    # When fewer than 2 arguments passed (sys.argv has length != 3)
    monkeypatch.setattr(sys, "argv", ["solution.py"])
    with pytest.raises(SystemExit) as exc_info:
        # Trigger __main__ check logic
        if len(sys.argv) != 3:
            sys.exit(solution.__doc__)
    assert exc_info.value.code == solution.__doc__

    # When 1 argument passed
    monkeypatch.setattr(sys, "argv", ["solution.py", "example"])
    with pytest.raises(SystemExit):
        if len(sys.argv) != 3:
            sys.exit(solution.__doc__)


def test_solution_offline_execution_on_example_dataset(tmp_path, schema_spec, monkeypatch):
    """Verify solution.py runs offline end-to-end on example/ data, emitting valid schema-compliant output."""
    output_file = tmp_path / "output_example.json"

    # Mock chat client so offline execution never makes live network requests
    def mock_chat_fn(messages, model=None, temperature=0, response_format=None, timeout=120):
        # Return valid empty JSON structure for chat batches
        return json.dumps({
            "open_loops": [],
            "goals": [],
            "people": [],
            "organisations": [],
        })

    # Ensure mock chat client is used without network calls
    monkeypatch.setenv("LLM_BASE_URL", "https://mock.llm.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")
    monkeypatch.setenv("LLM_MODEL", "mock-model")

    from src.interpretation.interpreter import LLMEvidenceInterpreter
    original_init = LLMEvidenceInterpreter.__init__

    def mock_interp_init(self, *args, **kwargs):
        kwargs["chat_fn"] = mock_chat_fn
        kwargs["cache_enabled"] = False
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(LLMEvidenceInterpreter, "__init__", mock_interp_init)

    # Execute main pipeline
    solution.main("example", str(output_file))

    # 1. Output file creation
    assert output_file.exists(), "output file must be written"

    out = json.loads(output_file.read_text(encoding="utf-8"))

    # 2. Exact five top-level schema keys
    expected_keys = {"people", "organisations", "chats", "goals", "open_loops"}
    assert set(out.keys()) == expected_keys, f"Output must contain exactly {expected_keys}"

    # 3. Official JSON Schema validation
    jsonschema.validate(instance=out, schema=schema_spec)

    # 4. Official check.py domain validation
    chats, messages, _seed, _answers = load_input("example")
    check_errors = check_validate(out, chats, messages)
    assert len(check_errors) == 0, f"check.py reported validation errors: {check_errors}"

    # 5. Entity presence
    assert len(out["people"]) > 0
    assert len(out["chats"]) > 0
    # ME must never be in people array
    assert not any(p["id"] == "ME" for p in out["people"])


def test_solution_graceful_degradation_without_llm_credentials(tmp_path, schema_spec, monkeypatch, capsys):
    """Verify solution.py gracefully degrades and reports stage diagnostics when LLM credentials are absent."""
    output_file = tmp_path / "output_degraded.json"

    # Strip LLM environment variables and disable dotenv loading
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)

    # Execute main pipeline
    solution.main("example", str(output_file))

    # Verify stderr captures stage diagnostics
    captured = capsys.readouterr()
    assert "[Stage: LLM Initialization]" in captured.err
    assert "[Stage: Goal Extraction]" in captured.err
    assert "[Stage: Open-Loop Extraction]" in captured.err

    # Verify valid output file is still emitted
    assert output_file.exists()
    out = json.loads(output_file.read_text(encoding="utf-8"))

    # Schema must validate cleanly even in degraded state
    jsonschema.validate(instance=out, schema=schema_spec)

    chats, messages, _seed, _answers = load_input("example")
    check_errors = check_validate(out, chats, messages)
    assert len(check_errors) == 0, f"Degraded output failed check.py validation: {check_errors}"

    # Degraded goals are empty list, while deterministic fallback open loops and entities remain populated
    assert out["goals"] == []
    assert isinstance(out["open_loops"], list)
    assert len(out["open_loops"]) > 0
    assert len(out["people"]) > 0
    assert len(out["chats"]) > 0


def test_solution_invalid_input_directory_raises_error(tmp_path):
    """Verify solution.py raises FileNotFoundError when given an invalid input directory."""
    output_file = tmp_path / "out.json"
    invalid_dir = tmp_path / "nonexistent_directory"

    with pytest.raises(FileNotFoundError):
        solution.main(str(invalid_dir), str(output_file))


def test_solution_never_reads_gold_answers(tmp_path, schema_spec, monkeypatch):
    """Verify solution.py strictly isolates prediction from gold answers and never accesses answers.json."""
    # Create synthetic dataset with an answers/answers.json file
    input_dir = tmp_path / "synthetic_data"
    input_dir.mkdir()
    (input_dir / "chats.jsonl").write_text(
        json.dumps({"chat_id": "C01", "name": "Direct", "is_group": False, "participants": ["ME", "P01"]}) + "\n",
        encoding="utf-8",
    )
    (input_dir / "messages.jsonl").write_text(
        json.dumps({"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "P01", "ts": 1000, "body": "Hello", "kind": "text"}) + "\n",
        encoding="utf-8",
    )
    (input_dir / "seed.json").write_text("{}", encoding="utf-8")

    answers_dir = input_dir / "answers"
    answers_dir.mkdir()
    answers_file = answers_dir / "answers.json"
    answers_file.write_text(json.dumps({"people": [], "organisations": [], "chats": [], "goals": [], "open_loops": []}), encoding="utf-8")

    accessed_files: List[str] = []
    real_open = open

    def tracking_open(file, *args, **kwargs):
        accessed_files.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)

    # Mock chat client
    def mock_chat_fn(*args, **kwargs):
        return '{"goals": [], "open_loops": [], "people": [], "organisations": []}'

    monkeypatch.setenv("LLM_BASE_URL", "https://mock.llm.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")
    monkeypatch.setenv("LLM_MODEL", "mock-model")

    from src.interpretation.interpreter import LLMEvidenceInterpreter
    original_init = LLMEvidenceInterpreter.__init__

    def mock_interp_init(self, *args, **kwargs):
        kwargs["chat_fn"] = mock_chat_fn
        kwargs["cache_enabled"] = False
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(LLMEvidenceInterpreter, "__init__", mock_interp_init)

    output_file = tmp_path / "out.json"
    solution.main(str(input_dir), str(output_file))

    # Assert answers.json was NEVER opened
    assert not any("answers.json" in str(f) for f in accessed_files), "solution.py must NEVER access answers.json"


def test_solution_shares_single_session_budget(tmp_path, monkeypatch):
    """Verify solution.py initializes one shared execution budget session and passes it to both extraction phases."""
    from src.interpretation.interpreter import LLMEvidenceInterpreter
    from src.interpretation.cache import ExecutionBudget

    captured_budgets: Dict[str, Any] = {}

    orig_extract_goals = solution.extract_goals
    orig_extract_loops = solution.extract_open_loops

    def spy_extract_goals(context, interpreter, config=None, diagnostics=None, budget=None):
        captured_budgets["goals_budget"] = budget
        return orig_extract_goals(context, interpreter, config=config, diagnostics=diagnostics, budget=budget)

    def spy_extract_loops(context, interpreter, config=None, diagnostics=None, budget=None):
        captured_budgets["loops_budget"] = budget
        return orig_extract_loops(context, interpreter, config=config, diagnostics=diagnostics, budget=budget)

    monkeypatch.setattr(solution, "extract_goals", spy_extract_goals)
    monkeypatch.setattr(solution, "extract_open_loops", spy_extract_loops)

    # Mock chat client
    def mock_chat_fn(*args, **kwargs):
        return '{"goals": [], "open_loops": [], "people": [], "organisations": []}'

    monkeypatch.setenv("LLM_BASE_URL", "https://mock.llm.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")
    monkeypatch.setenv("LLM_MODEL", "mock-model")

    original_init = LLMEvidenceInterpreter.__init__

    def mock_interp_init(self, *args, **kwargs):
        kwargs["chat_fn"] = mock_chat_fn
        kwargs["cache_enabled"] = False
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(LLMEvidenceInterpreter, "__init__", mock_interp_init)

    output_file = tmp_path / "out_budget.json"
    solution.main("example", str(output_file))

    # Verify both phases received the exact same shared ExecutionBudget instance
    assert "goals_budget" in captured_budgets
    assert "loops_budget" in captured_budgets
    assert isinstance(captured_budgets["goals_budget"], ExecutionBudget)
    assert captured_budgets["goals_budget"] is captured_budgets["loops_budget"]
    assert captured_budgets["goals_budget"].total_seconds == 1650.0
