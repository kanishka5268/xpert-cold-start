#!/usr/bin/env python3
"""Main solution pipeline connecting ingestion, entity extraction, goals, and open loops.

Reads conversation input data and writes schema-compliant output.json:

    python3 solution.py <input_dir> <output.json>
"""
import json
from pathlib import Path
import sys
from typing import Optional

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion.loader import load_dataset
from src.preprocessing.context_builder import build_conversation_context
from src.entities.extractor import extract_entities
from src.goals.extractor import extract_goals
from src.loops.extractor import extract_open_loops
from src.interpretation.interpreter import (
    LLMEvidenceInterpreter,
    LLMConfigurationError,
    _sanitize_error_message,
)


def main(input_dir: str, output_path: str, offline: Optional[bool] = None) -> None:
    """Execute end-to-end extraction pipeline across all five challenge domains."""
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass

    if offline is None:
        import os
        offline = (
            "--offline" in sys.argv
            or "-o" in sys.argv
            or os.environ.get("OFFLINE_MODE", "").strip().lower() in ("1", "true", "yes")
            or os.environ.get("LLM_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        )

    # 1. Ingestion & Preprocessing
    dataset = load_dataset(input_dir)
    context = build_conversation_context(dataset)

    # 2. Entity Extraction (People, Organisations, Chats)
    entity_result = extract_entities(context)

    # Ensure 100% participant coverage, defaulting unclassified participants to 'friend'
    existing_pids = {p["id"] for p in entity_result.people}
    all_pids = sorted(pid for pid in context.participants if pid != "ME")
    final_people = list(entity_result.people)
    for pid in all_pids:
        if pid not in existing_pids:
            final_people.append({"id": pid, "circle": "friend"})
    final_people.sort(key=lambda p: p["id"])

    # Instantiate interpreter for LLM interpretation stages (goals and open loops)
    interpreter: Optional[LLMEvidenceInterpreter] = None
    shared_budget = None
    if offline:
        try:
            interpreter = LLMEvidenceInterpreter(allow_offline=True, offline_mode=True, cache_enabled=False)
            shared_budget = interpreter.start_session(max_runtime_seconds=1650.0)
        except Exception as err:
            sanitized_err = _sanitize_error_message(err)
            print(
                f"[Stage: LLM Initialization] Warning creating offline LLM interpreter ({sanitized_err}). Proceeding in offline fallback mode.",
                file=sys.stderr,
            )
            interpreter = None
    else:
        try:
            interpreter = LLMEvidenceInterpreter(allow_offline=False)
            interpreter.allow_offline = True
            shared_budget = interpreter.start_session(max_runtime_seconds=1650.0)
        except Exception as err:
            sanitized_err = _sanitize_error_message(err)
            print(
                f"[Stage: LLM Initialization] Warning creating LLM interpreter ({sanitized_err}). Proceeding in offline fallback mode.",
                file=sys.stderr,
            )
            interpreter = None

    # 3. Goal Extraction
    final_goals = []
    try:
        if interpreter is None:
            print(
                "[Stage: Goal Extraction] LLM interpreter unavailable. Emitting degraded empty goals list.",
                file=sys.stderr,
            )
            final_goals = []
        else:
            goal_result = extract_goals(context=context, interpreter=interpreter, budget=shared_budget)
            final_goals = goal_result.goals
    except Exception as err:
        sanitized_err = _sanitize_error_message(err)
        print(
            f"[Stage: Goal Extraction] Unexpected failure ({sanitized_err}). Emitting degraded empty goals list.",
            file=sys.stderr,
        )
        final_goals = []

    # 4. Open-Loop Extraction
    final_loops = []
    try:
        if interpreter is None:
            print(
                "[Stage: Open-Loop Extraction] LLM interpreter unavailable. Executing deterministic fallback extraction.",
                file=sys.stderr,
            )
        loop_result = extract_open_loops(context=context, interpreter=interpreter, budget=shared_budget)
        final_loops = loop_result.open_loops
    except Exception as err:
        sanitized_err = _sanitize_error_message(err)
        print(
            f"[Stage: Open-Loop Extraction] Unexpected failure ({sanitized_err}). Emitting degraded empty open loops list.",
            file=sys.stderr,
        )
        final_loops = []

    out = {
        "people": final_people,
        "organisations": entity_result.organisations,
        "chats": entity_result.chats,
        "goals": final_goals,
        "open_loops": final_loops,
    }

    Path(output_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raw_args = sys.argv[1:]
    is_offline_flag = any(a in ("--offline", "-o") for a in raw_args)
    pos_args = [a for a in raw_args if a not in ("--offline", "-o")]
    if len(pos_args) != 2:
        sys.exit(__doc__)
    main(pos_args[0], pos_args[1], offline=True if is_offline_flag else None)
