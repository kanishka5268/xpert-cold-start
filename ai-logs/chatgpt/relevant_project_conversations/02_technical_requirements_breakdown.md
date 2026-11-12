# ChatGPT Consultation: Technical Requirements Breakdown

## Role of ChatGPT

ChatGPT supported the developer in decomposing the Cold Start Challenge specifications and the official evaluator (`check.py`) into concrete, actionable engineering requirements and testable milestones.

## Key Consultation Topics

### 1. Reverse-Engineering Evaluator Mechanics (`check.py`)
- **Problem**: Understanding exactly how scores are calculated across entities, goals, open loops, and help suggestions to ensure no effort was wasted on un-scored attributes.
- **Discussion with ChatGPT**:
  - Analyzed `check.py` matching rules:
    - **Entities**: Exact set matching on circles (participants), organizations (members), and chats.
    - **Goals**: Seeded goals in `example/seed_goals.json` are excluded from scoring. Remaining goals require:
      - Exact string match on `area` (e.g. `work`, `health`).
      - Overlapping evidence: at least one message ID in `evidence_message_ids` must match ground truth.
      - 1-to-1 maximum bipartite matching scored via F1.
    - **Open Loops**: Evaluated in two dimensions:
      - *Discovery (Loops Found)*: Correct message grounding and assignment.
      - *Closure Accuracy (Closed Right)*: Correct distinction between `open` vs `done`/`dropped`.
    - **Where-to-Help**: Action recommendations (`remind`, `leave`, `step_in`) evaluated for open obligations.
  - Outcome: Mapped out exact JSON schema contracts for each component.

### 2. Decomposing the Pipeline into Milestones
- **Discussion with ChatGPT**:
  - Structured the development roadmap into clean, test-driven phases:
    - Phase A: Ingestion and schema baseline (`ingest.py`, `output.py`).
    - Phase B: Social graph and entity resolution (`entities/`).
    - Phase C: Open loop extraction and lifecycle resolution (`loops/`).
    - Phase D: Goal extraction and temporal alignment (`goals/`).
    - Phase E: Live LLM interpretation and selective batching (`interpretation/`).
    - Phase F: System integration and budget enforcement (`solution.py`).

### 3. Non-Functional Constraints & Guardrails
- **Zero External Dependencies**: Must run on Python standard library alone (`json`, `re`, `datetime`, `urllib.request`, `pathlib`).
- **30-Minute Execution Budget**: Shared `ExecutionBudget` timer wrapping all network requests and fallback pipelines.
- **Strict Baseline Preservation**: Any live optimization must maintain 100% backward compatibility and protected baseline scores.

## Developer Decision & Integration

The developer codified these breakdowns into milestone specifications, created mock fixtures matching `schema.json`, and established automated pytest assertions verifying each requirement before implementing business logic.

