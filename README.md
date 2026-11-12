# Cold Start Challenge: Personal Assistant Backend

A high-performance, robust WhatsApp conversation understanding backend developed for the **Cold Start Challenge** ([goals.xpert.chat/w/intern](https://goals.xpert.chat/w/intern)).

The system processes raw, multi-chat WhatsApp export histories and initial user signup profiles to automatically construct a structured personal intelligence graph. It identifies key contacts and social circles, resolves organizational affiliations, maps direct and group conversations, detects long-term personal and professional goals, and tracks actionable open loops (commitments, deadlines, and follow-ups) with grounded message evidence.

---

## 1. Project Overview

When a user connects their WhatsApp data to an AI personal assistant, the system faces a "cold start" problem: it must interpret weeks or months of unstructured conversational context across multiple one-on-one and group chats without prior history.

This backend performs end-to-end extraction across five interconnected domains:
1. **People**: Resolves participant identities and categorizes social relationships into standardized circles (`spouse`, `family`, `friend`, `colleague`, `client`, `vendor`, `staff`).
2. **Organisations**: Discovers workplace companies, residences, societies, and institutions, classifying user roles and organizational sides (`home` vs. `work`).
3. **Chats**: Maps each direct conversation to its canonical participant and associates group chats with their governing organization.
4. **Goals**: Identifies high-level user aspirations across areas (`work`, `health`, `finances`, `personal_growth`, etc.) with message-grounded evidence and temporal horizons.
5. **Open Loops**: Discovers active obligations and promises, tracks lifecycle state (`open`, `done`, `dropped`) under strict conservative closure rules, identifies ownership (`me_to_them` vs. `them_to_me`), and generates proactive help recommendations (`remind`, `assist`, `leave`).

---

## 2. Technology Stack

- **Runtime**: Python 3.9+ (developed and verified on Python 3.13)
- **Core Execution Architecture**: **Zero third-party runtime dependencies**. The production pipeline (`solution.py`, `run.sh`, `llm.py`, `check.py`, and all packages under `src/`) runs exclusively on the Python standard library (`json`, `urllib.request`, `re`, `datetime`, `pathlib`, `os`, `sys`, `time`, `typing`).
- **LLM Integration**: OpenAI-compatible HTTP client implemented in standard-library `urllib.request`, with configurable base URL, authorization headers, exponential backoff, and local usage logging (`usage.jsonl`).
- **Development & Testing Dependencies** (confined to testing/offline validation in `requirements.txt`):
  - `python-dotenv==1.2.4`: Environment variable loading from `.env`.
  - `jsonschema==4.26.0`: Strict JSON Schema verification against `schema.json`.
  - `pytest==9.2.0`: Automated test execution across 1,000+ unit and integration tests.
- **Shell / Automation**: POSIX-compliant Bash wrapper (`run.sh`) compatible with Linux, macOS, and Windows Git Bash.

---

## 3. Project Structure

```text
cold-start/
├── run.sh                      # Official one-command execution entry point (max 30 min)
├── solution.py                 # Primary pipeline orchestrator connecting all extraction stages
├── llm.py                      # Standard-library OpenAI-compatible HTTP chat client with usage logging
├── schema.json                 # Formal JSON Schema specifying required output format
├── check.py                    # Official scoring and validation tool for output verification
├── requirements.txt            # Development and testing dependencies (pytest, jsonschema, dotenv)
├── src/                        # Core production pipeline packages
│   ├── ingestion/              # Ingests chats.jsonl, messages.jsonl, and seed.json
│   ├── preprocessing/          # Builds conversational graph context and sender metadata
│   ├── entities/               # Extracts and resolves people, social circles, and organisations
│   ├── goals/                  # Detects goal milestones and temporal horizons with evidence grounding
│   ├── loops/                  # Heuristic candidate extraction, lifecycle authority, and conservative closure
│   ├── interpretation/         # Selective hybrid LLM routing, token-bucket TPM limiter, batcher, and parser
│   └── extraction/             # Linguistic pattern matching and candidate window extraction
├── tests/                      # Comprehensive test suite (1,000+ unit, integration, and regression tests)
├── example/                    # Official baseline example dataset (chats, messages, seed, answers)
├── ai-logs/                    # Complete AI interaction logs, Antigravity trajectories, and ChatGPT records
│   ├── README.md               # Consolidated documentation of AI methodology and provenance
│   └── development_log.md      # Chronological engineering log across all development milestones
└── reports/                    # Forensic audit reports, architectural trade-offs, and benchmark evaluations
```

---

## 4. Setup Instructions

### 4.1 Clone Repository and Prepare Virtual Environment

```bash
# Clone the repository
git clone <repository-url>
cd cold-start

# Create virtual environment
python3 -m venv .venv

# Activate on Linux / macOS:
source .venv/bin/activate

# Activate on Windows (Command Prompt / PowerShell):
.venv\Scripts\activate
# Or in Windows Git Bash:
source .venv/Scripts/activate
```

### 4.2 Install Dependencies

Install the development and testing dependencies:

```bash
pip install -r requirements.txt
```

### 4.3 Configure LLM Environment Variables

The live pipeline communicates with Groq's OpenAI-compatible API endpoint. Create a `.env` file in the project root:

```bash
LLM_BASE_URL="https://api.groq.com/openai/v1"
LLM_API_KEY="your-groq-api-key-here"
LLM_MODEL="openai/gpt-oss-120b"
```

> **Security Note**: Never commit `.env`, API keys, or private bearer tokens to version control. The `.env` file is permanently ignored by `.gitignore`.

If running in offline mode without network calls, the pipeline automatically detects the absence of credentials or the `--offline` flag and executes deterministic heuristic fallbacks.

---

## 5. Execution

### 5.1 One-Command Execution Interface

The challenge specifies a strict one-command launcher interface:

```bash
./run.sh <input_dir> <output.json>
```

`run.sh` automatically locates the virtual environment Python interpreter (on Linux, macOS, or Windows) and dispatches execution to `solution.py`.

### 5.2 Running the Datasets

**Running the Example Dataset**:
```bash
./run.sh example out.json
```

**Running the Development Dataset** (when unpacked into `data/dev`):
```bash
./run.sh data/dev out.json
```

**Running Directly via Python**:
```bash
# Standard live run
python3 solution.py example out.json

# Deterministic offline run (no API calls)
python3 solution.py example out.json --offline
```

The output JSON file is written to the specified path (`out.json`), fully validated against `schema.json`.

### 5.3 Evaluating Output with `check.py`

Evaluate the generated output against ground truth using the official checker:

```bash
python3 check.py example out.json
```

If the input directory contains an `answers/answers.json` key, `check.py` validates the schema structure and prints a detailed score breakdown across all evaluated categories.

---

## 6. Input and Output Specifications

### 6.1 Input Directory Structure
Each input dataset folder must contain:
- `chats.jsonl`: Chat metadata containing `chat_id`, `name`, `is_group`, and `participants` (where the user is `"ME"`).
- `messages.jsonl`: Chronological message records containing `id`, `chat_id`, `sender_id`, `sender_name`, `ts` (Unix epoch seconds), `kind` (`text`, `voice_transcript`, `image_caption`, `document`), `body`, and optional `quoted_id`.
- `seed.json`: Initial signup profile containing known social circles, organization affiliations, and pre-seeded goals.
- `answers/answers.json`: (Development and evaluation sets only) Ground-truth annotations for scoring.

### 6.2 Output Format
The resulting output JSON contains five top-level keys adhering to `schema.json`:
- `"people"`: List of contacts with resolved participant IDs, social circle assignments, organization links, and dependent relationships (`looks_after`).
- `"organisations"`: Discovered workplace, residential, and social organizations with names, types, role classifications, and sides.
- `"chats"`: Mapping of direct chats to contact IDs and group chats to organization IDs.
- `"goals"`: Long-term milestones with category areas, temporal horizons (`3mo`, `1yr`, `3yr`, `5yr`), and supporting evidence message IDs.
- `"open_loops"`: Actionable commitments with ownership (`ME` or contact), source message IDs, due dates, statuses (`open`, `done`, `dropped`), closure message references, and proactive help recommendations (`assist`, `remind`, `leave`).

---

## 7. Runtime and Evaluation

- **30-Minute Execution Limit**: The challenge mandates that `./run.sh` must complete within **30 minutes** wall-clock time. The pipeline includes an integrated `ExecutionBudget` controller (capped at 27.5 minutes) and a proactive token-bucket TPM rate limiter (`rate_limiter.py`) to guarantee timely completion without crashing or timing out.
- **Evaluation Rubric**: `check.py` scores 74 points of the 100-point total:
  - Circles: 12 points
  - Organisations: 8 points
  - Chats: 8 points
  - Goals: 8 points (F1 matching on area and shared evidence)
  - Open Loops Found: 14 points (precision and recall on source message grounding)
  - Open Loops Closed Right: 14 points (correct status identification; false closures penalized)
  - Where to Help: 10 points (appropriate action recommendations for open loops)
- The remaining points are evaluated on execution cost, write-up quality, and one-command execution fidelity.

---

## 8. Important Notes & Operational Safeguards

- **Data Privacy & Exclusions**: Raw conversation datasets (`data/dev/`, `data/`) and gold answers are strictly excluded by `.gitignore` and must never be committed.
- **Zero-Dependency Core Guarantee**: Do not introduce third-party imports into `src/` or `solution.py`. The solution is engineered to execute cleanly on any standard Python 3.9+ Linux/macOS/Windows host.
- **Preserved CLI Contract**: Do not alter the flags or signature of `run.sh`. It must always accept `./run.sh <input_dir> <output.json>`.
- **Conservative Closure Principle**: The pipeline treats unconfirmed obligations as open by default; obligations are only marked closed when unambiguous, grounded evidence confirms completion.
