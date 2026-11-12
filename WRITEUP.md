# Technical Write-up — Cold Start Challenge

## 1. What I Focused On, and Why

The Cold Start Challenge tasks an intelligent assistant with interpreting unstructured WhatsApp conversation histories for a user ("ME" / Arjun) to reconstruct personal and professional context across five domains: **People & Circles**, **Organisations**, **Chats**, **Goals**, and **Open Loops**.

Rather than treating this as an unconstrained generative task, I focused on building an **evidence-grounded, modular pipeline with deterministic safeguards and bounded LLM orchestration**. Core engineering priorities were:

1. **Reliable Context Extraction:** Capturing commitments, questions, and goals across 1:1 and group chats without losing surrounding conversational context.
2. **Evidence-Grounded Predictions:** Guaranteeing every goal, entity, loop, and closure references authentic dataset identifiers (`source_msg`, `evidence_message_ids`, `closed_by`) without hallucinating IDs or dates.
3. **Correct Entity and Structure Mapping:** Resolving participants, social circles, workplace organisations, and chat links while preserving pre-seeded profile entities.
4. **Reliable Ownership and Lifecycle Resolution:** Disambiguating assignees (`me_to_them` vs. `them_to_me`), anchoring relative due dates to message timestamps, and enforcing conservative closure rules.
5. **Schema Conformance and Pipeline Stability:** Ensuring 100% participant coverage and strict adherence to `schema.json` under an integrated execution budget designed to keep execution within the 30-minute challenge limit.

---

## 2. Approach

The pipeline is implemented in standard-library Python (3.9+ compatible) with zero external runtime dependencies, operating across seven coordinated stages:

```text
Raw Data (JSONL) ──► Ingestion & Preprocessing ──► Candidate Evidence Extraction
                                                             │
Final output.json ◄── Assembly & Validation    ◄── LLM Interpretation & Parsing
                              ▲                              │
                              └── Resolution Engines ◄───────┘
                                  (Entities, Goals, Loops)
```

- **Ingestion and Preprocessing (`src/ingestion/`, `src/preprocessing/`):** Ingests `chats.jsonl`, `messages.jsonl`, and `seed.json`, validating schemas and normalizing timestamps to UTC. Constructs a `ConversationContext` indexing messages chronologically with chat timelines and participant dossiers.
- **Candidate Evidence Extraction (`src/extraction/`):** High-recall linguistic extractors isolate candidate messages for commitments, questions, deadlines, and goals, filtering out conversational banter, news feeds, and non-actionable idioms.
- **LLM Interpretation and Structured Parsing (`src/interpretation/`):** Formats candidate windows into structured prompts for `openai/gpt-oss-120b` via Groq. Enforces JSON output and snippet independence. A structured response parser with validation checks enums and extracts responses into typed dataclasses.
- **People, Organisation, and Chat Resolution (`src/entities/`):** Classifies participants into standardized circles (`spouse`, `family`, `friend`, `colleague`, `client`, etc.) using interaction cues. Deduplicates organisations while preserving seed entities. Maps direct chats to people and group chats to organisations, defaulting unclassified contacts to "friend" for 100% coverage.
- **Goal Extraction (`src/goals/`):** Consolidates candidate statements into personal, health, and professional aspirations across ten areas, aligning temporal horizons (`3mo`, `1yr`, `3yr`, `5yr`) with evidence message grounding.
- **Open-Loop Ownership, Lifecycle, and Help Resolution (`src/loops/`):** Resolves ownership via participant mapping, anchors relative dates (*"by Friday"*) to message timestamps, and applies the **Conservative Closure Principle** (tasks remain open unless explicit evidence confirms completion). Requires strict adjacency for bare completion markers (*"Done"*), and assigns help actions (`remind`, `assist`, `leave`) based on ownership and reversibility.
- **Validation and Output Assembly (`solution.py`):** Reconciles domain outputs, verifies structure against `schema.json`, and writes `output.json`. A monotonic execution budget controller oversees network operations to keep execution within the 30-minute challenge limit.

---

## 3. What It Costs

### Token Usage and Financial Estimates
Cost was measured during live execution with `openai/gpt-oss-120b` via Groq, recording token counts in `usage.jsonl`:

| Dataset | Messages | API Calls | Measured Prompt Tokens | Measured Completion Tokens | Total Measured Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Example** | 29 | 7 | 31,376 | 7,445 | 38,821 |
| **Development** | 4,502 | 29 | 134,662 | 23,383 | 158,045 |

- **Cost Analysis (Development Set):**
  - At published Groq pricing for 120B-class models ($0.15 / 1M prompt tokens, $0.60 / 1M completion tokens):
    $$\text{Estimated Total Cost} \approx (0.1347 \times \$0.15) + (0.0234 \times \$0.60) \approx \$0.020 + \$0.014 \approx \mathbf{\$0.034}$$
  - Normalized cost estimate: **$\approx \$0.008$ per 1,000 conversation messages** (across 4,502 verified messages).
- **Cost Drivers:**
  - **Prompt Tokens (~85%):** Multi-turn conversational context windows required for pronoun resolution, quoted message linkages, and commitment disambiguation.
  - **Completion Tokens (~15%):** Structured JSON records containing extracted obligations, evidence IDs, and help metadata.

*Note: Token counts and request numbers are measured figures from execution logs; monetary figures are calculated estimates based on published token pricing.*

---

## 4. Where It Is Wrong

Evaluation against ground-truth labels and validation testing reveals five key failure modes:

1. **Implicit and Colloquial Commitments:** Intentions phrased without explicit action verbs (e.g., *"I might take a look later"*) are filtered out by pre-filtering heuristics to prevent noisy false positives, resulting in missed commitments.
2. **Ambiguous Multi-Party Ownership:** In group chats where tasks are discussed collectively without tagging specific names, assigning ownership between the user and group members remains prone to misattribution.
3. **Informal and Multilingual Phrasing:** Rapid code-switching (e.g. English-Hindi colloquialisms) occasionally bypasses English-oriented grammatical heuristics.
4. **Entity Disambiguation for Infrequent Contacts:** Participants appearing only once or twice in group chats lack sufficient conversational evidence for precise circle classification, requiring default baseline assignments.
5. **False Closures vs. False Openings:** In group threads where acknowledgments occur without quoted message IDs, conservative closure leaves tasks open to avoid penalizing false closures, occasionally missing valid multi-party completions.

---

## 5. Final Evaluation Results

Final evaluation results using the official evaluator (`check.py`):

| Dataset | Evaluation | Score |
| :--- | :--- | ---: |
| **Example** | Live | **74.0 / 74** |
| **Development** | Evaluated | **44.7 / 56** |

### Development Score Breakdown

| Category | Score | Possible Points |
| :--- | :---: | :---: |
| **Circles** | 12.0 | 12 |
| **Organisations** | 8.0 | 8 |
| **Chats** | 8.0 | 8 |
| **Open Loops Found** | 10.9 | 14 |
| **Open Loops Closed Right** | 5.9 | 14 |
| **Goals** | *Unlabelled in dev set* | — |
| **Where to Help** | *Unlabelled in dev set* | — |
| **Total** | **44.7** | **56** |

*Note: Goals and Where-to-Help are not labelled in the development evaluation set. Evaluator scores reflect the official rubrics implemented in `check.py`.*
