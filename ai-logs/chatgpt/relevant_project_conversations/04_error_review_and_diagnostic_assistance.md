# ChatGPT Consultation: Error Review & Diagnostic Assistance

## Role of ChatGPT

ChatGPT was consulted to diagnose unexpected test failures, evaluator score drops, and live-vs-offline output discrepancies without modifying production code prematurely.

## Key Consultation Topics

### 1. Goal Horizon `null` Rejection Defect (Phase 6.1 & 6.2)
- **Symptom**: In the Phase 6 live benchmark on `example`, the Goal score dropped from 8/8 to 0/8, even though the live LLM correctly extracted the goals.
- **Diagnostic Discussion with ChatGPT**:
  - Inspected the raw LLM responses recorded in `usage.jsonl` and parser rejection counters.
  - Discovery: The live LLM returned `horizon: null` for three genuine goals (Brightside milestone, marathon training, reading goal) because no explicit date string appeared in the chat.
  - The parser had a rigid validation check that immediately discarded any candidate with `horizon is None` as `invalid horizon`.
  - Formulated the fix: Implement resilient horizon normalization (`normalize_goal_horizon`) that infers plausible default horizons (e.g. `near_term` / `medium_term`) from message context or preserves valid goals with unknown horizons rather than dropping them.

### 2. Decision-Question Help-Action Precedence (Phase 6.2)
- **Symptom**: Where-to-help score dropped from 10/10 to 8/10 on the live benchmark.
- **Diagnostic Discussion with ChatGPT**:
  - Traced evaluator matching in `check.py`: the ground truth classified open decision questions ("Should we proceed with Option A or B?") as needing a reminder/decision action.
  - The live LLM had classified these under `step_in` or `leave` because of ambiguous priority wording.
  - Formulated the fix: Adjusted prompt guidance and deterministic resolution precedence to recognize open decision blocks as priority response actions.

### 3. Live vs. Offline Development Corpus Discrepancy (Phase 6.3)
- **Symptom**: On the 22-minute live run of the full `dev` dataset (29 API calls), the score was 45.0/56 vs. 46.2/56 offline.
- **Diagnostic Discussion with ChatGPT**:
  - Systematically audited the exact diff between `output.json` and `phase6_2_live_dev.json` against `data/dev/answers/answers.json`.
  - Isolated the 4 distinct contributing factors:
    1. **1 False Negative (C73 / M01874)**: The live LLM filtered out a subtle obligation that the deterministic regex had captured.
    2. **5 False Positives (C09, C10, C60, C82, C94)**: The live LLM extracted casual conversational commitments (e.g. "I'll think about it") that the ground truth did not score.
    3. **Status Divergences (C10 / M02581, C110 / M03375)**: Premature closure in C10 ("Done" from third-party) and missed closure in C110.
    4. **Budget Boundary Drops (C10, C34)**: Proactive token-bucket rate limiter rejected two oversized snippets near the end of the run to stay strictly within the 30-minute time cap.

## Developer Decision & Integration

The developer implemented targeted parser resilience and precedence updates in `src/interpretation/parser.py` and `src/interpretation/prompts.py` while keeping extraction heuristics frozen, verifying the changes against isolated test suites before re-running benchmarks.

