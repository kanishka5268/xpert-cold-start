# ChatGPT Consultation: Structured Antigravity Instruction Packets

## Role of ChatGPT

To prevent autonomous agent drift, uncontrolled scope expansion, or code regressions during complex refactorings, the developer collaborated with ChatGPT to author tightly bounded **instruction packets** before handing tasks over to Antigravity.

## Anatomy of a Structured Instruction Packet

Every prompt provided to Antigravity followed a standardized architectural structure developed and refined with ChatGPT:

```text
# [Phase / Milestone Title]

## 1. Context & Objective
- Concise explanation of the problem and the single engineering goal.
- Explicit reference to prior forensic reports or baseline scores.

## 2. Protected Constraints & Immutability Rules
- List of frozen files that must NOT be touched (e.g., "src/loops/extractor.py is frozen").
- Mandatory negative constraints:
  - Do not introduce third-party libraries.
  - Do not modify run.sh or CLI contracts.
  - Do not read gold answer files from production code.
  - Do not execute live API calls during unit testing.
  - Preserve protected output hashes and baseline caches.

## 3. Strict File Scope
- Explicit list of files permitted to be edited (e.g., "Modify ONLY src/interpretation/parser.py and tests/test_parser.py").

## 4. Required Implementation Details
- Precise description of the functions, classes, and algorithmic changes required.
- Concrete edge cases to handle (e.g., "handle horizon is None without dropping candidate").

## 5. Verification & Acceptance Criteria
- Step-by-step verification commands:
  1. python -m py_compile <modified_file>
  2. pytest tests/<target_test> -v
  3. pytest tests/ -q (confirm all 1,000+ tests pass)
  4. SHA-256 hash check on baseline outputs.
```

## Benefits of this Structured Approach

1. **Elimination of Agent Drift**: Antigravity was prevented from making "helpful" refactorings to unrelated modules (e.g. touching `resolver.py` when only `prompts.py` needed updates).
2. **Deterministic Quality Control**: Clear verification gates ensured every step was validated before proceeding to the next milestone.
3. **Budget & Rate-Limit Protection**: Enforcing offline testing prevented accidental API credit depletion during intermediate iterations.

## Developer Decision & Integration

The developer retained complete editorial authority over all instruction packets, reviewing ChatGPT's drafted specifications against project priorities before dispatching them to Antigravity.

