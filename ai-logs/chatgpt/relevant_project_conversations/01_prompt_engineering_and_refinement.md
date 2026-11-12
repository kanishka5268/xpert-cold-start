# ChatGPT Consultation: Prompt Engineering & Refinement

## Role of ChatGPT

ChatGPT was consulted to refine the in-context prompts passed to Gemini in `src/interpretation/prompts.py`. Effective prompting was critical because open-ended instructions caused semantic misclassifications, hallucinations of ungrounded message IDs, and schema validation failures.

## Key Consultation Topics

### 1. Goal Area Semantic Disambiguation
- **Problem**: In early live LLM runs, goals like the "Brightside launch milestone" were classified as `career` instead of `work`, and half-marathon fitness goals were categorized as `personal_growth` instead of `health`.
- **Discussion with ChatGPT**:
  - Analyzed the semantic boundaries expected by the ground-truth evaluator (`check.py`):
    - `work`: Specific team, company, project deliverable, or product milestone (e.g., Brightside MVP launch, product shipping).
    - `career`: High-level individual professional path, promotions, job transitions, or resumes.
    - `health`: Physical conditioning, running/marathon training, diet, medical milestones.
    - `personal_growth`: Non-physical personal hobbies, habits, reading goals, language learning.
  - Refined prompt examples and taxonomy guidelines in `src/interpretation/prompts.py` to give the live model crystal-clear discriminatory rules.

### 2. Snippet-Independent Interpretation Rule (Phase C Corrections)
- **Problem**: When disjoint batching grouped multiple non-contiguous snippets from the same chat into one prompt, the LLM occasionally cross-referenced evidence across snippets, linking a commitment in snippet 2 to an acknowledgment in snippet 1.
- **Discussion with ChatGPT**:
  - Formulated explicit mandatory isolation rules for `CHAT_SYSTEM_PROMPT`:
    1. Every snippet must be evaluated as an independent temporal block.
    2. `closed_by` and `due_anchor_msg` message IDs must belong strictly to the snippet in which the candidate appeared.
    3. References outside the snippet boundary are strictly invalid.

### 3. Negative Constraints & Conservative Closure Guardrails
- **Problem**: LLMs exhibit a known bias toward eager closure (assuming any positive word like "Sure" or "Thanks" closes a commitment).
- **Discussion with ChatGPT**:
  - Drafted strict negative constraints for the prompt:
    - "Do NOT assume an obligation is closed merely because someone acknowledged receipt or said 'thanks'."
    - "Require concrete proof of delivery or explicit confirmation (e.g., 'Sent', 'Done', 'Deployed')."
    - "If in doubt between open and closed, mark as open (conservative closure principle)."

## Developer Decision & Integration

The developer reviewed ChatGPT's drafted prompt clauses, pruned unnecessary verbosity to conserve prompt tokens, and directly incorporated the distilled rules into `src/interpretation/prompts.py`.

