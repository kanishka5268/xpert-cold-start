# ChatGPT Consultation: Architectural & Trade-Off Analysis

## Role of ChatGPT

ChatGPT was used as an architectural sounding board to analyze alternative system designs, evaluate cost-latency trade-offs, and assess risk mitigation strategies under API rate limits.

## Key Consultation Topics

### 1. Architectural Strategy Comparison: Scenario A vs. B vs. C
- **Context**: The challenge required optimizing live LLM performance while respecting severe Gemini API quotas (Tier-1 TPM limits) and maintaining a 30-minute runtime cap.
- **Discussion with ChatGPT**:
  - **Scenario A (Full Live LLM)**: Send all chats and messages to the live model.
    - *Risk*: Guaranteed to exceed TPM rate limits, leading to frequent 429 throttling, timeouts, and incomplete runs on large datasets like `dev`.
  - **Scenario B (Pure Offline Deterministic)**: Rely entirely on regex and static heuristics without any live calls.
    - *Risk*: Robust and fast, but leaves subtle semantic nuances uncaptured and fails to demonstrate live LLM agentic capabilities.
  - **Scenario C (Selective Hybrid Routing)**: Heuristic pre-filtering and scoring selects only ambiguous or high-value candidate snippets for live LLM interpretation; deterministic logic handles clear-cut patterns and provides guaranteed offline fallback.
    - *Advantage*: Achieves optimal token efficiency, remains safely within the TPM budget, and guarantees that if the LLM is offline or throttled, the deterministic fallback achieves full baseline coverage.
- **Developer Decision**: Adopted **Scenario C**.

### 2. Rate-Limiting Strategy: Proactive Token-Bucket vs. Reactive Backoff
- **Problem**: Reactive retry upon encountering HTTP 429 errors wastes round-trip time and can cause cascade failures when multiple requests retry simultaneously.
- **Discussion with ChatGPT**:
  - Evaluated sliding-window vs. token-bucket algorithms.
  - Designed a client-side proactive token bucket in `src/interpretation/rate_limiter.py`:
    - Tracks consumed input and estimated output tokens per minute.
    - Enforces proactive sleep delays *before* dispatching a request if the request would breach the 1,000,000 TPM ceiling.
    - Incorporates conservative safety margins (e.g. 80% watermark).

### 3. Snippet Batching: Disjoint Multi-Snippet Packing
- **Problem**: One API call per candidate snippet creates high HTTP overhead and consumes system prompt tokens repeatedly.
- **Discussion with ChatGPT**:
  - Designed disjoint multi-snippet batching in `src/interpretation/batcher.py`:
    - Packs non-overlapping candidate message windows from the same conversation into a single prompt up to a configurable token limit.
    - Preserves snippet boundaries with unique `snippet_id` tags.
    - Dramatically reduced total live requests (from dozens down to 7 requests on `example` and 29 requests on `dev`).

## Developer Decision & Integration

The developer implemented Scenario C, the token bucket limiter, and the disjoint batcher in `src/interpretation/`, verifying each component with standalone unit tests before integrating into `solution.py`.

