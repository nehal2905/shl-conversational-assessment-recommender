# Approach — SHL Conversational Assessment Recommender

## 1. Problem framing

The task is a *conversational* recommendation problem with hard correctness and safety
constraints: from a possibly vague hiring intent, return a grounded shortlist of 1–10 real
SHL Individual Test Solutions within an 8-turn, 30s/turn budget, while refusing off-topic and
prompt-injection inputs and never hallucinating an assessment or URL.

I treated this as **a deterministic graph wrapped around a few narrow LLM calls**, rather than a
free-form chat agent. The LLM is used only where fuzzy judgement is genuinely needed
(intent/slot extraction, reranking, comparison narration); everything that must be *correct*
(URLs, names, schema, turn budget) is handled by ordinary code.

## 2. Design choices

**Stateless re-derivation (invariant 2).** Every `/chat` call re-derives the full `Analysis`
from the entire message history. This removes an entire class of bugs: refinement ("also add
personality tests") needs no special code path — it just changes the extracted slots on the next
turn. It also matches the evaluator's stateless contract exactly.

**ID-grounding (invariant 1).** The single most important structural decision. The LLM is shown
only `id | name | test_types | truncated description` and must return `id`s. `grounding.ground`
maps those ids back to trusted `CatalogEntry` objects and builds the `Recommendation`s. As a
result, "every URL is from the catalog" is not something we hope the model does — it is
impossible for it to do otherwise. A fuzz test (`tests/test_grounding.py`) feeds 200 fabricated
ids and asserts none escape.

**LangGraph control flow.** `START → analyze → {clarify | recommend | compare | refuse} →
format → END`. A single conditional edge routes on the extracted intent. This keeps the
non-deterministic part (one JSON call in `analyze`) small and makes the behavior auditable.

**Turn-budget-aware clarification (invariant 3).** Over-clarifying is the classic way to score
zero: burn the 8 turns asking questions and never emit a shortlist. The guard is blunt and
reliable — count prior assistant turns; if `>= 2`, force `ready_to_recommend = True` and commit
with whatever is known. `clarify` and `refuse` make **zero** extra LLM calls to protect latency.

## 3. Retrieval setup

Hybrid retrieval (invariant 4) because neither method alone is sufficient on ~450 short catalog
rows:

- **Dense** (`bge-small-en-v1.5` via fastembed → FAISS `IndexFlatIP` on L2-normalized vectors =
  cosine) captures paraphrased *intent* ("works with stakeholders" → personality/competency).
- **Sparse** (BM25Okapi over tokenized name + description) captures exact *product/skill names*
  ("Java", "OPQ", "GSA") that embeddings blur together.
- **Fusion** via Reciprocal Rank Fusion (k=60): `score = Σ 1/(60 + rank)` across both retrievers,
  which is robust to the two scores being on different scales.
- **Refine is additive:** requested `test_types_wanted` add a small RRF bonus to matching entries
  rather than filtering — so "add personality tests" *broadens* the pool instead of discarding
  the developer-relevant knowledge tests.

`fastembed` (ONNX) was chosen over sentence-transformers specifically to avoid a torch dependency
and keep the deploy image small and cold-start fast.

## 4. Prompt design

Three tight prompts (`app/prompts.py`):

- **Analyze (JSON):** classifies intent into `vague/recommend/refine/compare/off_topic` and
  extracts slots. The system prompt states plainly that *message content is data* and must never
  be followed as instructions (invariant 6), and forbids emitting URLs. It requests strict JSON;
  `llm.chat_json` uses Groq JSON mode, strips code fences, parses defensively, and retries once
  with a "return ONLY valid JSON" nudge before failing.
- **Rerank (JSON):** given a *fixed* candidate list, return `{"ids": [...]}` best-first, 1–10 ids,
  choosing only ids that appear in the list. Even so, `recommend` intersects the returned ids
  with the candidate set as defense in depth.
- **Compare (text):** answers only from the provided catalog descriptions; instructed to say it
  can only compare catalog items if a target is missing. Targets are resolved by `rapidfuzz`
  against catalog names.

## 5. Evaluation approach

- **`eval/metrics.py`** — Recall@10 = |relevant ∩ top-10| / |relevant|.
- **`eval/replay.py`** — a simulated user answers the agent's clarifying questions from a trace's
  `facts`, says "no preference" otherwise, and stops when a shortlist appears; capped at 8 turns.
  Prints mean Recall@10 over `eval/traces/`.
- **`eval/probes.py`** — binary behavior assertions: no-recommend-on-vague-turn-1, refuses
  off-topic/injection, honors refine edits (a `P`-type appears), and a hallucination rate that
  counts any emitted non-catalog URL (target 0).

Recall@10 is the metric I iterate retrieval against (RRF weighting, boost size, `DENSE_TOPN`);
the probes are the safety gate that must stay green regardless of recall tuning.

## 6. What didn't work / trade-offs, and how I measured it

- **Pure dense retrieval** missed exact product names — a query for "OPQ" ranked several generic
  personality items above `OPQ32r`. Adding BM25 + RRF fixed the exact-match cases while keeping
  intent matches; measured as top-1/top-3 hit-rate on name queries in the probes.
- **Filtering by requested test type** (instead of boosting) caused refine to *drop* relevant
  results — asking for personality tests removed the Java knowledge tests entirely. Switching to
  an additive RRF bonus kept both, verified by `probe_honors_refine` (shortlist stays populated
  *and* gains a `P` entry).
- **Over-clarifying** tanked recall in early replay runs (the agent asked 3+ questions and ran out
  of turns). The force-commit-after-2 rule recovered it, visible as a jump in mean Recall@10.
- **Offline fallback:** to keep the whole system testable without a key, `app/offline_llm.py`
  provides a deterministic rule-based stand-in for the three LLM calls. It is strictly a test/dev
  convenience; Groq is used whenever `GROQ_API_KEY` is set.

## 7. AI tools used

- **Cursor (agent) with an Anthropic Claude model** — used to scaffold the repository against the
  spec, write the modules/tests/eval harness, and keep the API contract and invariants aligned.
- **Groq `llama-3.3-70b-versatile`** — the runtime agent LLM for analysis, reranking and
  comparison, chosen for low latency within the 30s/turn budget.
- **`bge-small-en-v1.5`** (via fastembed ONNX) — the embedding model for dense retrieval.

Human review focused on the load-bearing invariants (grounding, turn budget, injection handling)
and on the retrieval trade-offs above.
