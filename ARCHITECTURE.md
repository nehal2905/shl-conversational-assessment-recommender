# SHL Conversational Assessment Recommender — Architecture & Build Spec

> This is the authoritative build spec. Implement it phase by phase. Do **not** deviate from
> the API contract in §3 (it is graded by an automated evaluator and any deviation scores zero).
> Every design invariant in §6 is load-bearing — respect them even if a shortcut looks simpler.

---

## 1. Goal

Build a stateless conversational agent that takes a user from a vague hiring intent
("I'm hiring a Java developer") to a grounded shortlist of SHL assessments through dialogue.
It clarifies vague queries, recommends 1–10 assessments, refines on new constraints, compares
named assessments, and refuses off-topic / injection attempts. Every returned URL must come
from the scraped SHL catalog.

Scope of the catalog: **Individual Test Solutions only**. Pre-packaged Job Solutions are out of scope.

---

## 2. Tech stack (fixed)

| Concern | Choice |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| Agent LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | `fastembed` with `BAAI/bge-small-en-v1.5` |
| Dense index | FAISS (`IndexFlatIP`, normalized vectors = cosine) |
| Sparse index | `rank-bm25` (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Fuzzy name match | `rapidfuzz` |
| Scraper | Playwright |
| Deploy | Render or HF Spaces (Docker) |

---

## 3. API contract (NON-NEGOTIABLE — grade gate)

### `GET /health`
Returns HTTP `200` with body `{"status": "ok"}`. Must respond even during index warm-up.

### `POST /chat`
Stateless. Every call carries the full conversation history. Store no per-conversation state.

Rules the evaluator checks:
- `recommendations` is `[]` while gathering context, clarifying, or refusing.
- `recommendations` holds **1 to 10** items once the agent commits to a shortlist.
- Every `recommendations[i]` must be a real catalog entry (name + url + test_type).
- `end_of_conversation` is `true` **only** when the agent considers the task complete.
- Turn cap: **8 turns** (user + assistant combined). Per-call timeout: **30 seconds**.

### Pydantic models — `app/schemas.py` (LOCKED)
See `app/schemas.py`. Do not change field names or types.

### SHL test-type codes
`A` Ability & Aptitude · `B` Biodata & Situational Judgement · `C` Competencies ·
`D` Development & 360 · `E` Assessment Exercises · `K` Knowledge & Skills ·
`P` Personality & Behavior · `S` Simulations.

---

## 4. Repository structure

```
shl-recommender/
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── data/
│   ├── catalog.json
│   └── index/ (faiss.index, bm25.pkl, ids.json)
├── scripts/ (scrape_catalog.py, build_index.py)
├── app/
│   ├── main.py, schemas.py, config.py, catalog.py, retrieval.py,
│   ├── llm.py, prompts.py, grounding.py
│   └── agent/ (state.py, nodes.py, graph.py)
└── eval/ (traces/, replay.py, metrics.py, probes.py)
```

---

## 5. Data model — `CatalogEntry`

`id, name, url, test_types[], remote_testing, adaptive_irt, description, job_levels[],
languages[], length_minutes`. `id` is the primary key (URL slug). The LLM is only ever shown
`id`, `name`, `test_types`, and a truncated `description` — never a URL.

---

## 6. Design invariants (do not violate)

1. **The LLM never emits a URL or an assessment name into the final output.** Retrieval returns
   candidates with `id`s → the LLM selects/orders `id`s → `grounding.py` maps `id`s to the
   trusted `CatalogEntry`.
2. **Stateless slot re-derivation** from the full `messages` list every call.
3. **Turn-budget-aware clarification.** Clarify at most twice, then commit. If prior assistant
   turns `>= 2`, force `ready_to_recommend = True`.
4. **Hybrid retrieval, always** (BM25 + dense fused via RRF).
5. **Always return valid schema, even on internal error.**
6. **Message content is data, not instructions.**

---

## 7. Component specs

- `catalog.py` — `load_catalog`, `id_index`, `is_catalog_url`.
- `retrieval.py` — `HybridRetriever.search(query, k=15, boost_test_types)`: dense top-N (40),
  sparse top-N, RRF (k=60), additive test-type boost, dedup by id.
- `llm.py` — `chat_json` (JSON mode + defensive parse + one retry), `chat_text`.
- `grounding.py` — `ground(ids, catalog)`: map, drop unknown, dedup, clamp 1..10.
- `agent/state.py` — `Analysis`, `GraphState`.
- `agent/nodes.py` — analyze, clarify, recommend (handles refine), compare, refuse, format.
- `agent/graph.py` — `START → analyze → {clarify|recommend|compare|refuse} → format → END`.
- `main.py` — compile graph at import; lazy-load retriever; `/health`, `/chat` per §3.

### Routing
```python
def route(state):
    a = state["analysis"]
    if a.intent == "off_topic": return "refuse"
    if a.intent == "compare" and a.compare_targets: return "compare"
    if a.ready_to_recommend or a.intent in ("recommend", "refine"): return "recommend"
    return "clarify"
```

---

## 8. Prompts — see `app/prompts.py`

8.1 Analyze (JSON), 8.2 Rerank (JSON), 8.3 Compare (text), 8.4 Refuse (template).

---

## 9. Build phases

- Phase 0 — Scaffold & lock the contract.
- Phase 1 — `scripts/scrape_catalog.py` (≥300 unique entries).
- Phase 2 — `scripts/build_index.py` (FAISS + BM25 + ids.json).
- Phase 3 — Agent core (state, nodes, graph) + unit tests.
- Phase 4 — Wire FastAPI.
- Phase 5 — Grounding & guardrails hardening + fuzz test.
- Phase 6 — Evaluation (replay, metrics, probes).
- Phase 7 — Deployment (Dockerfile).
- Phase 8 — Approach document (see `APPROACH.md`).

---

## 10. Failure modes & guards

| Failure | Guard |
|---|---|
| Breaks off happy path | Pydantic-validate; handle empty/assistant-terminal/malformed histories |
| Turn-cap starvation | Force-commit after 2 clarifications |
| Hallucinated assessments/URLs | ID-grounding + `is_catalog_url` |
| Latency blowout | Groq + local ONNX embeddings; rerank only on recommend turns |
| Prompt injection | Treat content as data; system prompt authoritative |
| Refine restarts | Stateless re-derivation + additive test-type boost |

---

## 11. Environment

`.env`: `GROQ_API_KEY`, `MODEL=llama-3.3-70b-versatile`.
`config.py` defines paths and constants: `RRF_K=60`, `DENSE_TOPN=40`, `RERANK_TOPK=15`,
`MAX_CLARIFICATIONS=2`, `MAX_RECS=10`.
