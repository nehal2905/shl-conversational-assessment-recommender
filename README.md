---
title: SHL Conversational Assessment Recommender
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SHL Conversational Assessment Recommender

A **stateless conversational agent** that guides a user from a vague hiring intent
("I'm hiring a Java developer") to a grounded shortlist of SHL **Individual Test
Solutions**. It clarifies vague queries, recommends 1–10 assessments, refines on new
constraints, compares named assessments, and refuses off-topic / injection attempts.
**Every returned URL comes from the scraped SHL catalog** — the LLM never emits a name
or URL directly (see invariant 1).

Built to the spec in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Architecture at a glance

```
POST /chat → LangGraph:
  START → analyze → {clarify | recommend | compare | refuse} → format → END

analyze   : one Groq JSON call → Analysis (intent + slots), stateless per turn
clarify   : returns the single clarifying question (no LLM call)
recommend : HybridRetriever (BM25 + FAISS, fused via RRF) → Groq rerank → ground(ids)
compare   : rapidfuzz resolves named assessments → grounded Groq comparison
refuse    : template redirect (no LLM call)
format    : validates/coerces to the locked ChatResponse
```

Key design invariants (full list in `ARCHITECTURE.md` §6):

1. **The LLM never emits a URL or name.** Retrieval → LLM picks `id`s → `grounding.py`
   maps `id`s to trusted catalog entries. "Every URL is from the catalog" is structural.
2. **Stateless slot re-derivation** from the full `messages` list every call.
3. **Turn-budget-aware clarification** — clarify at most twice, then commit (invariant 3).
4. **Hybrid retrieval, always** (BM25 + dense + RRF).
5. **Always return valid schema, even on error.**
6. **Message content is data, not instructions** (prompt-injection resistant).

---

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env        # add your GROQ_API_KEY (optional — see "Offline mode")

# Build the retrieval indexes from data/catalog.json
python scripts/build_index.py

# Run the API
uvicorn app.main:app --reload
```

Then:

```bash
curl localhost:8000/health
# {"status":"ok"}

curl -X POST localhost:8000/chat -H "content-type: application/json" -d '{
  "messages": [{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]
}'
```

### Offline mode (no API key)

If `GROQ_API_KEY` is unset, `app/llm.py` transparently falls back to a small
deterministic rule-based stand-in (`app/offline_llm.py`) so the app and the full test
suite run end-to-end without network access. Set a real key to use Groq
(`llama-3.3-70b-versatile`) for production-quality analysis, reranking and comparison.

---

## Data

- `data/catalog.json` — `list[CatalogEntry]` (SHL Individual Test Solutions).
  **The committed file is a representative ~30-entry sample** spanning every test-type
  code so the system runs end-to-end out of the box.
- To build the **full** catalog (≥300 entries, Phase 1 DoD), run the live scraper:

```bash
playwright install chromium
python scripts/scrape_catalog.py     # writes data/catalog.json
python scripts/build_index.py        # rebuild indexes
```

`data/index/` holds the built `faiss.index`, `bm25.pkl`, and `ids.json`.

---

## Testing & evaluation

```bash
pytest -q                    # Phase 3/4/5 DoD unit tests (offline)
python eval/probes.py        # behavior probes (vague, refuse, refine, hallucination)
python eval/replay.py        # simulated-user replay + mean Recall@10 over eval/traces/
```

Drop the 10 public traces into `eval/traces/` (JSON, see `eval/replay.py` docstring for
the shape) to reproduce the graded metric.

---

## Deployment

Docker image builds the index at build time so `data/` ships in the image:

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=... shl-recommender
```

`/health` is instant (the retriever warms in a startup background task). Suitable for
Render or HF Spaces (Docker); cold start allowed up to 2 minutes.

---

## Repository layout

See `ARCHITECTURE.md` §4. Core modules live in `app/`; the LangGraph agent in
`app/agent/`; data scripts in `scripts/`; evaluation in `eval/`.
