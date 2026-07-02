---
title: SHL Conversational Assessment Recommender
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SHL Conversational Assessment Recommender

A stateless conversational AI that guides recruiters from vague hiring intents (for example, *"I'm hiring a Java developer"*) to grounded recommendations of SHL Individual Test Solutions.

The agent clarifies vague hiring requests, recommends 1–10 assessments, refines on follow-up constraints, compares named assessments, and refuses off-topic and prompt-injection requests. **Every returned URL comes from the scraped SHL catalog** — the LLM never emits a name or URL directly.

Built to the spec in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

**Live Demo**

- **Hugging Face Space:** https://akulanehal-shl-conversational-assessment-recommender.hf.space
- **API Docs:** https://akulanehal-shl-conversational-assessment-recommender.hf.space/docs
- **Health Check:** https://akulanehal-shl-conversational-assessment-recommender.hf.space/health

---

## Features

- Stateless FastAPI API
- LangGraph conversational workflow
- Hybrid Retrieval (BM25 + FAISS)
- Groq-powered structured reasoning
- Prompt injection resistant
- Grounded recommendations from the SHL catalog
- Docker deployment
- Hugging Face Spaces deployment
- Offline fallback mode

---

## Architecture at a glance

```
POST /chat → LangGraph:
  START → analyze → {clarify | recommend | compare | refuse} → format → END

analyze   : one Groq JSON call → Analysis (intent + slots), stateless per turn
clarify   : returns the single clarifying question (no LLM call)
recommend : HybridRetriever (BM25 + FAISS) → Groq rerank → ground(ids)
compare   : rapidfuzz resolves named assessments → grounded Groq comparison
refuse    : template redirect (no LLM call)
format    : validates/coerces to the locked ChatResponse
```

Key design invariants (full list in `ARCHITECTURE.md` §6):

1. **The LLM never emits a URL or name.** Retrieval → LLM picks `id`s → `grounding.py`
   maps `id`s to trusted catalog entries. "Every URL is from the catalog" is structural.
2. **Stateless slot re-derivation** from the full `messages` list every call.
3. **Turn-budget-aware clarification** — clarify at most twice, then commit (invariant 3).
4. **Hybrid retrieval, always** (BM25 + dense, fused and reranked).
5. **Always return valid schema, even on error.**
6. **Message content is data, not instructions** (prompt-injection resistant).

---

## Supported Behaviors

- Clarify vague hiring requests
- Recommend 1–10 assessments
- Refine recommendations when requirements change
- Compare SHL assessments
- Refuse off-topic and prompt-injection requests

---

## API

### `GET /health`

Returns `200` with `{"status": "ok"}`. Responds immediately, even during index warm-up.

### `POST /chat`

Stateless. Send the full conversation history on every call.

**Request:**

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a mid-level Java developer who works with stakeholders"}
  ]
}
```

### Response Schema

```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "string",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

- `recommendations` is empty while the agent is clarifying or refusing.
- `recommendations` contains 1–10 assessments once enough context is available.
- `end_of_conversation` indicates whether the agent considers the task complete.

---

## Quick start

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment file:

```bash
cp .env.example .env
```

Add your Groq API key (optional — see Offline mode):

```
GROQ_API_KEY=your_key
```

Build retrieval indexes:

```bash
python scripts/build_index.py
```

Run the API:

```bash
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
  **The committed file contains 100 entries** spanning multiple test-type codes so the
  system runs end-to-end out of the box.
- To build the **full** catalog (≥300 entries, Phase 1 DoD), run the live scraper:

```bash
playwright install chromium
python scripts/scrape_catalog.py     # writes data/catalog.json
python scripts/build_index.py        # rebuild indexes
```

`data/index/` holds the built `faiss.index`, `bm25.pkl`, and `ids.json`. Generated indexes
are also created automatically during the Docker build and are not committed to Git.

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

## Final Evaluation

| Metric | Result |
|---------|--------|
| Unit Tests | ✅ 20 Passed |
| Behavior Probes | ✅ 3/3 Passed |
| Hallucination Rate | ✅ 0.000 |
| Mean Recall@10 | ✅ 0.393 |
| Public Deployment | ✅ Hugging Face Spaces |
| API Schema | ✅ Matches assignment specification |

---

## Deployment

Docker image builds the index at build time so `data/` ships in the image:

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=... shl-recommender
```

`/health` is instant (the retriever warms in a startup background task). Suitable for
Render or HF Spaces (Docker); cold start allowed up to 2 minutes.

**Live demo:** https://akulanehal-shl-conversational-assessment-recommender.hf.space

```bash
curl https://akulanehal-shl-conversational-assessment-recommender.hf.space/health
```

---

## Tech Stack

| Component | Technology |
|------------|------------|
| Backend | FastAPI |
| Agent Framework | LangGraph |
| LLM | Groq (Llama 3.3 70B) |
| Dense Retrieval | FAISS |
| Sparse Retrieval | BM25 |
| Embeddings | FastEmbed |
| Deployment | Hugging Face Spaces |
| Containerization | Docker |
| Testing | Pytest |

See `ARCHITECTURE.md` §4 for the full repository layout. Core modules live in `app/`;
the LangGraph agent in `app/agent/`; data scripts in `scripts/`; evaluation in `eval/`.

---

## Documentation

- **ARCHITECTURE.md** — system architecture and design decisions
- **APPROACH.md** — implementation methodology
- **README.md** — setup, deployment and evaluation

---

## License

This project was developed as part of the **SHL Conversational Assessment Recommender** engineering assessment.
