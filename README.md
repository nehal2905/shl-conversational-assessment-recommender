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

The agent:

- Clarifies vague hiring requests
- Recommends 1–10 relevant SHL assessments
- Refines recommendations based on follow-up constraints
- Compares SHL assessments
- Refuses off-topic and prompt-injection requests
- Guarantees every assessment name and URL comes directly from the SHL catalog

**Live Demo**

- **Hugging Face Space:** https://akulanehal-shl-conversational-assessment-recommender.hf.space
- **API Docs:** https://akulanehal-shl-conversational-assessment-recommender.hf.space/docs
- **Health Check:** https://akulanehal-shl-conversational-assessment-recommender.hf.space/health

---

# Architecture

```
POST /chat

START
   │
   ▼
analyze
   │
   ├── clarify
   ├── recommend
   ├── compare
   └── refuse
        │
        ▼
     format
        │
        ▼
       END
```

### Components

**analyze**

- Groq structured JSON analysis
- Stateless slot extraction
- Full conversation analysis every turn

**clarify**

- Returns a single clarification question
- No unnecessary LLM call

**recommend**

- Hybrid Retrieval
  - BM25
  - FAISS Dense Search
- Reciprocal Rank Fusion (RRF)
- Groq reranking
- Grounding to catalog IDs

**compare**

- rapidfuzz assessment matching
- Grounded comparison

**refuse**

- Handles:
  - off-topic requests
  - prompt injection
  - unrelated conversations

**format**

Produces the locked API schema.

---

# Engineering Invariants

- LLM never generates assessment names or URLs.
- Every recommendation is grounded to the SHL catalog.
- Stateless conversation processing.
- Hybrid retrieval is always used.
- Valid response schema is always returned.
- Prompt injection resistant.
- Clarification limited to avoid endless loops.

---

# Quick Start

Create a virtual environment

```bash
python -m venv .venv
```

Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create environment file

```bash
cp .env.example .env
```

Add

```
GROQ_API_KEY=your_key
```

Build retrieval indexes

```bash
python scripts/build_index.py
```

Run

```bash
uvicorn app.main:app --reload
```

---

# API

Health

```
GET /health
```

returns

```json
{
  "status": "ok"
}
```

Main endpoint

```
POST /chat
```

Example

```json
{
  "messages": [
    {
      "role": "user",
      "content": "I'm hiring a mid-level Java backend developer with Spring Boot and SQL."
    }
  ]
}
```

---

# Offline Mode

If `GROQ_API_KEY` is not provided, the application automatically falls back to a deterministic offline implementation.

This allows:

- local development
- automated testing
- CI
- evaluation

without requiring API access.

---

# Data

The recommender operates over the SHL Individual Test Solutions catalog.

Build the latest catalog:

```bash
playwright install chromium

python scripts/scrape_catalog.py

python scripts/build_index.py
```

Generated indexes are created automatically during the Docker build and are not committed to Git.

---

# Testing

Run all tests

```bash
python -m pytest -q
```

Behavior probes

```bash
python eval/probes.py
```

Replay evaluation

```bash
python eval/replay.py
```

---

# Final Evaluation

Current verification results

| Metric | Result |
|---------|--------|
| Unit Tests | ✅ 20 Passed |
| Behavior Probes | ✅ 3/3 Passed |
| Hallucination Rate | ✅ 0.000 |
| Replay Evaluation | ✅ Mean Recall@10 = 0.393 |
| Public Deployment | ✅ Hugging Face Spaces |

---

# Deployment

Docker

```bash
docker build -t shl-recommender .

docker run -p 8000:8000 \
-e GROQ_API_KEY=YOUR_KEY \
shl-recommender
```

Public deployment is hosted on **Hugging Face Spaces** using Docker.

---

# Repository Structure

```
app/
    agent/
    retrieval/
    grounding/
    llm/

scripts/

tests/

eval/

data/

Dockerfile

README.md

ARCHITECTURE.md

APPROACH.md
```

---

# Documentation

- **ARCHITECTURE.md** — system architecture and design decisions
- **APPROACH.md** — implementation methodology
- **README.md** — setup, deployment and evaluation

---

# Technologies

- Python
- FastAPI
- LangGraph
- Groq
- FAISS
- BM25
- fastembed
- Docker
- Hugging Face Spaces
- Playwright

---

# License

This project was developed as part of the **SHL Conversational Assessment Recommender** engineering assessment.
