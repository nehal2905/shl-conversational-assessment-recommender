"""Phase 2 — build the dense (FAISS) + sparse (BM25) indexes.

Run:
    python scripts/build_index.py

Reads data/catalog.json, writes:
    data/index/faiss.index   (IndexFlatIP over normalized bge-small vectors = cosine)
    data/index/bm25.pkl      (BM25Okapi over tokenized name + description)
    data/index/ids.json      (row order → catalog id mapping)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.catalog import load_catalog  # noqa: E402
from app.retrieval import doc_text, embed_text, tokenize  # noqa: E402


def main() -> None:
    import faiss
    import numpy as np
    from fastembed import TextEmbedding
    from rank_bm25 import BM25Okapi

    entries = load_catalog(str(config.CATALOG_PATH))
    if not entries:
        raise SystemExit("catalog.json is empty — run scrape_catalog.py first.")

    ids = [e.id for e in entries]
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # --- Dense (FAISS) ------------------------------------------------------
    print(f"Embedding {len(entries)} entries with {config.EMBED_MODEL} ...")
    embedder = TextEmbedding(model_name=config.EMBED_MODEL)
    vectors = list(embedder.embed([embed_text(e) for e in entries]))
    mat = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(mat)  # cosine via inner product on normalized vectors
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, str(config.FAISS_PATH))
    print(f"  wrote {config.FAISS_PATH} (dim={mat.shape[1]})")

    # --- Sparse (BM25) ------------------------------------------------------
    corpus = [tokenize(doc_text(e)) for e in entries]
    bm25 = BM25Okapi(corpus)
    with open(config.BM25_PATH, "wb") as fh:
        pickle.dump(bm25, fh)
    print(f"  wrote {config.BM25_PATH}")

    # --- id mapping ---------------------------------------------------------
    with open(config.IDS_PATH, "w", encoding="utf-8") as fh:
        json.dump(ids, fh)
    print(f"  wrote {config.IDS_PATH} ({len(ids)} ids)")

    print("Index build complete.")


if __name__ == "__main__":
    main()
