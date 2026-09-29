"""
MMR (Maximal Marginal Relevance) retrieval over the PDF collection.

Plain similarity search often returns near-duplicates (e.g. the overlapping
pieces of one paragraph). MMR picks results one at a time, balancing:

    relevance  - similarity to the query
    diversity  - dissimilarity to results already picked

    score(d) = λ · sim(query, d)  -  (1 - λ) · max sim(d, picked)

    λ = 1.0  -> pure relevance (same as similarity search)
    λ = 0.0  -> pure diversity
    λ = 0.5  -> common default

Usage:
    python retrieval_mmr.py "What does the document say about data privacy?"
    python retrieval_mmr.py "BLEU score" -k 5 --fetch-k 30 --lambda 0.7
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

PERSIST_DIR = "db/chroma_pdf_db"
COLLECTION = "pdf_chunks"
EMBEDDING_MODEL = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = 5            # results returned
FETCH_K = 25         # candidates MMR chooses from (must be > TOP_K)
LAMBDA = 0.5         # 1.0 = relevance only, 0.0 = diversity only


# ============================================================
# VECTOR STORE
# ============================================================

def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        collection_metadata={"hnsw:space": "cosine"},
    )


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


# ============================================================
# MMR
# ============================================================

def mmr_select(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int = TOP_K,
    lambda_mult: float = LAMBDA,
) -> List[int]:
    """Return indices of doc_vecs chosen by MMR, in pick order."""

    query_vec = normalize(query_vec)
    doc_vecs = normalize(doc_vecs)

    relevance = doc_vecs @ query_vec            # cosine sim to query, shape (n,)
    doc_sim = doc_vecs @ doc_vecs.T             # cosine sim between docs, (n, n)

    selected: List[int] = []
    remaining = list(range(len(doc_vecs)))

    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])   # first pick = most relevant
        else:
            def score(i: int) -> float:
                redundancy = max(doc_sim[i, j] for j in selected)
                return lambda_mult * relevance[i] - (1 - lambda_mult) * redundancy
            best = max(remaining, key=score)

        selected.append(best)
        remaining.remove(best)

    return selected


def fetch_candidates(
    store: Chroma,
    query_vec: List[float],
    fetch_k: int,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Top fetch_k by similarity, with their stored embeddings."""
    return store._collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "embeddings", "distances"],
    )


def mmr_search(
    store: Chroma,
    query: str,
    k: int = TOP_K,
    fetch_k: int = FETCH_K,
    lambda_mult: float = LAMBDA,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:

    query_vec = store.embeddings.embed_query(query)
    res = fetch_candidates(store, query_vec, fetch_k, where)

    if not res["ids"][0]:
        return []

    doc_vecs = np.array(res["embeddings"][0], dtype=float)
    picks = mmr_select(np.array(query_vec, dtype=float), doc_vecs, k, lambda_mult)

    return [
        {
            "id": res["ids"][0][i],
            "similarity": 1 - res["distances"][0][i],     # cosine distance -> similarity
            "candidate_rank": i + 1,                       # rank in plain similarity order
            "metadata": res["metadatas"][0][i],
            "embedding": doc_vecs[i],
        }
        for i in picks
    ]


def similarity_search(
    store: Chroma,
    query: str,
    k: int = TOP_K,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Plain top-k, for comparison."""
    query_vec = store.embeddings.embed_query(query)
    res = fetch_candidates(store, query_vec, k, where)
    return [
        {
            "id": res["ids"][0][i],
            "similarity": 1 - res["distances"][0][i],
            "candidate_rank": i + 1,
            "metadata": res["metadatas"][0][i],
            "embedding": np.array(res["embeddings"][0][i], dtype=float),
        }
        for i in range(len(res["ids"][0]))
    ]


# ============================================================
# REPORTING
# ============================================================

def redundancy(results: List[Dict[str, Any]]) -> float:
    """Average pairwise cosine similarity between results (lower = more diverse)."""
    if len(results) < 2:
        return 0.0
    vecs = normalize(np.array([r["embedding"] for r in results]))
    sim = vecs @ vecs.T
    n = len(results)
    return float((sim.sum() - n) / (n * (n - 1)))


def print_results(title: str, results: List[Dict[str, Any]]):
    sections = {r["metadata"].get("section_path") for r in results}
    print(f"\n{title}")
    print(f"  relevance avg {np.mean([r['similarity'] for r in results]):.3f} | "
          f"redundancy {redundancy(results):.3f} | {len(sections)} distinct sections")
    print("-" * 80)
    for n, r in enumerate(results, start=1):
        m = r["metadata"]
        text = (m.get("content") or "").replace("\n", " ")
        print(f"[{n}] sim={r['similarity']:.3f}  (similarity rank #{r['candidate_rank']})  "
              f"{m.get('type')}  p.{m.get('page_start')}  {m.get('section_path') or 'front matter'}")
        print(f"    {text[:160]}")


# ============================================================
# MAIN
# ============================================================

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Compare similarity search and MMR")
    parser.add_argument("query", nargs="?", default="What does the document say about data privacy?")
    parser.add_argument("-k", type=int, default=TOP_K)
    parser.add_argument("--fetch-k", type=int, default=FETCH_K)
    parser.add_argument("--lambda", dest="lambda_mult", type=float, default=LAMBDA)
    parser.add_argument("--type", choices=["text", "table", "figure"], help="filter by chunk type")
    parser.add_argument("--source", help="filter by PDF file name, e.g. Google.pdf")
    args = parser.parse_args()

    filters = [{"type": args.type}] if args.type else []
    if args.source:
        filters.append({"source": args.source})
    where = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)

    store = get_vector_store()
    print(f"Collection '{COLLECTION}': {store._collection.count()} chunks")
    print(f"Query: {args.query}")

    plain = similarity_search(store, args.query, args.k, where)
    mmr = mmr_search(store, args.query, args.k, args.fetch_k, args.lambda_mult, where)

    print_results(f"SIMILARITY top-{args.k}", plain)
    print_results(f"MMR top-{args.k}  (fetch_k={args.fetch_k}, λ={args.lambda_mult})", mmr)

    # Cross-check with LangChain's built-in MMR (same algorithm)
    lc = store.max_marginal_relevance_search(
        args.query, k=args.k, fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult, filter=where,
    )
    # LangChain returns the picks in similarity order, not pick order -> compare sets
    same = {d.id for d in lc} == {r["id"] for r in mmr}
    print(f"\nLangChain max_marginal_relevance_search picks the same chunks: {same}")


if __name__ == "__main__":
    main()
