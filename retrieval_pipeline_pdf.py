"""
Retrieval over chunks.json (from chunking_pipeline_pdf.py).

1. Sync   chunks.json -> Chroma (adds new chunk_ids, deletes stale ones)
2. Search hybrid: vector (embedding_text) + BM25 keywords, fused with RRF
3. Expand optionally add neighbouring chunks from the same section
4. Format numbered context with citations, ready for an LLM prompt

Usage:
    python retrieval_pipeline_pdf.py "What is multi-head attention?"
    python retrieval_pipeline_pdf.py "BLEU score on WMT 2014" -k 3 --type table
    python retrieval_pipeline_pdf.py --reindex
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

CHUNKS_JSON = "chunks.json"
PERSIST_DIR = "db/chroma_pdf_db"
COLLECTION = "pdf_chunks"
EMBEDDING_MODEL = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = 5
CANDIDATES = 20          # results fetched from each retriever before fusion
RRF_K = 60               # reciprocal-rank-fusion constant (standard value)
VECTOR_WEIGHT = 1.0
KEYWORD_WEIGHT = 1.0
NEIGHBOURS = 0           # chunks to add before/after each hit, same section

# Chroma metadata must be str / int / float / bool
METADATA_FIELDS = (
    "chunk_index", "parent_id", "type", "source",
    "section_title", "section_path", "page_start", "page_end",
)


# ============================================================
# LOAD
# ============================================================

def load_chunks(path: str = CHUNKS_JSON) -> List[Dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"{file.resolve()} not found - run chunking_pipeline_pdf.py first"
        )
    with file.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    meta = {k: chunk[k] for k in METADATA_FIELDS if chunk.get(k) is not None}
    meta["content"] = chunk["content"]
    if chunk.get("image_paths"):
        meta["image_paths"] = json.dumps(chunk["image_paths"])
    return meta


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


def sync_index(
    store: Chroma,
    chunks: List[Dict[str, Any]],
    reindex: bool = False,
    batch_size: int = 64,
):
    """Make the collection match chunks.json. chunk_ids are content hashes,
    so only new / changed chunks get embedded."""

    existing = set(store.get(include=[])["ids"])
    wanted = {c["chunk_id"]: c for c in chunks}

    stale = list(existing - wanted.keys()) if not reindex else list(existing)
    if stale:
        store.delete(ids=stale)

    to_add = [c for cid, c in wanted.items() if reindex or cid not in existing]

    for i in range(0, len(to_add), batch_size):
        batch = to_add[i:i + batch_size]
        store.add_texts(
            texts=[c["embedding_text"] for c in batch],
            metadatas=[to_metadata(c) for c in batch],
            ids=[c["chunk_id"] for c in batch],
        )

    print(f"Index: {len(wanted)} chunks "
          f"(+{len(to_add)} added, -{len(stale)} removed) -> {PERSIST_DIR}")


# ============================================================
# KEYWORD SEARCH (BM25)
# ============================================================

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with",
}


def tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Small Okapi BM25, no extra dependency."""

    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)

        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> List[tuple]:
        terms = tokenize(query)
        scores = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    score += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ============================================================
# RETRIEVER
# ============================================================

class PdfRetriever:

    def __init__(self, chunks_path: str = CHUNKS_JSON, reindex: bool = False):
        self.chunks = load_chunks(chunks_path)
        self.by_id = {c["chunk_id"]: c for c in self.chunks}

        self.store = get_vector_store()
        sync_index(self.store, self.chunks, reindex=reindex)

        # BM25 over content + section path (headings are strong keywords)
        self.bm25 = BM25([f"{c['section_path']} {c['content']}" for c in self.chunks])

        # Chunks per section in order, for neighbour expansion
        self.sections: Dict[str, List[str]] = {}
        for c in sorted(self.chunks, key=lambda c: c["chunk_index"]):
            self.sections.setdefault(c["parent_id"], []).append(c["chunk_id"])

    # ---------------- individual retrievers ----------------

    def vector_search(self, query: str, k: int, where: Optional[Dict] = None) -> List[tuple]:
        """[(chunk_id, cosine_similarity)]"""
        hits = self.store.similarity_search_with_score(query, k=k, filter=where)
        return [
            (doc.id or doc.metadata.get("chunk_id"), 1 - distance)
            for doc, distance in hits
        ]

    def keyword_search(self, query: str, k: int, chunk_type: Optional[str] = None) -> List[tuple]:
        """[(chunk_id, bm25_score)]"""
        results = []
        for i, score in self.bm25.search(query, len(self.chunks)):
            chunk = self.chunks[i]
            if chunk_type and chunk["type"] != chunk_type:
                continue
            results.append((chunk["chunk_id"], score))
            if len(results) >= k:
                break
        return results

    # ---------------- hybrid ----------------

    def search(
        self,
        query: str,
        k: int = TOP_K,
        chunk_type: Optional[str] = None,
        neighbours: int = NEIGHBOURS,
    ) -> List[Dict[str, Any]]:

        where = {"type": chunk_type} if chunk_type else None

        vector_hits = self.vector_search(query, CANDIDATES, where)
        keyword_hits = self.keyword_search(query, CANDIDATES, chunk_type)

        # Reciprocal Rank Fusion: robust to the two scores' different scales
        fused: Dict[str, Dict[str, Any]] = {}
        for weight, hits, name in (
            (VECTOR_WEIGHT, vector_hits, "vector"),
            (KEYWORD_WEIGHT, keyword_hits, "keyword"),
        ):
            for rank, (cid, score) in enumerate(hits, start=1):
                entry = fused.setdefault(cid, {"score": 0.0})
                entry["score"] += weight / (RRF_K + rank)
                entry[f"{name}_rank"] = rank
                entry[f"{name}_score"] = round(score, 4)

        ranked = sorted(fused.items(), key=lambda x: x[1]["score"], reverse=True)[:k]

        results = []
        for cid, info in ranked:
            chunk = self.by_id.get(cid)
            if chunk is None:        # index out of sync with chunks.json
                continue
            results.append({
                **chunk,
                "score": round(info["score"], 5),
                "vector_rank": info.get("vector_rank"),
                "keyword_rank": info.get("keyword_rank"),
                "vector_score": info.get("vector_score"),
                "keyword_score": info.get("keyword_score"),
                "context": self.expand(cid, neighbours),
            })
        return results

    def expand(self, chunk_id: str, neighbours: int) -> str:
        """Hit text plus up to `neighbours` chunks either side in its section."""
        chunk = self.by_id[chunk_id]
        if neighbours <= 0:
            return chunk["content"]

        ids = self.sections[chunk["parent_id"]]
        i = ids.index(chunk_id)
        window = ids[max(0, i - neighbours): i + neighbours + 1]
        return "\n\n".join(self.by_id[c]["content"] for c in window)


# ============================================================
# FORMATTING
# ============================================================

def citation(r: Dict[str, Any]) -> str:
    pages = r["page_start"] if r["page_start"] == r["page_end"] else f"{r['page_start']}-{r['page_end']}"
    section = r["section_path"] or "front matter"
    return f"{r['source']}, p.{pages}, {section}"


def format_context(results: List[Dict[str, Any]]) -> str:
    """Numbered sources for an LLM prompt; ask the model to cite [n]."""
    blocks = []
    for n, r in enumerate(results, start=1):
        blocks.append(f"[{n}] ({citation(r)})\n{r['context']}")
    return "\n\n---\n\n".join(blocks)


def print_results(query: str, results: List[Dict[str, Any]]):
    print(f"\nQuery: {query}\n" + "=" * 80)
    for n, r in enumerate(results, start=1):
        print(f"\n[{n}] {r['type']:6} score={r['score']:.4f}  "
              f"vector#{r['vector_rank'] or '-'} ({r['vector_score'] or '-'})  "
              f"keyword#{r['keyword_rank'] or '-'}")
        print(f"    {citation(r)}")
        if r.get("image_paths"):
            print(f"    images: {', '.join(r['image_paths'])}")
        print("    " + r["context"][:400].replace("\n", "\n    "))


# ============================================================
# MAIN
# ============================================================

SAMPLE_QUERIES = [
    "What is scaled dot-product attention?",
    "How many attention heads does the Transformer use?",
    "What BLEU score does the Transformer achieve on English-to-German?",
    "Which optimizer and learning rate schedule were used?",
    "Show the Transformer model architecture figure",
]


def main():
    # Windows consoles default to cp1252 and choke on math symbols
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Search chunks.json")
    parser.add_argument("query", nargs="?", help="question (default: sample queries)")
    parser.add_argument("-k", type=int, default=TOP_K, help="results to return")
    parser.add_argument("--type", choices=["text", "table", "figure"], help="filter by chunk type")
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS, help="add N surrounding chunks")
    parser.add_argument("--reindex", action="store_true", help="re-embed everything")
    parser.add_argument("--context", action="store_true", help="print LLM-ready context")
    args = parser.parse_args()

    retriever = PdfRetriever(reindex=args.reindex)

    queries = [args.query] if args.query else SAMPLE_QUERIES
    for query in queries:
        results = retriever.search(query, k=args.k, chunk_type=args.type, neighbours=args.neighbours)
        print_results(query, results)
        if args.context:
            print("\n" + "=" * 80 + "\nLLM CONTEXT\n" + "=" * 80)
            print(format_context(results))


if __name__ == "__main__":
    main()
