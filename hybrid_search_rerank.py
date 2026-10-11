"""
Hybrid Search + Re-Ranking RAG

Pipeline:
    Documents
        -> Vector Search (semantic)       -> top 20
        -> BM25 Search (keyword)          -> top 20
        -> Ensemble/Weighted Fusion       -> top 20 candidates
        -> Cross-Encoder Re-Ranking       -> best 3-5
        -> LLM
        -> Final answer

Why re-rank?
    Vector search and BM25 are fast but approximate: they score the query
    and each document independently. A cross-encoder reads the query and
    the document TOGETHER, so it is much more accurate - but also slower.
    So we use hybrid search to cheaply narrow down to ~20 candidates, then
    let the cross-encoder pick the best few for the LLM.

Requirements:
    pip install -U langchain langchain-community langchain-huggingface \
        langchain-ollama langchain-chroma rank_bm25 python-dotenv \
        sentence-transformers

Ollama:
    Install Ollama and make sure it is running.

    ollama pull llama3

Optional .env:
    OLLAMA_MODEL=llama3

Embedding model:
    sentence-transformers/all-MiniLM-L6-v2

Re-ranking model (cross-encoder):
    cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import json
import os

from dotenv import load_dotenv

from sentence_transformers import CrossEncoder

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates hybrid search retrieves
RETRIEVAL_K = 20

# How many re-ranked documents are sent to the LLM (3-5)
RERANK_TOP_N = 4


# ---------------------------------------------------------
# 2. Load documents from JSON
#    Format: [{"text": "...", "topic": "..."}, ...]
# ---------------------------------------------------------

CHUNKS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rerank_chunks.json",
)


def load_documents(path=CHUNKS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Convert into LangChain Documents
    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": f"chunk_{i}",
                "topic": chunk.get("topic", "unknown"),
            },
        )
        for i, chunk in enumerate(chunks)
    ]


documents = load_documents()


# ---------------------------------------------------------
# 3. Vector Retriever
#    Semantic / Dense Search
# ---------------------------------------------------------

def create_vector_retriever(documents, k=RETRIEVAL_K):
    print("Setting up Vector Retriever...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return vector_retriever


# ---------------------------------------------------------
# 4. BM25 Retriever
#    Keyword / Sparse Search
# ---------------------------------------------------------

def create_bm25_retriever(documents, k=RETRIEVAL_K):
    print("Setting up BM25 Retriever...")

    bm25_retriever = BM25Retriever.from_documents(documents)

    bm25_retriever.k = k

    return bm25_retriever


# ---------------------------------------------------------
# 5. Hybrid Retriever
# ---------------------------------------------------------

def create_hybrid_retriever(vector_retriever, bm25_retriever):
    print("Setting up Hybrid Retriever...")

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever,
        ],
        weights=[
            0.7,  # Vector / semantic search
            0.3,  # BM25 / keyword search
        ],
    )

    return hybrid_retriever


# ---------------------------------------------------------
# 6. Re-Ranker
#    Cross-Encoder scores (query, document) pairs together
# ---------------------------------------------------------

def create_reranker():
    print("Setting up Cross-Encoder Re-Ranker...")

    return CrossEncoder(RERANKER_MODEL)


def rerank(query, docs, reranker, top_n=RERANK_TOP_N):
    """
    Score every (query, document) pair with the cross-encoder
    and return the top_n documents with their scores.
    """

    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked[:top_n]


# ---------------------------------------------------------
# 7. LLM
# ---------------------------------------------------------

def create_llm():
    # Ollama model can be configured through .env:
    # OLLAMA_MODEL=llama3
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")

    return ChatOllama(
        model=ollama_model,
        temperature=0,
    )


# ---------------------------------------------------------
# 8. Hybrid Search + Re-Ranking RAG
# ---------------------------------------------------------

def hybrid_rerank_rag(
    query,
    hybrid_retriever,
    reranker,
    llm,
    retrieval_k=RETRIEVAL_K,
    top_n=RERANK_TOP_N,
):
    """
    1. Retrieve up to `retrieval_k` candidates with hybrid search.
    2. Re-rank them with a cross-encoder.
    3. Send the best `top_n` to the LLM.
    """

    # Step 1: Hybrid retrieval (70% Vector + 30% BM25)
    candidates = hybrid_retriever.invoke(query)[:retrieval_k]

    # Step 2: Re-rank candidates
    reranked = rerank(query, candidates, reranker, top_n=top_n)

    top_docs = [doc for doc, _ in reranked]

    # Step 3: Build context for the LLM
    context = "\n\n".join(
        [
            f"[Document {i}]\n{doc.page_content}"
            for i, doc in enumerate(top_docs, 1)
        ]
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.

If the answer cannot be found in the context, say:

"I don't have enough information to answer that based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    messages = [
        SystemMessage(
            content="You answer questions using retrieved documents."
        ),
        HumanMessage(
            content=prompt
        ),
    ]

    response = llm.invoke(messages)

    return response.content, candidates, reranked


# ---------------------------------------------------------
# 9. Pretty printing
# ---------------------------------------------------------

def print_results(query, answer, candidates, reranked):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    print(f"\nHYBRID SEARCH CANDIDATES ({len(candidates)}):")
    for i, doc in enumerate(candidates, 1):
        print(f"  {i:>2}. [{doc.metadata.get('source')}] {doc.page_content}")

    print(f"\nRE-RANKED TOP {len(reranked)} (sent to LLM):")
    for i, (doc, score) in enumerate(reranked, 1):
        print(
            f"  {i}. score={score:.4f} "
            f"[{doc.metadata.get('source')}] {doc.page_content}"
        )

    print("\nANSWER:")
    print(answer)


# ---------------------------------------------------------
# 10. Main
# ---------------------------------------------------------

def main():

    print("=" * 80)
    print("HYBRID SEARCH + RE-RANKING RAG")
    print("=" * 80)

    # Create retrievers
    vector_retriever = create_vector_retriever(documents)
    bm25_retriever = create_bm25_retriever(documents)

    hybrid_retriever = create_hybrid_retriever(
        vector_retriever,
        bm25_retriever,
    )

    # Create re-ranker and LLM
    reranker = create_reranker()
    llm = create_llm()

    print("\nSetup complete!\n")

    queries = [
        "How much did Microsoft pay for GitHub?",
        "What happened with Tesla's financial performance?",
        "Which programming language is popular in AI?",
    ]

    for query in queries:
        answer, candidates, reranked = hybrid_rerank_rag(
            query=query,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            llm=llm,
        )

        print_results(query, answer, candidates, reranked)


if __name__ == "__main__":
    main()
