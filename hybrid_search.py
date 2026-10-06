"""
Hybrid Search RAG - Base Example

Pipeline:
    Documents
        -> Vector Search (semantic)
        -> BM25 Search (keyword)
        -> Ensemble/Weighted Fusion
        -> Top-K context
        -> LLM
        -> Final answer

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
"""

import os

from dotenv import load_dotenv

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


# ---------------------------------------------------------
# 2. Sample documents
# ---------------------------------------------------------

chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    (
        "Tesla reported strong quarterly results. Tesla continues to lead "
        "in electric vehicles. Tesla announced new manufacturing facilities."
    ),
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.",
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.",
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments.",
]


# ---------------------------------------------------------
# 3. Convert text into LangChain Documents
# ---------------------------------------------------------

documents = [
    Document(
        page_content=chunk,
        metadata={"source": f"chunk_{i}"}
    )
    for i, chunk in enumerate(chunks)
]


# ---------------------------------------------------------
# 4. Vector Retriever
#    Semantic / Dense Search
# ---------------------------------------------------------

def create_vector_retriever(documents):
    print("Setting up Vector Retriever...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    return vector_retriever


# ---------------------------------------------------------
# 5. BM25 Retriever
#    Keyword / Sparse Search
# ---------------------------------------------------------

def create_bm25_retriever(documents):
    print("Setting up BM25 Retriever...")

    bm25_retriever = BM25Retriever.from_documents(documents)

    bm25_retriever.k = 5

    return bm25_retriever


# ---------------------------------------------------------
# 6. Hybrid Retriever
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
# 8. Hybrid Search RAG
# ---------------------------------------------------------

def hybrid_rag(query, hybrid_retriever, llm, top_k=5):
    """
    Retrieve documents using hybrid search and generate
    an answer using only the retrieved context.
    """

    # Retrieve using:
    #   70% Vector Search
    #   30% BM25
    retrieved_docs = hybrid_retriever.invoke(query)

    # Limit the final context
    retrieved_docs = retrieved_docs[:top_k]

    # Build context for the LLM
    context = "\n\n".join(
        [
            f"[Document {i}]\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs, 1)
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

    return response.content, retrieved_docs


# ---------------------------------------------------------
# 9. Main
# ---------------------------------------------------------

def main():

    print("=" * 80)
    print("HYBRID SEARCH RAG")
    print("=" * 80)

    # Create retrievers
    vector_retriever = create_vector_retriever(documents)
    bm25_retriever = create_bm25_retriever(documents)

    hybrid_retriever = create_hybrid_retriever(
        vector_retriever,
        bm25_retriever,
    )

    # Create LLM
    llm = create_llm()

    print("\nSetup complete!\n")

    # -----------------------------------------------------
    # Example 1
    # -----------------------------------------------------

    query = "How much did Microsoft pay for GitHub?"

    print("-" * 80)
    print(f"QUERY: {query}")
    print("-" * 80)

    answer, sources = hybrid_rag(
        query=query,
        hybrid_retriever=hybrid_retriever,
        llm=llm,
        top_k=5,
    )

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED DOCUMENTS:")
    for i, doc in enumerate(sources, 1):
        print(f"\n[{i}] {doc.page_content}")
        print(f"Source: {doc.metadata.get('source')}")

    # -----------------------------------------------------
    # Example 2
    # -----------------------------------------------------

    query = "What happened with Tesla's financial performance?"

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    answer, sources = hybrid_rag(
        query=query,
        hybrid_retriever=hybrid_retriever,
        llm=llm,
        top_k=5,
    )

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED DOCUMENTS:")
    for i, doc in enumerate(sources, 1):
        print(f"\n[{i}] {doc.page_content}")
        print(f"Source: {doc.metadata.get('source')}")


if __name__ == "__main__":
    main()
