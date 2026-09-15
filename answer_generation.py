import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# --------------------------------------------------
# 1. Load your existing embedding model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name=os.environ["MODEL_NAME"]
)

# --------------------------------------------------
# 2. Load ChromaDB
# --------------------------------------------------

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# --------------------------------------------------
# 3. Local Ollama LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --------------------------------------------------
# 4. User query
# --------------------------------------------------

query = "Can you please provide google, microsoft, tesla  the company's headquarters ?"

print(f"User Query: {query}")
print("--- Context ---")

# --------------------------------------------------
# 5. Retrieve relevant documents
# --------------------------------------------------

relevant_docs = db.similarity_search_with_score(
    query,
    k=10
)

filtered_docs = []
seen = set()

# Only keep highly relevant documents
for doc, score in relevant_docs:
    content = doc.page_content.strip()

    # Skip duplicate documents
    if content in seen:
        continue

    seen.add(content)

    filtered_docs.append((doc, score))

for i, (doc, score) in enumerate(filtered_docs, 1):
    print(f"\nDocument {i}")
    print(f"Distance: {score:.4f}")
    print(doc.page_content[:100])

# --------------------------------------------------
# 6. Build context for LLM
# --------------------------------------------------

context = "\n\n".join(
    doc.page_content
    for doc, score in filtered_docs
)

# --------------------------------------------------
# 7. Prompt Llama 3
# --------------------------------------------------

prompt = f"""
You are a RAG question-answering assistant.

Your job is to answer the user's question using ONLY the information
contained in the CONTEXT.

Rules:
1. Carefully identify which document/entity is relevant to the question.
2. Ignore documents that are unrelated to the question.
3. Do not use your own knowledge.
4. Do not assume the answer.
5. If the answer is explicitly present in the context, answer it directly.
6. If the answer is not present in the context, respond:
   "I don't know based on the provided context."


Context:
{context}

Question:
{query}

Answer:
"""

# --------------------------------------------------
# 8. Generate answer using Ollama
# --------------------------------------------------

response = llm.invoke(prompt)

print("\n--- Answer ---")
print(response)