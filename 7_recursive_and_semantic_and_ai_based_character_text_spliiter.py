from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
import json
import numpy as np
import re


# ============================================================
# 1. SOURCE TEXT
# ============================================================

mgi_text = """
MGI represents an enterprise and brand with 51 years of national and global experience. It currently operates more than 57 industrial units, houses over 65,000 employees, 6,650 distributors, and 20,000 suppliers under its umbrella.

The history of one of Bangladesh’s largest leading conglomerates, Meghna Group of Industries (MGI) can be traced all the way back to 1976 when its predecessor operated under the name of Kamal Trading Company. The conglomerate itself has humble origins and began its life as Meghna Vegetable Oil Industries Ltd. in 1989 on a small patch of land in Meghnaghat, Narayanganj.

The secret to the success and vast expansion of MGI has been diversification. The group has entered a broad array of different markets and industries including Fast Moving Consumer Goods (FMCG), building materials, pulp and paper, LPG, feeds, fiber, power plants, shipping, seeds crushing, chemicals, ship building, dockyard, securities, insurance, media and aviation.

The product range of MGI today is truly impressive and the conglomerate markets most of its products under the recognisable brand names of "Fresh", "No.1", "Actifit", "Pure" and "Meghnacem Deluxe".

The result of this level of reach and diversification has been that one in every two households in Bangladesh uses MGI products.

Internationally MGI has a substantial presence in the Middle East, Southeast Asia, Europe, South Africa, and North and South America.

As a result of this relentless process of expansion MGI has become a powerful player within Bangladesh and has become the largest investor in relation to industrial development in Bangladesh over the last few years.

MGI became the first company in Bangladesh to establish a private economic zone known as the "Meghna Economic Zone", which has since been followed by the creation of three further economic zones, which are named “Meghna Industrial Economic Zone”, “Cumilla Economic Zone” and "Titas Economic Zone" respectively.

The conglomerate has expanded even further since this point, with an unprecedented investment of $451 million in 2020 that has erected nine new industrial units within its multiple economic zones.

Throughout this process the unwavering commitment of its visionary leader, Mostafa Kamal, has been pivotal for both the conglomerate and the Bangladeshi Economy.

Renowned for his entrepreneurial expertise and patriotism, Mostafa Kamal has played a key role in the development of industry, healthcare, education, sports and social welfare in Bangladesh.

The integrity and dedication towards the group that he has played a vital part in the overall success of MGI.
"""


# ============================================================
# 2. RECURSIVE SPLITTING
# ============================================================

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=600,
    chunk_overlap=0,
)

recursive_chunks = recursive_splitter.split_text(mgi_text)

print("=" * 70)
print("RECURSIVE CHUNKS")
print("=" * 70)

for i, chunk in enumerate(recursive_chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars)")
    print(chunk)


# ============================================================
# 3. OLLAMA EMBEDDINGS
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)


# ============================================================
# 4. EMBED ALL RECURSIVE CHUNKS
# ============================================================

chunk_embeddings = embeddings.embed_documents(recursive_chunks)


# ============================================================
# 5. COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ============================================================
# 6. SEMANTIC MERGING
# ============================================================

MAX_SEMANTIC_CHUNK_SIZE = 1000
SIMILARITY_THRESHOLD = 0.70

semantic_chunks = []

current_chunk = recursive_chunks[0]
current_embedding = chunk_embeddings[0]

for i in range(1, len(recursive_chunks)):

    next_chunk = recursive_chunks[i]
    next_embedding = chunk_embeddings[i]

    similarity = cosine_similarity(
        current_embedding,
        next_embedding
    )

    merged_size = len(current_chunk) + len(next_chunk) + 2

    print(
        f"\nSimilarity: Chunk {i} → Chunk {i + 1}"
        f" = {similarity:.4f}"
    )

    print(f"Merged size would be: {merged_size}")

    # Merge ONLY when:
    # 1. Semantically similar
    # 2. Result is not too large

    if (
        similarity >= SIMILARITY_THRESHOLD
        and merged_size <= MAX_SEMANTIC_CHUNK_SIZE
    ):

        current_chunk = (
            current_chunk
            + "\n\n"
            + next_chunk
        )

        # Re-embed merged chunk
        current_embedding = embeddings.embed_query(
            current_chunk
        )

    else:

        semantic_chunks.append(current_chunk)

        current_chunk = next_chunk
        current_embedding = next_embedding


# Save final chunk
semantic_chunks.append(current_chunk)

# ============================================================
# 7. SHOW SEMANTIC CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL SEMANTIC CHUNKS")
print("=" * 70)

for i, chunk in enumerate(semantic_chunks, 1):

    print(f"\nSemantic Chunk {i}")
    print(f"Characters: {len(chunk)}")
    print("-" * 70)
    print(chunk)
    

# implement LLM for title section and summery
llm = OllamaLLM(
    model="llama3",
    base_url="http://localhost:11434"
)


# ============================================================
# 8. LLM PROMPT FOR METADATA
# ============================================================

prompt = ChatPromptTemplate.from_template("""
You are a document metadata generator for a RAG system.

Analyze the following text and generate:

1. title
2. section
3. summary

Rules:

- title must be short and specific
- section must describe the broader topic
- summary must be one or two sentences
- Use ONLY information present in the text
- Do not invent facts
- Do not add opinions
- Return ONLY valid JSON
- Do not use Markdown
- Do not wrap the JSON in ```json

Required format:

{{
    "title": "Short title",
    "section": "Broader section",
    "summary": "Short summary"
}}

TEXT:

{text}
""")

chain = prompt | llm



metadata_list = []

for i, chunk in enumerate(semantic_chunks, 1):

    print("\n" + "=" * 70)
    print(f"GENERATING METADATA FOR CHUNK {i}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Generate metadata using LLM
    # --------------------------------------------------------

    response = chain.invoke({
        "text": chunk
    })

    response = response.strip()

    print("\nLLM Response:")
    print(response)

    metadata = None

    # --------------------------------------------------------
    # 2. Try parsing response directly
    # --------------------------------------------------------

    try:
        metadata = json.loads(response)

    except json.JSONDecodeError:

        # ----------------------------------------------------
        # 3. Remove Markdown code fences
        # ----------------------------------------------------

        cleaned = re.sub(
            r"```json\s*",
            "",
            response,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(
            r"```\s*",
            "",
            cleaned
        )

        cleaned = cleaned.strip()

        # ----------------------------------------------------
        # 4. Try parsing cleaned response
        # ----------------------------------------------------

        try:
            metadata = json.loads(cleaned)

        except json.JSONDecodeError:

            # ------------------------------------------------
            # 5. Extract JSON object from additional text
            # ------------------------------------------------

            match = re.search(
                r"\{.*\}",
                response,
                re.DOTALL
            )

            if match:
                json_text = match.group(0)

                try:
                    metadata = json.loads(json_text)

                except json.JSONDecodeError:
                    metadata = None

    # --------------------------------------------------------
    # 6. Handle invalid JSON
    # --------------------------------------------------------

    if metadata is None:

        print("WARNING: Invalid JSON returned by LLM")

        metadata = {
            "title": "",
            "section": "",
            "summary": ""
        }

    # --------------------------------------------------------
    # 7. Store metadata
    # --------------------------------------------------------

    metadata_list.append(metadata)
    
    
# ============================================================
# 10. CREATE FINAL DOCUMENTS
# ============================================================

final_documents = []

for chunk, metadata in zip(
    semantic_chunks,
    metadata_list
):

    document = Document(
        page_content=chunk,
        metadata={
            "title": metadata.get("title", ""),
            "section": metadata.get("section", ""),
            "summary": metadata.get("summary", ""),
            "source": "mgi_company_profile"
        }
    )

    final_documents.append(document)
    
    
    
# ============================================================
# 10. CREATE FINAL DOCUMENTS
# ============================================================

final_documents = []

for chunk, metadata in zip(
    semantic_chunks,
    metadata_list
):

    document = Document(
        page_content=chunk,
        metadata={
            "title": metadata.get("title", ""),
            "section": metadata.get("section", ""),
            "summary": metadata.get("summary", ""),
            "source": "mgi_company_profile"
        }
    )

    final_documents.append(document)


# ============================================================
# 11. PREPARE DOCUMENTS FOR VECTOR DB
# ============================================================

db_documents = []

for i, doc in enumerate(final_documents, 1):

    # Stable chunk ID
    chunk_id = f"mgi_profile_chunk_{i:03d}"

    metadata = {
        "document_id": "mgi_profile_001",
        "chunk_id": i,
        "chunk_key": chunk_id,

        "title": doc.metadata.get("title", ""),
        "section": doc.metadata.get("section", ""),
        "summary": doc.metadata.get("summary", ""),

        "source": "mgi_company_profile",
        "source_type": "text",

        "version": 1,
    }

    db_doc = Document(
        page_content=doc.page_content,
        metadata=metadata
    )

    db_documents.append(db_doc)


# ============================================================
# 12. VERIFY BEFORE INSERT
# ============================================================

print("\n" + "=" * 70)
print("DOCUMENTS READY FOR VECTOR DB")
print("=" * 70)

for doc in db_documents:

    print("\nID:", doc.metadata["chunk_key"])
    print("Document ID:", doc.metadata["document_id"])
    print("Chunk ID:", doc.metadata["chunk_id"])
    print("Title:", doc.metadata["title"])
    print("Section:", doc.metadata["section"])
    print("Source:", doc.metadata["source"])
    print("Content length:", len(doc.page_content))


# ============================================================
# 13. CREATE CHROMA VECTOR STORE
# ============================================================

vectorstore = Chroma(
    collection_name="mgi",
    persist_directory="db/chroma_nomic_db",
    embedding_function=embeddings
)


# ============================================================
# 14. INSERT DOCUMENTS
# ============================================================

ids = [
    doc.metadata["chunk_key"]
    for doc in db_documents
]

vectorstore.add_documents(
    documents=db_documents,
    ids=ids
)


# ============================================================
# 15. INSERTION RESULT
# ============================================================

print("\n" + "=" * 70)
print("INSERTION COMPLETE")
print("=" * 70)

print(f"Documents inserted: {len(db_documents)}")
print("Collection: mgi")
print("Embedding model: nomic-embed-text")
print("Database: db/chroma_nomic_db")