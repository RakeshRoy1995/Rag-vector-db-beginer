import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
        model_name=os.environ["MODEL_NAME"]
    )

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "What is NVIDIA?"

print(f"User Query: {query}")
print("--- Context ---")

relevant_docs = db.similarity_search_with_score(
    query,
    k=10
)

for i, (doc, score) in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}")
    print(f"Distance: {score:.4f}")
    print(doc.page_content[:300])
    
# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"