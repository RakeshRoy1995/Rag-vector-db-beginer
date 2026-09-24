import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    # print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks,
    persist_directory="db/chroma_db"):

    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=os.environ["MODEL_NAME"]
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vector store ---")
    return vectorstore

def main():
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    # Step 1: Load documents
    documents = load_documents(docs_path)
    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # # Step 3: Create vector store
    create_vector_store(chunks, persistent_directory)
    
    print("main")

if __name__ == "__main__":
   main()