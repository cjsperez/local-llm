from vector_store import VectorStoreManager
from data_loader import DocumentLoader
from config import Config
import os

def main():
    # Step 1: Load and prepare documents
    documents = DocumentLoader.load_and_prepare_documents()
    print(f"Looking for JSON at: {Config.MAIN_DOCUMENTS_FILE}")
    print(f"File exists: {os.path.exists(Config.MAIN_DOCUMENTS_FILE)}")
    if not documents:
        print("No documents found - exiting.")
        return
    
    # Step 2: Initialize vector store
    if Config.RECREATE_STORE:
        print("Creating new vector store...")
        db = VectorStoreManager.create_vector_store(documents)
    else:
        print("Loading existing vector store...")
        db = VectorStoreManager.get_vector_store()
    
    # Step 3: Verify
    if db and db._collection.count() > 0:
        print(f"Vector store ready with {db._collection.count()} chunks")
        # Example query
        results = db.similarity_search("What is PNC-AI?", k=2)

        print(f"Retrieved {len(results)} relevant documents")
        print(f"Retrieved {results} relevant documents")
    else:
        print("Failed to initialize vector store")

if __name__ == "__main__":
    main()