import asyncio
from vector_store import VectorStoreManager
from data_loader import DocumentLoader
from config import Config
import logging
import json
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initialize():
    """Initialize or reinitialize the vector store with validation"""
    try:
        # Load and validate documents
        print("Loading and validating documents...")
        documents = DocumentLoader.load_and_prepare_documents()
        
        # Verify we have documents
        if not documents:
            raise ValueError("No documents available after processing")
        
        # Additional validation
        from document_validator import validate_langchain_documents, print_sample_document
        if not validate_langchain_documents(documents):
            print_sample_document(documents)
            raise ValueError("Documents failed validation")
        
        print_sample_document(documents)
        
        print(f"Creating vector store with {len(documents)} documents...")
        store = VectorStoreManager.create_vector_store(documents, recreate=True)
        
        # Verify creation
        if not hasattr(store, '_collection'):
            raise RuntimeError("Vector store creation failed - no collection attribute")
            
        count = store._collection.count()
        print(f"Successfully created vector store with {count} documents")
        return {
            "status": "success",
            "document_count": count,
        }
    except Exception as e:
        print(f"Initialization failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "solution": "Ensure all documents have 'page_content' string and 'metadata' dict"
        }

if __name__ == "__main__":
    result = asyncio.run(initialize())
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "success" else 1)