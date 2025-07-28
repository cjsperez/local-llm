from langchain_core.documents.base import Document
from typing import List

def validate_langchain_documents(docs: List[Document]) -> bool:
    """Validate documents are properly formatted for Chroma"""
    required_attrs = ['page_content', 'metadata']
    for doc in docs:
        if not all(hasattr(doc, attr) for attr in required_attrs):
            print(f"Invalid document: missing required attributes {required_attrs}")
            return False
        if not isinstance(doc.page_content, str):
            print(f"Invalid page_content type in document: {type(doc.page_content)}")
            return False
    return True

def print_sample_document(docs: List[Document]):
    """Print a sample document for verification"""
    if docs:
        sample = docs[0]
        print("\nSample Document Structure:")
        print(f"Page Content: {sample.page_content[:100]}...")
        print("Metadata:")
        for k, v in sample.metadata.items():
            print(f"  {k}: {v}")