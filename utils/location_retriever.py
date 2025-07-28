import re
from typing import Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore

class LocationRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def query_location(self, location_name: str) -> Optional[Document]:
        """Specialized location query with filters"""
        results = self.vector_store.similarity_search(
            query=location_name,
            k=3,
            filter={
                "$or": [
                    {"metadata.document_type": "location_info"},
                    {"metadata.location": {"$eq": location_name}},
                    {"metadata.title": {"$contains": location_name}}
                ]
            }
        )
        return results[0] if results else None

    @staticmethod
    def extract_location_from_query(query: str) -> Optional[str]:
        """Extract location from natural language query"""
        patterns = [
            r'in\s+([A-Za-z\s]+?)(?:\?|$|,)',
            r'at\s+([A-Za-z\s]+?)(?:\?|$|,)',
            r'station\s+in\s+([A-Za-z\s]+)',
            r'address\s+(?:in|at)\s+([A-Za-z\s]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None