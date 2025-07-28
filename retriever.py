from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents.base import Document
from rank_bm25 import BM25Okapi
from typing import List, Union, Optional, Dict
import pickle
import os
import logging
import time
import warnings
from pathlib import Path
from pydantic import Field
from typing import Any

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

class PersistentBM25Retriever(BM25Retriever):
    """Proper implementation with Pydantic field declaration"""
    bm25: Any = Field(default=None, exclude=True)  # Declare the field
    precompute_scores: bool = Field(default=False, exclude=True)

    """Fixed implementation with proper Pydantic validation"""
    def __init__(
        self,
        docs: List[Document],  # Required by parent class
        bm25_model: BM25Okapi = None,
        k: int = 4,
        precompute_scores: bool = False,
        **kwargs
    ):
        # Initialize parent class with required params
        super().__init__(docs=docs, k=k, **kwargs)
        
        # Custom initialization
        self.bm25 = bm25_model
        self.precompute_scores = precompute_scores
        
        if precompute_scores and bm25_model:
            self._precompute_doc_scores()

    def _precompute_doc_scores(self):
        """Cache document scores for common terms"""
        self._score_cache = {}
        all_terms = set()
        for doc in self.docs:  # Using parent class's docs field
            all_terms.update(doc.page_content.split())
        
        for term in all_terms:
            self._score_cache[term] = self.bm25.get_scores([term])

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        persist_path: Union[str, Path] = "bm25_index.pkl",
        k: int = 4,
        rebuild_threshold: float = 0.3
    ) -> "PersistentBM25Retriever":
        """Simplified creation method"""
        persist_path = Path(persist_path)
        corpus = [doc.page_content.split() for doc in documents]
        
        # Load or create BM25 model
        bm25 = cls._load_or_create_bm25(
            corpus=corpus,
            persist_path=persist_path,
            documents=documents,
            rebuild_threshold=rebuild_threshold
        )
        
        return cls(
            docs=documents,  # Using required field name
            bm25_model=bm25,
            k=k,
            precompute_scores=True
        )

    @staticmethod
    def _load_or_create_bm25(corpus, persist_path, documents, rebuild_threshold):
        """Handle BM25 model loading/creation"""
        current_hashes = [hash(doc.page_content) for doc in documents]
        
        if persist_path.exists():
            try:
                with open(persist_path, "rb") as f:
                    data = pickle.load(f)
                    if (isinstance(data, dict) and 
                        len(data.get("doc_hashes", [])) > 0 and
                        sum(1 for h in current_hashes if h in data["doc_hashes"]) / len(documents) > (1 - rebuild_threshold)):
                        logger.info("Loaded persisted BM25 index")
                        return data["bm25"]
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {str(e)}")

        logger.info("Building new BM25 index...")
        bm25 = BM25Okapi(corpus)
        with open(persist_path, "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "doc_hashes": current_hashes,
                "created_at": time.time()
            }, f)
        return bm25

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Optimized retrieval with fallback"""
        if not self.bm25:
            return super().get_relevant_documents(query)
            
        start_time = time.time()
        tokenized_query = query.split()
        
        if self.precompute_scores and hasattr(self, '_score_cache'):
            doc_scores = [0] * len(self.docs)
            for term in tokenized_query:
                if term in self._score_cache:
                    scores = self._score_cache[term]
                    for i in range(len(scores)):
                        doc_scores[i] += scores[i]
        else:
            doc_scores = self.bm25.get_scores(tokenized_query)

        return self._get_top_docs(doc_scores, start_time)

    def _get_top_docs(self, doc_scores, start_time):
        """Extracted scoring logic"""
        scored_docs = sorted(
            zip(self.docs, doc_scores),
            key=lambda x: -x[1]
        )[:self.k]
        logger.debug(f"BM25 retrieval took {time.time()-start_time:.4f}s")
        return [doc for doc, _ in scored_docs]

def create_retriever(
    vector_db,
    llm=None,
    bm25_weight: float = 0.6,
    vector_weight: float = 0.4,
    default_k: int = 3,
    bm25_persist_path: str = "bm25_index.pkl"
) -> Union[EnsembleRetriever, BM25Retriever]:
    """Create hybrid retriever with persistent BM25"""
    try:
        # Validate weights
        if abs((bm25_weight + vector_weight) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")

        # Document loading with caching
        if not hasattr(vector_db, '_cached_documents'):
            start_load = time.time()
            raw_docs = vector_db._collection.get(include=["documents", "metadatas"])
            vector_db._cached_documents = [
                Document(
                    page_content=doc[0],
                    metadata=doc[1] or {}
                )
                for doc in zip(raw_docs["documents"], raw_docs["metadatas"])
                if doc[0]  # Skip empty documents
            ]
            logger.info(f"Loaded {len(vector_db._cached_documents)} documents in {time.time()-start_load:.2f}s")

        # Create retrievers
        start_bm25 = time.time()
        bm25_retriever = PersistentBM25Retriever.from_documents(
            documents=vector_db._cached_documents,
            persist_path=bm25_persist_path,
            k=min(len(vector_db._cached_documents), default_k)
        )
        logger.info(f"BM25 ready in {time.time()-start_bm25:.2f}s")

        start_vector = time.time()
        vector_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": default_k,
                "fetch_k": min(50, len(vector_db._cached_documents) * 2),
                "lambda_mult": 0.7,
            }
        )
        logger.info(f"Vector retriever ready in {time.time()-start_vector:.2f}s")

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[bm25_weight, vector_weight]
        )

    except Exception as e:
        logger.error(f"Retriever creation failed: {str(e)}", exc_info=True)
        if hasattr(vector_db, '_cached_documents'):
            logger.warning("Falling back to BM25-only retriever")
            return PersistentBM25Retriever.from_documents(
                documents=vector_db._cached_documents,
                k=min(len(vector_db._cached_documents), default_k)
            )
        raise RuntimeError("Retriever initialization failed with no fallback")