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
import re

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

class QueryAnalyzer:
    """Enhanced query understanding for EV pricing queries"""
    
    PRICE_PATTERNS = [
        r'price of',
        r'how much (does|is)',
        r'cost of',
        r'\$?\d+\.?\d*'  # Currency amounts
    ]
    
    PRODUCT_PATTERNS = {
        'package': [r'package', r'bundle', r'plan'],
        'model': [r'model [A-Z]', r'version \d+']
    }
    
    @classmethod
    def analyze(cls, query: str) -> dict:
        """Extract pricing intent and product references"""
        analysis = {
            'is_price_query': any(
                re.search(pattern, query.lower()) 
                for pattern in cls.PRICE_PATTERNS
            ),
            'products': [],
            'modifiers': []
        }
        
        # Extract product references
        for prod_type, patterns in cls.PRODUCT_PATTERNS.items():
            for pattern in patterns:
                if matches := re.findall(pattern, query, re.IGNORECASE):
                    analysis['products'].extend(matches)
        
        return analysis

    @classmethod
    def expand_query(cls, query: str) -> str:
        """Add synonyms and related terms"""
        expansions = {
            'price': ['cost', 'pricing', 'rate', 'fee'],
            'package': ['bundle', 'plan', 'tier'],
            'model': ['version', 'type', 'edition']
        }
        
        for term, synonyms in expansions.items():
            if term in query.lower():
                query += " " + " ".join(synonyms)
                
        return query

class PersistentBM25Retriever(BM25Retriever):
    """BM25 retriever with persistent storage and enhanced initialization"""
    bm25: Any = Field(default=None, exclude=True)
    precompute_scores: bool = Field(default=False, exclude=True)

    def __init__(
        self,
        docs: List[Document],
        bm25_model: BM25Okapi = None,
        k: int = 4,
        precompute_scores: bool = False,
        **kwargs
    ):
        super().__init__(docs=docs, k=k, **kwargs)
        self.bm25 = bm25_model
        self.precompute_scores = precompute_scores
        
        if precompute_scores and bm25_model:
            self._precompute_doc_scores()

    def _precompute_doc_scores(self):
        """Cache document scores for common terms"""
        self._score_cache = {}
        all_terms = set()
        for doc in self.docs:  # Using parent class's docs field
            # Example fix for retriever.py
            if isinstance(doc.page_content, str):
                all_terms.update(doc.page_content.split())
            else:
                logger.warning(f"Skipping non-string doc: {doc.page_content}")
        
        for term in all_terms:
            self._score_cache[term] = self.bm25.get_scores([term])

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        persist_path: Union[str, Path] = "bm25_index.pkl",
        k: int = 4,
        rebuild_threshold: float = 0.3,
        llm: Any = None,
        precompute_scores: bool = False  # Added parameter here
    )  -> "PersistentBM25Retriever":
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
            docs=documents,
            bm25_model=bm25,
            k=k,
            precompute_scores=precompute_scores,  # Pass it to __init__
            llm=llm
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

    def get_relevant_documents(self, query: Union[str, dict]) -> List[Document]:
        """Optimized retrieval with fallback that handles both string and dict inputs.
        
        Args:
            query: Either a string query or a dictionary containing the query under 
                'query', 'input', or 'question' keys.
                
        Returns:
            List of relevant documents sorted by relevance score.
            
        Raises:
            ValueError: If query extraction fails or tokenization produces no terms.
        """
        if not self.bm25:
            return super().get_relevant_documents(query)
            
        start_time = time.time()
        
        # Extract query text from different input formats
        if isinstance(query, dict):
            query_text = (query.get("question") or 
                        query.get("query") or 
                        query.get("input") or 
                        str(query))
        else:
            query_text = str(query)
        
        # Validate and tokenize the query
        if not query_text.strip():
            raise ValueError("Empty query after extraction")
        
        tokenized_query = query_text.split()
        if not tokenized_query:
            logger.warning(f"Query tokenization failed for: {query_text}")
            return []
        
        # Calculate document scores
        try:
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
            
        except Exception as e:
            logger.error(f"Scoring failed for query: {query_text}. Error: {str(e)}")
            # Fallback to parent implementation if scoring fails
            return super().get_relevant_documents(query_text)

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
    brand_key: Optional[str] = None,
    bm25_weight: float = 0.6,
    vector_weight: float = 0.4,
    default_k: int = 3,
    k: int = None,
    bm25_persist_path: str = "bm25_index.pkl",
    enable_hybrid: bool = True,
    score_threshold: float = 0.25  # Add this new parameter with default value
) -> Union[EnsembleRetriever, BM25Retriever]:
    """
    Creates a hybrid retriever with comprehensive document validation and debugging.
    """
    effective_k = k if k is not None else default_k
    def _normalize_brand(brand: Optional[str]) -> Optional[str]:
        """Normalizes brand strings for consistent comparison"""
        if not brand:
            return None
        return str(brand).lower().strip()

    try:
        # Initialize brand context
        norm_brand = _normalize_brand(brand_key)
        brand_context = f"for brand '{brand_key}'" if brand_key else "for all brands"
        logger.info(f"Initializing retriever {brand_context}")

        # Input validation
        if abs((bm25_weight + vector_weight) - 1.0) > 0.01:
            raise ValueError("Weights must sum to approximately 1.0")
        if default_k < 1 or default_k > 20:
            raise ValueError("default_k must be between 1 and 20")

        # Document loading with enhanced validation
        if not hasattr(vector_db, '_cached_documents'):
            logger.info(f"Loading documents from vector store {brand_context}")
            start_load = time.time()
            
            try:
                # First try with brand filter
                raw_docs = vector_db._collection.get(
                    include=["documents", "metadatas"],
                    # where={"brand": brand_key} if brand_key else None
                )

                if raw_docs and raw_docs.get("documents"):
                    sample_size = min(3, len(raw_docs["documents"]))
                    logger.info(f"Document sample (first {sample_size} of {len(raw_docs['documents'])}):")
                    
                    for i in range(sample_size):
                        doc_content = raw_docs["documents"][i]
                        metadata = raw_docs["metadatas"][i] if raw_docs.get("metadatas") and i < len(raw_docs["metadatas"]) else {}
                        
                        logger.info(
                            f"Document {i+1}:\n"
                            f"Content: {doc_content[:200]}...\n"
                            f"Metadata: {metadata}\n"
                            f"Brand: {metadata.get('brand', 'MISSING')}\n"
                            "-----"
                        )

                # Add safe inspection logging
                logger.info(f"Initial raw_docs structure: {type(raw_docs)}")
                if raw_docs:
                    logger.info(f"raw_docs keys: {raw_docs.keys()}")
                    if "documents" in raw_docs:
                        doc_count = len(raw_docs["documents"])
                        logger.info(f"Found {doc_count} documents")
                        if doc_count > 0:
                            logger.info(f"First document sample: {raw_docs['documents'][0][:100]}...")
                        else:
                            logger.info("Documents list is empty")
                    else:
                        logger.info("No 'documents' key found in raw_docs")
                        
                    if "metadatas" in raw_docs:
                        meta_count = len(raw_docs["metadatas"]) if raw_docs["metadatas"] else 0
                        logger.info(f"Found {meta_count} metadata entries")
                        if meta_count > 0:
                            logger.info(f"First metadata sample: {raw_docs['metadatas'][0]}")
                    else:
                        logger.info("No 'metadatas' key found in raw_docs")

                # If no results with brand filter, try without filter to info
                if brand_key and (not raw_docs or not raw_docs.get("documents")):
                    logger.info(f"No documents found with brand filter, trying unfiltered")
                    raw_docs = vector_db._collection.get(
                        include=["documents", "metadatas"]
                    )
                    
                    # Add safe inspection logging for unfiltered results
                    logger.info(f"Unfiltered raw_docs structure: {type(raw_docs)}")
                    if raw_docs:
                        logger.info(f"Unfiltered raw_docs keys: {raw_docs.keys()}")
                        if "documents" in raw_docs:
                            doc_count = len(raw_docs["documents"])
                            logger.info(f"Found {doc_count} unfiltered documents")
                            if doc_count > 0:
                                logger.info(f"First unfiltered document: {raw_docs['documents'][0][:100]}...")
                        if "metadatas" in raw_docs:
                            meta_count = len(raw_docs["metadatas"]) if raw_docs["metadatas"] else 0
                            logger.info(f"Found {meta_count} unfiltered metadata entries")
                    
                    if raw_docs and raw_docs.get("documents"):
                        found_brands = set(m.get('brand') for m in raw_docs["metadatas"] if m)
                        logger.warning(
                            f"Found {len(raw_docs['documents'])} documents but none matched brand '{brand_key}'. "
                            f"Available brands: {found_brands}"
                        )
                
                # Process documents with metadata validation
                vector_db._cached_documents = []
                valid_count = 0
                
                for content, meta in zip(raw_docs["documents"], raw_docs["metadatas"]):
                    if not content:
                        continue
                        
                    metadata = meta or {}
                    metadata.setdefault("brand", brand_key)
                    metadata.update({
                        'brand': metadata.get('brand', brand_key),
                        'doc_type': metadata.get('doc_type', 'general')
                    })
                    
                    # Auto-detect document types
                    if 'price' in content.lower() or 'cost' in content.lower():
                        metadata['doc_type'] = 'pricing'
                    
                    vector_db._cached_documents.append(
                        Document(page_content=content, metadata=metadata)
                    )
                    valid_count += 1
                
                load_time = time.time() - start_load
                logger.info(
                    f"Loaded {valid_count} documents {brand_context} "
                    f"in {load_time:.2f}s"
                )
                
                if valid_count == 0:
                    raise ValueError("No valid documents after processing")
                    
            except Exception as load_error:
                logger.error(f"Document loading failed: {str(load_error)}")
                raise RuntimeError(f"Could not load documents {brand_context}")

        # Brand metadata analysis
        if brand_key:
            found_brands = set(d.metadata.get('brand') for d in vector_db._cached_documents)
            logger.debug(f"Brand metadata - Filter: '{norm_brand}', Found: {found_brands}")
            
            missing_brand = [d for d in vector_db._cached_documents if not d.metadata.get('brand')]
            if missing_brand:
                logger.warning(f"{len(missing_brand)} documents missing brand metadata")

        # BM25 Retriever initialization
        start_bm25 = time.time()
        bm25_docs = [
            doc for doc in vector_db._cached_documents
            if not norm_brand or _normalize_brand(doc.metadata.get('brand')) == norm_brand
        ]
        
        if not bm25_docs:
            available_brands = set(_normalize_brand(d.metadata.get('brand')) 
                            for d in vector_db._cached_documents)
            raise ValueError(
                f"No documents matched brand filter '{norm_brand}'. "
                f"Available brands: {available_brands}"
            )
        
        try:
            bm25_retriever = PersistentBM25Retriever.from_documents(
                documents=bm25_docs,
                persist_path=bm25_persist_path,
                k=min(len(bm25_docs), effective_k),  # Changed from default_k to effective_k
                llm=llm,
                precompute_scores=True
            )
            logger.info(
                f"BM25 ready with {len(bm25_docs)} docs "
                f"in {time.time()-start_bm25:.2f}s"
            )
        except Exception as bm25_error:
            logger.error(f"BM25 initialization failed: {str(bm25_error)}")
            raise RuntimeError("BM25 index creation failed")

        # Vector Retriever with MMR
        start_vector = time.time()
        vector_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": effective_k,
                "filter": {"brand": brand_key} if brand_key else None,
                "score_threshold": score_threshold,  # Use the parameter here
                "fetch_k": min(50, len(vector_db._cached_documents) * 2),
                "lambda_mult": 0.5
            }
        )
        logger.info(f"Vector retriever ready in {time.time()-start_vector:.2f}s")

        # Hybrid retrieval when possible
        if enable_hybrid and len(bm25_docs) > 0:
            return EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[bm25_weight, vector_weight],
                c=1
            )
        logger.warning(f"Using vector-only retriever {brand_context}")
        return vector_retriever

    except Exception as e:
        logger.error(f"Retriever creation failed {brand_context}", exc_info=True)
        
        # Enhanced fallback with metadata diagnostics
        if hasattr(vector_db, '_cached_documents'):
            logger.debug(
                "Document metadata sample:\n" +
                "\n".join(
                    f"Brand: {d.metadata.get('brand')} | "
                    f"Type: {d.metadata.get('doc_type')} | "
                    f"Content: {d.page_content[:50]}..."
                    for d in vector_db._cached_documents[:3]
                )
            )
            
            # Try direct string comparison as fallback
            fallback_docs = [
                d for d in vector_db._cached_documents
                if not brand_key or str(d.metadata.get('brand', '')).lower() == str(brand_key).lower()
            ]
            
            if fallback_docs:
                logger.warning(f"Attempting fallback with {len(fallback_docs)} docs")
                try:
                    return PersistentBM25Retriever.from_documents(
                        documents=fallback_docs,
                        k=min(len(fallback_docs), default_k)
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback failed: {str(fallback_error)}")
        
        raise RuntimeError(
            f"Retriever failed {brand_context}. "
            f"Total docs: {len(getattr(vector_db, '_cached_documents', []))}, "
            f"Error: {str(e)}"
        )