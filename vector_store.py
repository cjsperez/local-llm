from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_ollama import OllamaEmbeddings
from config import Config
from pathlib import Path
import shutil
import gc
import logging
import time
import re
import hashlib
import asyncio
from datetime import datetime
from typing import Optional, List, Union, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from dataclasses import dataclass
import psutil
import os
import concurrent.futures
import json

logger = logging.getLogger(__name__)

class SanitizedFileStore(LocalFileStore):
    """Custom file store with sanitized keys"""
    def _get_full_path(self, key: str) -> Path:
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', key)
        return super()._get_full_path(safe_key)

def get_cached_embeddings():
    """Create cached embeddings with sanitized keys"""
    embedding_cache = SanitizedFileStore(str(Config.EMBEDDING_CACHE_PATH))
    return CacheBackedEmbeddings.from_bytes_store(
        OllamaEmbeddings(
            model=Config.EMBEDDING_MODEL,
            base_url=Config.OLLAMA_BASE_URL
        ),
        embedding_cache,
        namespace=Config.sanitize_name(Config.EMBEDDING_MODEL)
    )

cached_embeddings = get_cached_embeddings()

def validate_documents(documents: List) -> List[Document]:
    """Ensure documents are properly formatted with comprehensive checks"""
    if not documents:
        raise ValueError("No documents provided for vector store")

    validated = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, Document):
            logger.error(f"[validate_documents] Skipping invalid doc at index {i}: {type(doc)} - {repr(doc)}")
            continue

        try:
            if not doc.page_content or not str(doc.page_content).strip():
                logger.warning(f"Found empty document at index {i}: {doc.metadata.get('title', 'Untitled')}")
                continue

            validated.append(Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            ))
        except Exception as e:
            logger.error(f"[validate_documents] Failed to validate doc at index {i}: {e}")
            continue

    return validated

@dataclass
class VectorStoreConfig:
    persist_path: Path
    collection_name: str
    embedding_function: CacheBackedEmbeddings

class VectorStoreManager:
    """Manages brand-specific vector stores only"""
    
    _stores: Dict[str, Chroma] = {}
    
    @classmethod
    def _get_store_config(cls, brand_key: str) -> VectorStoreConfig:
        """Get configuration for brand-specific store"""
        if not brand_key:
            raise ValueError("brand_key is required")
            
        return VectorStoreConfig(
            persist_path=Config.get_brand_vector_store_path(brand_key),
            collection_name=Config.get_collection_name(brand_key),
            embedding_function=cached_embeddings
        )
    
    @classmethod
    def _validate_config(cls, config: VectorStoreConfig):
        """Validate store configuration before operations"""
        if not isinstance(config.persist_path, Path):
            raise TypeError("persist_path must be Path object")
        if not config.collection_name:
            raise ValueError("collection_name cannot be empty")
        if not isinstance(config.embedding_function, CacheBackedEmbeddings):
            raise TypeError("Invalid embedding function type")

    @classmethod
    def _force_unlock_windows(cls, path: Path):
        """Aggressive Windows file unlock using multiple methods"""
        if os.name != 'nt':
            return

        path = path.resolve()
        try:
            # Method 1: Windows API
            from ctypes import windll
            FILE_SHARE_DELETE = 0x00000004
            handle = windll.kernel32.CreateFileW(
                str(path),
                0x80000000,  # GENERIC_READ
                FILE_SHARE_DELETE,
                None,
                3,  # OPEN_EXISTING
                0x80000000,  # FILE_FLAG_DELETE_ON_CLOSE
                None
            )
            if handle != -1:
                windll.kernel32.CloseHandle(handle)

            # Method 2: Kill holding processes
            for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                try:
                    if any(str(path).lower() == f.path.lower() for f in proc.open_files()):
                        proc.kill()
                        logger.warning(f"Killed process {proc.pid} holding {path}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Method 3: Rename trick
            temp_path = path.with_name(f"{path.name}.temp")
            try:
                path.rename(temp_path)
                temp_path.unlink()
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Windows unlock failed for {path}: {e}")

    @classmethod
    def cleanup_stores(cls):
        """Clean up all brand stores"""
        store_path = Config.PERSIST_DIRECTORY
        for attempt in range(3):
            try:
                if os.name == 'nt':
                    for root, _, files in os.walk(store_path):
                        for file in files:
                            if file.endswith(('.sqlite3', '.parquet')):
                                cls._force_unlock_windows(Path(root) / file)
                
                shutil.rmtree(store_path, ignore_errors=True)
                if not store_path.exists():
                    return True
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Cleanup attempt {attempt+1} failed: {e}")
        
        return False

    @classmethod
    def _prepare_documents(cls, raw_documents):
        """Convert various input formats to Document objects"""
        from langchain_core.documents.base import Document
        processed = []
        
        for doc in raw_documents:
            if isinstance(doc, Document):
                processed.append(doc)
            elif isinstance(doc, dict):
                if 'page_content' in doc:
                    content = doc['page_content']
                    metadata = doc.get('metadata', {})
                elif 'question' in doc and 'answer' in doc:
                    content = f"Question: {doc['question']}\nAnswer: {doc['answer']}"
                    metadata = {
                        'category': doc.get('category', 'unknown'),
                        'source': doc.get('source', 'unknown')
                    }
                else:
                    content = str(doc)
                    metadata = {k: v for k, v in doc.items() if k != 'content'}
                
                processed.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            else:
                processed.append(Document(
                    page_content=str(doc),
                    metadata={'source': 'unknown'}
                ))
        
        return processed
    
    @classmethod
    def ensure_persistence(cls, brand_key: str):
        """Ensure the store is properly persisted"""
        if brand_key in cls._stores:
            try:
                cls._stores[brand_key].persist()
                return True
            except Exception as e:
                logger.error(f"Failed to persist store for {brand_key}: {str(e)}")
        return False
    
    @classmethod
    def _ensure_writable_directory(cls, path: Path) -> bool:
        """Ensure the directory exists and is writable"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Test write permission
            test_file = path / ".permission_test"
            test_file.touch()
            test_file.unlink()
            
            return True
        except Exception as e:
            logger.error(f"Directory {path} is not writable: {str(e)}")
            return False

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def create_vector_store(cls, documents: List[Document], brand_key: str, recreate: bool = False) -> Chroma:
        """Create/recreate vector store for specific brand"""
        config = cls._get_store_config(brand_key)

        if not cls._ensure_writable_directory(config.persist_path):
            raise PermissionError(f"Cannot write to {config.persist_path}")
        
        try:
            if recreate:
                cls.cleanup_stores()
            
            # Explicitly delete existing store if recreating
            if recreate and config.persist_path.exists():
                cls.delete_vector_store(brand_key)
                
            processed_docs = cls._prepare_documents(documents)
            validated_docs = validate_documents(processed_docs)
            
            if not validated_docs:
                raise ValueError("No valid documents after validation")

            # Create with explicit collection name
            store = Chroma.from_documents(
                documents=validated_docs,
                embedding=config.embedding_function,
                persist_directory=str(config.persist_path),
                collection_name=config.collection_name
            )

            # Verify creation
            if not hasattr(store, '_collection'):
                raise RuntimeError("Store creation failed - no collection")
            
            # Force persistence
            # store.persist()
            
            cls._stores[brand_key] = store
            logger.info(f"Created store for {brand_key} with {store._collection.count()} docs")
            return store
            
        except Exception as e:
            logger.error(f"Store creation failed for {brand_key}: {type(e).__name__} - {str(e)}")
            cls.cleanup_stores()
            raise

    @classmethod
    def get_vector_store(cls, brand_key: str, create_if_missing: bool = False, 
                        documents: Optional[List[Document]] = None) -> Optional[Chroma]:
        """Get initialized store instance for specific brand"""
        if brand_key in cls._stores:
            return cls._stores[brand_key]
            
        config = cls._get_store_config(brand_key)
        
        try:
            # First try loading with collection name
            store = Chroma(
                persist_directory=str(config.persist_path),
                embedding_function=config.embedding_function,
                collection_name=config.collection_name
            )
            
            # Verify the collection exists
            if hasattr(store, '_collection') and store._collection.count() > 0:
                cls._stores[brand_key] = store
                logger.info(f"Loaded store for {brand_key} with {store._collection.count()} docs")
                return store
                
            # If collection is empty or doesn't exist, handle accordingly
            if create_if_missing:
                if documents:
                    return cls.create_vector_store(documents, brand_key)
                else:
                    from data_loader import DocumentLoader
                    docs = DocumentLoader.load_brand_documents(brand_key)
                    return cls.create_vector_store(docs, brand_key)
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to load store for {brand_key}: {str(e)}")
            if "does not exist" in str(e):
                if create_if_missing:
                    return cls.get_vector_store(brand_key, create_if_missing=True, documents=documents)
            return None
        
    @classmethod
    def delete_vector_store(cls, brand_key: str) -> bool:
        """Completely delete a vector store and its references"""
        if not brand_key:
            raise ValueError("brand_key is required")

        config = cls._get_store_config(brand_key)
        success = False

        # 1. Remove from memory cache if present
        if brand_key in cls._stores:
            del cls._stores[brand_key]
            logger.info(f"Removed {brand_key} store from memory cache")

        # 2. Delete persistent files
        try:
            if config.persist_path.exists():
                # Windows-specific unlock handling
                if os.name == 'nt':
                    cls._force_unlock_windows(config.persist_path)

                # Delete the directory
                shutil.rmtree(config.persist_path, ignore_errors=True)
                
                # Verify deletion
                if not config.persist_path.exists():
                    success = True
                    logger.info(f"Deleted vector store files for {brand_key}")
                else:
                    logger.error(f"Failed to completely delete {config.persist_path}")
        except Exception as e:
            logger.error(f"Error deleting vector store files: {str(e)}")
            success = False

        # 3. Clean up brand documents references (if Config has this file)
        try:
            if hasattr(Config, 'BRAND_DOCUMENTS_FILE') and Config.BRAND_DOCUMENTS_FILE.exists():
                with open(Config.BRAND_DOCUMENTS_FILE, 'r+') as f:
                    brand_docs = json.load(f)
                    if brand_key in brand_docs:
                        logger.info(f"Preserved document references for {brand_key}")
        except Exception as e:
            logger.error(f"Error updating brand documents file: {str(e)}")

        # 4. Force garbage collection to clean up any remaining resources
        gc.collect()

        return success

    @classmethod
    async def initialize_all_stores(cls, max_retries: int = 3) -> bool:
        """Initialize all brand stores"""
        from data_loader import DocumentLoader
        
        async def _init_store(brand_key: str):
            for attempt in range(max_retries):
                try:
                    docs = await asyncio.to_thread(
                        DocumentLoader.load_brand_documents,
                        brand_key=brand_key
                    )
                    store = cls.get_vector_store(brand_key, create_if_missing=True)
                    if not store or not hasattr(store, '_collection'):
                        raise RuntimeError("Store initialization failed")
                    return True
                except Exception as e:
                    logger.error(f"Attempt {attempt+1}/{max_retries} failed for {brand_key}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
            return False

        try:
            brands = await asyncio.to_thread(DocumentLoader.get_all_brands)
            results = await asyncio.gather(
                *[_init_store(brand) for brand in brands],
                return_exceptions=True
            )
            
            failures = [
                brand for brand, result in zip(brands, results)
                if isinstance(result, Exception) or not result
            ]
            
            if failures:
                logger.error(f"Failed to initialize stores for brands: {failures}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Critical initialization failure: {str(e)}")
            return False

    @classmethod
    def health_check(cls, brand_key: str) -> dict:
        """Enhanced health check for specific brand store"""
        store = cls.get_vector_store(brand_key)
        status = {
            "exists": store is not None,
            "document_count": store._collection.count() if store else 0,
            "path": str(store._persist_directory) if store else None,
            "collection": store._collection.name if store else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Additional validation checks
        if store:
            try:
                test_query = store.similarity_search("test", k=1)
                status["query_test"] = bool(test_query)
            except Exception as e:
                status["query_test"] = False
                status["query_error"] = str(e)
        
        return status