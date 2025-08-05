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
import getpass
import sqlite3
import fcntl  # For Unix file locking

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
    """Manages brand-specific vector stores with enhanced permission handling"""
    
    _stores: Dict[str, Chroma] = {}
    _lock = asyncio.Lock()  # Global lock for thread-safe operations
    
    @classmethod
    def _acquire_file_lock(cls, path: Path) -> bool:
        """Acquire an exclusive lock on a file"""
        try:
            if os.name == 'nt':
                # Windows implementation
                import msvcrt
                fd = os.open(path, os.O_RDWR | os.O_CREAT)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    return True
                except IOError:
                    os.close(fd)
                    return False
            else:
                # Unix implementation
                fd = os.open(path, os.O_RDWR | os.O_CREAT)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    os.close(fd)
                    return False
        except Exception as e:
            logger.error(f"Failed to acquire lock for {path}: {e}")
            return False

    @classmethod
    def _release_file_lock(cls, fd: int):
        """Release a previously acquired file lock"""
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")

    @classmethod
    def _is_overlayfs(cls, path: Path) -> bool:
        """Check if path is on OverlayFS"""
        try:
            with open('/proc/mounts') as f:
                mounts = f.read()
            return 'overlay' in mounts and str(path) in mounts
        except Exception:
            return False

    @classmethod
    def _ensure_docker_writable(cls, path: Path) -> bool:
        """Special handling for Docker/OverlayFS"""
        if not cls._is_overlayfs(path):
            return True
            
        try:
            test_file = path / f".tmp_test_{os.getpid()}"
            test_file.touch()
            test_file.unlink()
            os.chmod(path, 0o777)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Docker write test failed: {str(e)}")
            return False

    @classmethod
    def _get_store_config(cls, brand_key: str) -> VectorStoreConfig:
        """Get configuration for brand-specific store with strict separation"""
        if not brand_key:
            raise ValueError("brand_key is required")
        
        collection_name = f"{Config.COLLECTION_NAME}_{Config.sanitize_name(brand_key)}"
        persist_path = Config.get_brand_vector_store_path(brand_key)
        
        if cls._is_overlayfs(persist_path):
            logger.info(f"Detected OverlayFS for {brand_key} store")
            if not cls._ensure_docker_writable(persist_path.parent):
                temp_path = Path(f"/tmp/vector_stores/{brand_key}")
                logger.warning(f"Using fallback storage at {temp_path}")
                persist_path = temp_path

        persist_path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(persist_path.parent, 0o700)
        
        return VectorStoreConfig(
            persist_path=persist_path,
            collection_name=collection_name,
            embedding_function=cached_embeddings
        )

    @classmethod
    def _ensure_directory_permissions(cls, path: Path) -> bool:
        """Ensure directory exists with correct permissions"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            os.chmod(path, 0o777)
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o777)
                for f in files:
                    try:
                        os.chmod(os.path.join(root, f), 0o666)
                    except Exception:
                        continue
            
            test_file = path / f".permission_test_{os.getpid()}"
            test_file.touch()
            test_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to set permissions for {path}: {str(e)}")
            return False

    @classmethod
    def _check_sqlite_writable(cls, path: Path) -> bool:
        """Verify SQLite database is writable"""
        test_db = path / "permission_test.db"
        try:
            with sqlite3.connect(test_db) as conn:
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.execute("DROP TABLE test")
            test_db.unlink()
            return True
        except Exception as e:
            logger.error(f"SQLite write test failed: {str(e)}")
            return False
    
    @classmethod
    def _validate_config(cls, config: VectorStoreConfig):
        """Validate store configuration"""
        if not isinstance(config.persist_path, Path):
            raise TypeError("persist_path must be Path object")
        if not config.collection_name:
            raise ValueError("collection_name cannot be empty")
        if not isinstance(config.embedding_function, CacheBackedEmbeddings):
            raise TypeError("Invalid embedding function type")

    @classmethod
    def _force_unlock_windows(cls, path: Path):
        """Windows-specific file unlocking"""
        if os.name != 'nt':
            return

        path = path.resolve()
        try:
            from ctypes import windll
            FILE_SHARE_DELETE = 0x00000004
            handle = windll.kernel32.CreateFileW(
                str(path),
                0x80000000,
                FILE_SHARE_DELETE,
                None,
                3,
                0x80000000,
                None
            )
            if handle != -1:
                windll.kernel32.CloseHandle(handle)

            for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                try:
                    if any(str(path).lower() == f.path.lower() for f in proc.open_files()):
                        proc.kill()
                        logger.warning(f"Killed process {proc.pid} holding {path}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            temp_path = path.with_name(f"{path.name}.temp")
            try:
                path.rename(temp_path)
                temp_path.unlink()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Windows unlock failed for {path}: {e}")

    @classmethod
    def _ensure_writable_directory(cls, path: Path) -> bool:
        """Ensure directory is writable"""
        for attempt in range(3):
            try:
                path.mkdir(parents=True, exist_ok=True)
                os.chmod(path, 0o777)
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o666)
                return True
            except Exception as e:
                logger.error(f"Directory permission setting failed (attempt {attempt+1}): {e}")
                time.sleep(1)
        return False

    @classmethod
    def _ensure_file_ownership(cls, path: Path):
        """Ensure files are owned by current user (Unix)"""
        try:
            uid = os.getuid()
            gid = os.getgid()
            os.chown(path, uid, gid)
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    os.chown(os.path.join(root, d), uid, gid)
                for f in files:
                    os.chown(os.path.join(root, f), uid, gid)
        except Exception as e:
            logger.warning(f"Could not change ownership: {e}")

    @classmethod
    def _check_database_writable(cls, path: Path) -> bool:
        """Verify database location is writable"""
        test_file = path / "permission_test.db"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Write test failed: {e}")
            return False

    @classmethod
    async def cleanup_stores(cls):
        """Clean up all brand stores with enhanced permission handling"""
        async with cls._lock:
            store_path = Config.PERSIST_DIRECTORY
            for attempt in range(3):
                try:
                    if os.name == 'nt':
                        for root, _, files in os.walk(store_path):
                            for file in files:
                                if file.endswith(('.sqlite3', '.parquet')):
                                    cls._force_unlock_windows(Path(root) / file)
                    
                    if store_path.exists():
                        cls._ensure_writable_directory(store_path)
                        if os.name != 'nt':
                            cls._ensure_file_ownership(store_path)
                    
                    shutil.rmtree(store_path, ignore_errors=True)
                    if not store_path.exists():
                        return True
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    logger.error(f"Cleanup attempt {attempt+1} failed: {e}")
            
            return False

    @classmethod
    def _prepare_documents(cls, raw_documents, brand_key: str):
        """Improved document preparation with strict metadata enforcement"""
        processed = []
        
        for doc in raw_documents:
            if not isinstance(doc, Document):
                doc = Document(
                    page_content=str(doc),
                    metadata=getattr(doc, 'metadata', {})
                )
            doc.metadata = doc.metadata or {}
            doc.metadata.update({
                'brand': brand_key,
                'timestamp': datetime.now().isoformat(),
                'doc_type': doc.metadata.get('doc_type', 'general')
            })
            
            if 'price' in doc.page_content.lower():
                doc.metadata['doc_type'] = 'pricing'
                doc.metadata['is_pricing'] = True
                
            processed.append(doc)
        
        return processed
    
    @classmethod
    async def ensure_persistence(cls, brand_key: str):
        """Ensure the store is properly persisted"""
        async with cls._lock:
            if brand_key in cls._stores:
                try:
                    cls._stores[brand_key].persist()
                    return True
                except Exception as e:
                    logger.error(f"Failed to persist store for {brand_key}: {str(e)}")
            return False
    
    @classmethod
    def _is_file_locked(cls, path: Path) -> bool:
        """Check if a file is locked"""
        try:
            with open(path, 'a') as f:
                pass
            return False
        except IOError:
            return True
        
    @classmethod
    async def repair_database(cls, brand_key: str) -> bool:
        """SQLite-specific database repair"""
        async with cls._lock:
            config = cls._get_store_config(brand_key)
            
            if not config.persist_path.exists():
                return False
                
            cls._ensure_directory_permissions(config.persist_path)
            cls._ensure_sqlite_permissions(config.persist_path)
            
            db_file = config.persist_path / "chroma.sqlite3"
            if not db_file.exists():
                return True
                
            try:
                with sqlite3.connect(str(db_file)) as conn:
                    conn.execute("PRAGMA quick_check;")
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    conn.execute("VACUUM;")
                return True
            except sqlite3.Error as e:
                logger.error(f"SQLite repair failed: {str(e)}")
                try:
                    backup_path = db_file.with_suffix('.bak')
                    db_file.rename(backup_path)
                    logger.warning(f"Renamed corrupted database to {backup_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to backup corrupted database: {str(e)}")
                    return False
                
    @classmethod
    def _validate_sqlite_schema(cls, path: Path) -> bool:
        """Verify critical Chroma tables exist"""
        db_file = path / "chroma.sqlite3"
        if not db_file.exists():
            return False
            
        required_tables = {"collections", "embeddings", "acquire_write"}
        
        try:
            with sqlite3.connect(str(db_file)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                existing_tables = {row[0] for row in cursor.fetchall()}
                return required_tables.issubset(existing_tables)
        except Exception:
            return False
        
    @classmethod
    async def clear_vector_store(cls, brand_key: str) -> bool:
        """Clear all documents from an existing vector store"""
        async with cls._lock:
            try:
                store = cls._stores.get(brand_key)
                if store is None:
                    config = cls._get_store_config(brand_key)
                    store = Chroma(
                        persist_directory=str(config.persist_path),
                        embedding_function=config.embedding_function,
                        collection_name=config.collection_name
                    )
                
                collection = store._collection
                if collection is not None:
                    ids = collection.get()["ids"]
                    if ids:
                        collection.delete(ids=ids)
                    return True
                
                return False
            except Exception as e:
                logger.error(f"Failed to clear vector store for {brand_key}: {str(e)}")
                return False
            
    @classmethod
    async def rebuild_vector_store(cls, brand_key: str, documents: List[Document]) -> bool:
        """Rebuild by clearing existing store and adding new documents"""
        async with cls._lock:
            try:
                if not await cls.clear_vector_store(brand_key):
                    logger.error("Failed to clear existing store")
                    return False

                processed_docs = cls._prepare_documents(documents, brand_key)
                validated_docs = validate_documents(processed_docs)
                
                if not validated_docs:
                    logger.error("No valid documents after validation")
                    return False

                config = cls._get_store_config(brand_key)
                store = cls._stores.get(brand_key)
                if store is None:
                    store = Chroma(
                        persist_directory=str(config.persist_path),
                        embedding_function=config.embedding_function,
                        collection_name=config.collection_name
                    )
                    cls._stores[brand_key] = store

                store.add_documents(validated_docs)
                logger.info(f"Successfully rebuilt store for {brand_key} with {len(validated_docs)} documents")
                return True

            except Exception as e:
                logger.error(f"Rebuild failed for {brand_key}: {str(e)}")
                return False

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_vector_store(cls, documents: List[Document], brand_key: str, recreate: bool = False) -> Chroma:
        """Create or recreate vector store"""
        async with cls._lock:
            try:
                if recreate:
                    if not await cls.rebuild_vector_store(brand_key, documents):
                        raise RuntimeError(f"Failed to rebuild store for {brand_key}")
                    return cls._stores[brand_key]
                
                config = cls._get_store_config(brand_key)
                processed_docs = cls._prepare_documents(documents, brand_key)
                validated_docs = validate_documents(processed_docs)
                
                store = Chroma.from_documents(
                    documents=validated_docs,
                    embedding=config.embedding_function,
                    persist_directory=str(config.persist_path),
                    collection_name=config.collection_name
                )
                cls._stores[brand_key] = store
                return store

            except Exception as e:
                logger.error(f"Store creation failed for {brand_key}: {type(e).__name__} - {str(e)}")
                if config.persist_path.exists():
                    shutil.rmtree(config.persist_path, ignore_errors=True)
                raise

    @classmethod
    async def get_vector_store(cls, brand_key: str, create_if_missing: bool = False, 
                            documents: Optional[List[Document]] = None) -> Optional[Chroma]:
        """Get initialized store instance for specific brand"""
        async with cls._lock:
            if brand_key in cls._stores:
                store = cls._stores[brand_key]
                if hasattr(store, '_collection') and brand_key not in store._collection.name:
                    logger.warning(f"Store collection name mismatch for {brand_key}")
                    del cls._stores[brand_key]
                    return await cls.get_vector_store(brand_key, create_if_missing, documents)
                return store
                
            config = cls._get_store_config(brand_key)
            
            try:
                store = Chroma(
                    persist_directory=str(config.persist_path),
                    embedding_function=config.embedding_function,
                    collection_name=config.collection_name,
                    collection_metadata={"brand": brand_key}
                )
                
                if hasattr(store, '_collection'):
                    if store._collection.count() > 0:
                        collection_brand = store._collection.metadata.get('brand')
                        if collection_brand != brand_key:
                            logger.error(f"Collection brand mismatch: expected {brand_key}, got {collection_brand}")
                            if create_if_missing:
                                logger.info(f"Recreating store for {brand_key}")
                                await cls.delete_vector_store(brand_key)
                                return await cls.get_vector_store(brand_key, True, documents)
                            return None
                        
                        cls._stores[brand_key] = store
                        logger.info(f"Loaded store for {brand_key} with {store._collection.count()} docs")
                        return store
                        
                if create_if_missing:
                    if documents:
                        return await cls.create_vector_store(documents, brand_key)
                    else:
                        from data_loader import DocumentLoader
                        docs = await asyncio.to_thread(DocumentLoader.load_brand_documents, brand_key)
                        return await cls.create_vector_store(docs, brand_key)
                        
                return None
                
            except Exception as e:
                logger.error(f"Failed to load store for {brand_key}: {str(e)}")
                if await cls.repair_database(brand_key):
                    return await cls.get_vector_store(brand_key, create_if_missing, documents)
                return None
            
    @classmethod
    async def delete_vector_store(cls, brand_key: str) -> bool:
        """Completely delete a vector store"""
        async with cls._lock:
            config = cls._get_store_config(brand_key)
            
            store = cls._stores.pop(brand_key, None)
            if store:
                try:
                    if hasattr(store, "_client"):
                        if hasattr(store._client, "stop"):
                            store._client.stop()
                        elif hasattr(store._client, "reset"):
                            store._client.reset()
                except Exception as e:
                    logger.warning(f"Client cleanup warning: {str(e)}")

            if config.persist_path.exists():
                try:
                    temp_deletion_path = config.persist_path.with_name(f"{config.persist_path.name}.deleting")
                    config.persist_path.rename(temp_deletion_path)
                    
                    def _bg_delete(path):
                        try:
                            shutil.rmtree(path, ignore_errors=True)
                        except Exception as e:
                            logger.warning(f"Background delete failed: {str(e)}")

                    await asyncio.to_thread(_bg_delete, temp_deletion_path)
                    return True
                except Exception as e:
                    logger.error(f"Atomic delete failed: {str(e)}")
                    shutil.rmtree(config.persist_path, ignore_errors=True)
            
            return not config.persist_path.exists()

    @classmethod
    async def repair_store_permissions(cls, brand_key: str):
        """Repair permissions for a store"""
        async with cls._lock:
            config = cls._get_store_config(brand_key)
            if config.persist_path.exists():
                cls._ensure_writable_directory(config.persist_path)
                if os.name != 'nt':
                    cls._ensure_file_ownership(config.persist_path)
                logger.info(f"Repaired permissions for {brand_key} store")
                return True
            return False
        
    @classmethod
    def _ensure_sqlite_permissions(cls, path: Path) -> bool:
        """Ensure SQLite database files have correct permissions"""
        try:
            db_file = path / "chroma.sqlite3"
            if db_file.exists():
                os.chmod(db_file, 0o666)
            
            for pattern in ["*-wal", "*-shm", "*-journal"]:
                for temp_file in path.glob(pattern):
                    try:
                        os.chmod(temp_file, 0o666)
                    except Exception:
                        continue
            return True
        except Exception as e:
            logger.error(f"Failed to set SQLite permissions: {str(e)}")
            return False

    @classmethod
    async def initialize_all_stores(cls, max_retries: int = 3) -> bool:
        """Initialize all brand stores"""
        from data_loader import DocumentLoader
        
        async def _init_store(brand_key: str):
            for attempt in range(max_retries):
                try:
                    config = cls._get_store_config(brand_key)
                    if config.persist_path.exists():
                        await cls.repair_store_permissions(brand_key)

                    docs = await asyncio.to_thread(
                        DocumentLoader.load_brand_documents,
                        brand_key=brand_key
                    )
                    store = await cls.get_vector_store(brand_key, create_if_missing=True)
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
    async def health_check(cls, brand_key: str) -> dict:
        """Enhanced health check for specific brand store"""
        async with cls._lock:
            store = await cls.get_vector_store(brand_key)
            config = cls._get_store_config(brand_key)
            
            status = {
                "exists": store is not None,
                "document_count": store._collection.count() if store else 0,
                "path": str(store._persist_directory) if store else None,
                "collection": store._collection.name if store else None,
                "timestamp": datetime.now().isoformat(),
                "permissions": None,
                "writable": False
            }
            
            if store and config.persist_path.exists():
                try:
                    stat_info = os.stat(config.persist_path)
                    status["permissions"] = oct(stat_info.st_mode)[-3:]
                    status["writable"] = cls._check_database_writable(config.persist_path)
                    test_query = store.similarity_search("test", k=1)
                    status["query_test"] = bool(test_query)
                except Exception as e:
                    status["query_test"] = False
                    status["query_error"] = str(e)
            
            return status