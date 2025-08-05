from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import PersistentClient  # New import
from config import Config
from logger import log
from pathlib import Path
import shutil
import time
import os
from typing import Optional
import psutil
import sys

from vector_store import VectorStoreManager
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(name)s - %(levelname)s]: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

def load_vector_store(brand_key: Optional[str] = None, retry_count: int = 2) -> Chroma:
    """Load vector store with proper error handling"""
    store_path = Config.get_vector_store_path(brand_key)
    last_error = None
    embeddings = OllamaEmbeddings(
        model=Config.EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL
    )

    for attempt in range(1, retry_count + 2):  # +2 to include initial try
        try:
            # First try normal load
            logger.info(f"Brand Key: {brand_key}")
            if attempt == 1:
                return _initialize_chroma_store(store_path, brand_key, embeddings)
            
            # Subsequent attempts use fresh start
            logger.warning(f"Attempt {attempt}: Fresh start...")
            return _initialize_chroma_store(
                store_path, 
                brand_key, 
                embeddings, 
                fresh_start=True
            )
            
        except Exception as e:
            last_error = str(e)
            logger.error(f"Attempt {attempt} failed: {last_error}")
            if attempt <= retry_count:
                time.sleep(2 ** attempt)  # Exponential backoff

    raise RuntimeError(
        f"Failed to load vector store after {retry_count} attempts. "
        f"Last error: {last_error}"
    )

def _prepare_store_directory(store_path: Path) -> None:
    """Enhanced Windows-compatible directory preparation with permission handling"""
    try:
        # 1. Ensure parent directory exists with explicit permissions
        try:
            store_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
        except TypeError:  # Fallback for Python versions without mode support
            store_path.parent.mkdir(parents=True, exist_ok=True)
            if os.name == 'nt':
                try:
                    os.chmod(store_path.parent, 0o777)
                except PermissionError:
                    log("Warning: Could not set directory permissions", level="WARNING")

        # 2. Clean existing directory with Windows-specific handling
        if store_path.exists():
            _windows_safe_cleanup(store_path)
            time.sleep(1)  # Increased delay for Windows lock release

        # 3. Create target directory with explicit permissions
        try:
            store_path.mkdir(exist_ok=True, mode=0o777)
        except TypeError:  # Fallback
            store_path.mkdir(exist_ok=True)
            if os.name == 'nt':
                try:
                    os.chmod(store_path, 0o777)
                except PermissionError:
                    log("Warning: Could not set directory permissions", level="WARNING")

        # 4. Verify directory was actually created
        if not store_path.exists():
            raise RuntimeError(f"Directory creation failed: {store_path}")

        # 5. Windows-specific post-creation verification
        if os.name == 'nt':
            _verify_windows_permissions(store_path)

    except Exception as e:
        log(f"Directory preparation failed: {str(e)}", level="ERROR")
        _log_windows_solution_hint(store_path)
        raise RuntimeError(f"Could not prepare vector store directory: {str(e)}")

def _windows_safe_cleanup(store_path: Path) -> None:
    """Specialized cleanup for Windows systems"""
    if os.name != 'nt':
        shutil.rmtree(store_path, ignore_errors=True)
        return

    # Windows-specific cleanup sequence
    for attempt in range(3):
        try:
            # Release locks first
            _release_windows_locks(store_path)
            
            # Try standard deletion
            shutil.rmtree(store_path, ignore_errors=True)
            if not store_path.exists():
                return
                
            # Force delete remaining files
            for f in store_path.glob('*'):
                try:
                    if f.is_file():
                        os.chmod(f, 0o777)  # Ensure deletable
                        os.unlink(f)
                except Exception as e:
                    log(f"Failed to delete {f}: {str(e)}", level="DEBUG")
            
            time.sleep(0.5 * (attempt + 1))
            
        except Exception as e:
            log(f"Cleanup attempt {attempt+1} failed: {str(e)}", level="WARNING")

def _release_windows_locks(store_path: Path) -> None:
    """Attempt to release Windows file locks"""
    if os.name != 'nt':
        return
        
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000
        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        OPEN_EXISTING = 3
        FILE_ATTRIBUTE_NORMAL = 0x80
        
        for f in store_path.glob('*'):
            if f.is_file():
                try:
                    handle = kernel32.CreateFileW(
                        str(f),
                        GENERIC_READ | GENERIC_WRITE,
                        FILE_SHARE_READ | FILE_SHARE_WRITE,
                        None,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        None
                    )
                    if handle != -1:
                        kernel32.CloseHandle(handle)
                except Exception:
                    pass
    except Exception as e:
        log(f"Windows lock release failed: {str(e)}", level="DEBUG")

def _verify_windows_permissions(store_path: Path) -> None:
    """Verify we have proper permissions on Windows"""
    if os.name != 'nt':
        return

    test_file = store_path / "permission_test.tmp"
    try:
        # Test write permission
        with open(test_file, 'w') as f:
            f.write("test")
        # Test delete permission
        test_file.unlink()
    except Exception as e:
        log(f"Permission verification failed: {str(e)}", level="ERROR")
        raise PermissionError(f"Insufficient permissions for {store_path}")

def _log_windows_solution_hint(store_path: Path) -> None:
    """Provide admin-specific troubleshooting hints"""
    if os.name == 'nt':
        log("Windows Permission Resolution Hints:", level="ERROR")
        log(f"1. Run PowerShell as Administrator", level="ERROR")
        log(f"2. Execute: takeown /F \"{store_path}\" /R /D Y", level="ERROR")
        log(f"3. Execute: icacls \"{store_path}\" /grant \"Users:(OI)(CI)F\" /T", level="ERROR")

def _delete_problematic_files(store_path: Path) -> None:
    """Targeted deletion of known problematic files"""
    problematic_patterns = ["*.lock", "*-wal", "*-shm", "*.parquet"]
    
    for pattern in problematic_patterns:
        for f in store_path.glob(pattern):
            try:
                f.unlink(missing_ok=True)
            except Exception as e:
                log(f"Failed to remove {f.name}: {str(e)}", level="DEBUG")

def _windows_force_cleanup(store_path: Path) -> None:
    """Windows-specific cleanup methods"""
    try:
        for f in store_path.glob("*"):
            try:
                if f.is_file():
                    os.unlink(f)
            except Exception as e:
                log(f"Windows force delete failed for {f.name}: {str(e)}", level="DEBUG")
        
        try:
            store_path.rmdir()
        except OSError:
            pass
            
    except Exception as e:
        log(f"Windows force cleanup failed: {str(e)}", level="ERROR")

def _initialize_chroma_store(
    store_path: Path,
    brand_key: Optional[str],
    embeddings: OllamaEmbeddings,
    fresh_start: bool = False
) -> Chroma:
    """Initialize Chroma store without recursion"""
    try:
        if not store_path.exists():
            raise FileNotFoundError(
                f"Persistence directory not found at {store_path}. "
                "You may need to create the vector store first."
            )
        log(f"Store Path: {store_path}")
        log(f"Collection Name: {Config.get_collection_name(brand_key)}")
        
        vector_store = Chroma(
            persist_directory=str(store_path),
            embedding_function=embeddings,
            collection_name=Config.get_collection_name(brand_key)
        )

        doc_count = vector_store._collection.count()
        if doc_count == 0:
            print("Warning: Vector store exists but contains no documents.")

        log(f"Successfully loaded vector store with {doc_count} documents")
        return vector_store
    except Exception as e:
        logger.error(f"Chroma initialization failed: {str(e)}")
        raise RuntimeError(f"Chroma initialization failed: {str(e)}")


def _validate_store(store: Chroma) -> bool:
    """Comprehensive store validation"""
    try:
        return (store is not None and 
                hasattr(store, '_collection') and 
                isinstance(store._collection.count(), int))
    except Exception as e:
        log(f"Store validation failed: {str(e)}", level="ERROR")
        return False