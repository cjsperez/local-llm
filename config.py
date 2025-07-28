import os
import re
from pathlib import Path
from typing import Dict, List, TypedDict, Optional
import logging
import time

logger = logging.getLogger(__name__)

class CollectionConfig(TypedDict):
    """Configuration for a document collection"""
    json_file: str
    description: str
    chunk_size: int
    chunk_overlap: int

class Config:
    # Directory setup
    ENVIRONMENT = "development"
    BASE_DIR = Path(__file__).parent.resolve()
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    SYSTEM_DIR = DATA_DIR / "system"
    
    # Document paths
    BRAND_DOCUMENTS_FILE = DATA_DIR / "brand_documents.json"
    BRANDS_FILE = DATA_DIR / "brands.json"
    
    # System files
    DOCUMENT_LIST = SYSTEM_DIR / "document_list.json"
    THREADS_FILE = SYSTEM_DIR / "threads.json"
    
    # Vector store paths
    PERSIST_DIRECTORY = DATA_DIR / "vector_stores"  # Main directory for all vector stores
    EMBEDDING_CACHE_PATH = DATA_DIR / "embedding_cache"
    
    # Ollama settings
    LLM_MODEL = 'AI-AGENT:latest'
    EMBEDDING_MODEL = "all-minilm:latest"
    OLLAMA_BASE_URL = "http://localhost:4000"
    
    # Text processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    COLLECTION_NAME = "sysnet-ai-collection"
    ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.md', '.json'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # System behavior
    RECREATE_STORE = False
    VALID_PATH_PARTS = {'vector_store', 'vector_stores', 'brands', 'data'}  # Allowed path components
    INIT_TIMEOUT_VECTOR_STORE = 15
    INIT_TIMEOUT_LLM = 30
    INIT_TIMEOUT_TOTAL = 60

    @classmethod
    def init(cls):
        """Initialize all required directories and files with enhanced validation"""
        try:
            # Create directories with validation
            required_dirs = [
                cls.DATA_DIR,
                cls.DOCUMENTS_DIR,
                cls.SYSTEM_DIR,
                cls.PERSIST_DIRECTORY,
                cls.PERSIST_DIRECTORY / "brands",
                cls.EMBEDDING_CACHE_PATH
            ]
            
            for directory in required_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                cls._validate_directory_path(directory)
            
            # Initialize files with validation
            default_files = {
                cls.BRANDS_FILE: "{}",
                cls.BRAND_DOCUMENTS_FILE: "{}",
                cls.DOCUMENT_LIST: "[]",
                cls.THREADS_FILE: "{}"
                # Removed MAIN_DOCUMENTS_FILE
            }
            
            for file, default_content in default_files.items():
                if not file.exists():
                    file.write_text(default_content)
                    logger.info(f"Created default file: {file}")
                cls._validate_file_path(file)
                
        except Exception as e:
            logger.error(f"Configuration initialization failed: {e}")
            raise

    @classmethod
    def _validate_directory_path(cls, path: Path):
        """Validate directory path meets security requirements"""
        if not any(part in cls.VALID_PATH_PARTS for part in path.parts):
            raise ValueError(
                f"Invalid directory path '{path}'. Must contain one of: {cls.VALID_PATH_PARTS}"
            )

    @classmethod
    def _validate_file_path(cls, path: Path):
        """Validate file path meets security requirements"""
        if not any(part in cls.VALID_PATH_PARTS for part in path.parent.parts):
            raise ValueError(
                f"Invalid file location '{path}'. Parent must contain one of: {cls.VALID_PATH_PARTS}"
            )
        
    @staticmethod
    def get_brand_vector_store_path(brand_key: str) -> Path:
        return Config.PERSIST_DIRECTORY / "brands" / brand_key

    @classmethod
    def get_brand_dir(cls, brand_key: str) -> Path:
        """Get validated directory path for a brand's documents"""
        safe_key = cls.sanitize_name(brand_key)
        path = cls.DOCUMENTS_DIR / safe_key
        cls._validate_directory_path(path)
        return path

    @classmethod
    def get_brand_document_path(cls, brand_key: str) -> Path:
        """Get validated path to brand-specific document file"""
        path = cls.get_brand_dir(brand_key) / f"{cls.sanitize_name(brand_key)}-data.json"
        cls._validate_file_path(path)
        return path

    @classmethod
    def get_vector_store_path(cls, brand_key: Optional[str] = None) -> Path:
        """Get validated path for vector stores"""
        if brand_key:
            sanitized_brand = cls.sanitize_name(str(brand_key))  # Convert to string explicitly
            path = cls.PERSIST_DIRECTORY / "brands" / sanitized_brand
        else:
            path = cls.PERSIST_DIRECTORY / "default"
        
        # Create directory if it doesn't exist and validate
        path.mkdir(parents=True, exist_ok=True)
        cls._validate_directory_path(path)
        return path

    @staticmethod
    def sanitize_name(name: str) -> str:
        """Sanitize names for use in paths and collections"""
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name).strip('_')

    @classmethod
    def get_collection_name(cls, brand_key: Optional[str] = None) -> str:
        """Get validated collection name"""
        base_name = cls.sanitize_name(cls.COLLECTION_NAME)
        if brand_key:
            return f"{base_name}_{cls.sanitize_name(brand_key)}"
        return base_name
    
    @classmethod
    def windows_cleanup(cls):
        """Comprehensive resource cleanup for Windows"""
        if os.name == 'nt':
            try:
                import psutil
                # Clean up any Python processes holding files
                for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                    try:
                        if 'python' in proc.info['name'].lower():
                            if any(str(cls.PERSIST_DIRECTORY) in f.path 
                                for f in proc.info['open_files'] or []):
                                proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                # Additional cleanup if needed
                time.sleep(1)  # Allow time for releases
            except ImportError:
                logger.info(f"psutil not available for full cleanup", level="WARNING")

# Initialize configuration on import
try:
    Config.init()
except Exception as e:
    logger.critical(f"Failed to initialize configuration: {e}")
    raise