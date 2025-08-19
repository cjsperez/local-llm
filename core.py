from data_loader import DocumentLoader
from pathlib import Path
import platform
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from langchain_ollama import ChatOllama
from load_vector_store import load_vector_store, _validate_store
from retriever import create_retriever
from chain import create_rag_chain
from config import Config
from logger import log
import logging
from logging import getLogger
import traceback
import torch
import threading
import shutil
import psutil 
from typing import Optional, Dict, Any

class RAGSystem:
    _instance = None
    _is_ready = False
    _cache_file = Path("rag_system_initialized.flag")
    _initialization_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
            cls._instance._reset_state()
        return cls._instance

    def _init_logger(self):
        """Initialize a dedicated logger for the RAG system"""
        self.logger = getLogger('rag_system')
        self.logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        fh = logging.FileHandler('rag_system.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.info("RAG System logger initialized")

    def _reset_state(self):
        """Reset all components to initial state"""
        self.vector_dbs: Dict[str, Any] = {}  # Brand-keyed vector stores
        self.llm = None
        self.retrievers: Dict[str, Any] = {}  # Brand-keyed retrievers
        self.chains: Dict[str, Any] = {}  # Brand-keyed chains
        self._init_error = None
        self.__class__._is_ready = False

    def _init_llm(self, timeout: float = 120.0):
        """Safe LLM init in isolated thread to prevent hanging"""
        def _init():
            start = time.time()
            log(f"Initializing LLM ({Config.LLM_MODEL})...")

            llm = ChatOllama(
                model=Config.LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.3,
                num_gpu_layers=(
                    40 if (torch.cuda.is_available() and 
                        platform.system() == 'Linux' and 
                        torch.cuda.get_device_properties(0).total_memory >= 12*1024**3)
                    else 10
                ),
                num_thread=16 if not torch.cuda.is_available() else 8,
                top_k=20,
                top_p=0.7,
                repeat_penalty=1.0,
                stop=["\nObservation:", "\n\tObservation:"],
                keep_alive="10m",
                timeout=timeout * 0.9,
                num_ctx=5012,
                streaming=True
            )

            try:
                llm.invoke("ping")
                log("✓ LLM responsive")
            except Exception as e:
                log(f"⚠️ LLM ping failed: {e}", level="WARNING")

            log(f"LLM ready ({time.time()-start:.2f}s)")
            return llm

        try:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(_init)
                return future.result(timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"LLM init timeout or failure: {e}")

    def _safe_load_vector_store(self, brand_key: Optional[str] = None):
        """Safe store loading with proper resource management"""
        try:
            store = load_vector_store(brand_key)
            self.logger.info(f"Loaded vector store for brand: {brand_key}")
            
            if not store or not _validate_store(store):
                raise RuntimeError("Store validation failed after loading")
                
            return store
        except Exception as e:
            self.logger.error(f"Vector store load failed: {str(e)}")
            raise RuntimeError(f"Vector store initialization failed: {str(e)}")

    def get_retriever(self, brand_key: str):
        """Get or create a brand-specific retriever"""
        if brand_key not in self.retrievers:
            if brand_key not in self.vector_dbs:
                self.vector_dbs[brand_key] = self._safe_load_vector_store(brand_key)
            
            # Ensure we're passing correct parameter types to create_retriever
            self.retrievers[brand_key] = create_retriever(
                self.vector_dbs[brand_key], 
                self.llm, 
                brand_key=brand_key,
                # Add explicit parameters to avoid float/int confusion
                k=5,         # Example: make sure this is integer
                score_threshold=0.8  # Example: make sure this is float
            )
        return self.retrievers[brand_key]

    def get_chain(self, brand_key: str):
        """Get or create a brand-specific chain with proper initialization"""
        if brand_key not in self.chains:
            # Ensure we have all required components
            if not self.llm:
                raise RuntimeError("LLM not initialized")
            if brand_key not in self.retrievers:
                raise RuntimeError(f"No retriever available for brand {brand_key}")
            
            try:
                self.chains[brand_key] = create_rag_chain(
                    llm=self.llm,
                    retriever=self.retrievers[brand_key],
                    brand_key=brand_key
                )
                self.logger.info(f"Chain created for brand {brand_key}")
            except Exception as e:
                self.logger.error(f"Chain creation failed for {brand_key}: {str(e)}")
                raise ValueError(f"No chain available for brand {brand_key}")
        
        return self.chains[brand_key]

    def warm_up(self, brand_key: Optional[str] = None, timeout: int = 300) -> bool:
        """Thread-safe warm-up for specific brand or all brands"""
        current_thread = threading.current_thread()
        brand_context = f"for brand {brand_key}" if brand_key else "for all brands"
        self.logger.info(f"Warm-up started {brand_context} in thread {current_thread.name}")
        
        try:
            # Phase 1: LLM Initialization (shared across all brands)
            llm_start = time.time()
            try:
                self.llm = self._init_llm_with_retry(min(30, timeout//2))
                if not self.llm:
                    raise RuntimeError("LLM instance creation failed")
                
                try:
                    response = str(self.llm.invoke("ping"))
                    self.logger.debug(f"LLM response: {response[:100]}...")
                except Exception as e:
                    raise RuntimeError(f"LLM ping failed: {str(e)}")
                    
                self.logger.info(f"LLM initialized in {time.time()-llm_start:.2f}s")
            except Exception as e:
                self.logger.error(f"LLM init failed: {str(e)}")
                raise RuntimeError(f"LLM initialization failed: {str(e)}")

            # Phase 2: Brand-specific components
            try:
                if brand_key:
                    self._initialize_brand_components(brand_key, timeout)
                else:
                    # Initialize for all known brands
                    brands = DocumentLoader.load_brand_documents()
                    for brand in brands:
                        try:
                            self._initialize_brand_components(brand, timeout)
                        except Exception as e:
                            self.logger.error(f"Initialization failed for brand {brand}: {str(e)}")
                            continue  # Continue with other brands
            except Exception as e:
                self.logger.error(f"Brand components initialization failed: {str(e)}")
                raise

            # Final readiness check
            self.__class__._is_ready = True
            self.create_cache_file()
            self.logger.info(f"Warm-up completed successfully {brand_context}")
            return True

        except Exception as e:
            self.logger.error(f"Warm-up failed {brand_context}: {str(e)}", exc_info=True)
            self._cleanup()
            raise RuntimeError(f"Warm-up failed {brand_context}: {str(e)}")

    def _initialize_brand_components(self, brand_key: str, timeout: int):
        """Initialize components for a specific brand with enhanced error handling and retries"""
        start_time = time.time()
        max_retries = 2
        retry_delay = 1  # seconds
        
        def log_component(component: str, success: bool, duration: float, error: str = ""):
            status = "success" if success else "failed"
            message = f"{component} initialization {status} for {brand_key} in {duration:.2f}s"
            if error:
                message += f" | Error: {error}"
            if success:
                self.logger.info(message)
            else:
                self.logger.error(message)

        # Vector Store
        for attempt in range(max_retries + 1):
            try:
                vector_start = time.time()
                self.vector_dbs[brand_key] = self._safe_load_vector_store(brand_key)
                doc_count = self.vector_dbs[brand_key]._collection.count()
                log_component("Vector store", True, time.time() - vector_start)
                break
            except Exception as e:
                if attempt == max_retries:
                    log_component("Vector store", False, time.time() - start_time, str(e))
                    raise RuntimeError(f"Vector store init failed for {brand_key} after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))

        # Retriever
        for attempt in range(max_retries + 1):
            try:
                retriever_start = time.time()
                self.get_retriever(brand_key)
                log_component("Retriever", True, time.time() - retriever_start)
                break
            except Exception as e:
                if attempt == max_retries:
                    log_component("Retriever", False, time.time() - start_time, str(e))
                    raise RuntimeError(f"Retriever creation failed for {brand_key} after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))

        # Chain
        for attempt in range(max_retries + 1):
            try:
                chain_start = time.time()
                self.get_chain(brand_key)
                log_component("Chain", True, time.time() - chain_start)
                break
            except Exception as e:
                if attempt == max_retries:
                    log_component("Chain", False, time.time() - start_time, str(e))
                    
                    # Attempt fallback - create basic chain without brand customization
                    try:
                        self.logger.warning(f"Attempting fallback chain creation for {brand_key}")
                        basic_chain = create_rag_chain(
                            llm=self.llm,
                            retriever=self.retrievers[brand_key],
                            brand_key=None  # No brand-specific customization
                        )
                        self.chains[brand_key] = basic_chain
                        self.logger.warning(f"Fallback basic chain created for {brand_key}")
                        return
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"Chain creation failed for {brand_key} after {max_retries} attempts. "
                            f"Fallback also failed: {str(fallback_error)}. Original error: {str(e)}"
                        )
                time.sleep(retry_delay * (attempt + 1))

        self.logger.info(
            f"All components initialized for {brand_key} in {time.time()-start_time:.2f}s | "
            f"Documents: {self.vector_dbs[brand_key]._collection.count()}"
        )
        
    def _init_llm_with_retry(self, timeout, max_retries=2):
        """LLM initialization with retry logic"""
        for attempt in range(1, max_retries+1):
            try:
                return self._init_llm(timeout)
            except Exception as e:
                if attempt == max_retries:
                    raise
                self.logger.warning(f"LLM init attempt {attempt} failed, retrying...")
                time.sleep(1)  # Brief delay before retry

    def _cleanup(self):
        """Full system cleanup with state reset"""
        self._reset_state()
        try:
            if self._cache_file.exists():
                self._cache_file.unlink()
        except Exception as e:
            self.logger.debug(f"Cache file deletion error: {str(e)}")

    def create_cache_file(self):
        try:
            self._cache_file.write_text("initialized")
        except Exception as e:
            self.logger.debug(f"Cache file error: {str(e)}")

    def is_ready(self):
        """Simplified readiness check without recursive validation"""
        if not self.__class__._is_ready:
            return False
            
        required_components = [
            self.vector_dbs,
            self.llm,
            self.retrievers,
            self.chains
        ]
        
        return all(comp is not None for comp in required_components)

# Singleton instance
rag_system = RAGSystem()