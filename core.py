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
import traceback  # For detailed error logging
import torch
import threading
import shutil
import psutil 
from typing import Optional

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
            cls._instance._start_keepalive()
        return cls._instance

    def _init_logger(self):
        """Initialize a dedicated logger for the RAG system"""
        self.logger = getLogger('rag_system')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler('rag_system.log')
        fh.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handlers
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        
        self.logger.info("RAG System logger initialized")

    def _reset_state(self):
        """Reset all components to initial state"""
        self.vector_db = None
        self.llm = None
        self.retriever = None
        self.chain = None
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
                temperature=0.3,  # Slightly more flexible
                num_gpu_layers=(
                    40 if (torch.cuda.is_available() and 
                        platform.system() == 'Linux' and 
                        torch.cuda.get_device_properties(0).total_memory >= 12*1024**3)  # 12GB+ check
                    else 10
                ),  # Let Ollama decide
                num_thread=16 if not torch.cuda.is_available() else 8,  # More threads for CPU
                top_k=20,  # More natural responses
                top_p=0.7,  # Less restrictive
                repeat_penalty=1.0,  # Good value
                stop=["\nObservation:", "\n\tObservation:"],  # More robust stopping
                keep_alive="10m",
                timeout=timeout * 0.9,
                num_ctx=5012,  # Larger context if model supports it
                streaming=True
            )

            # Test connection with simple ping
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
            # Clean up before loading
            # Config.windows_cleanup()
           
            # Load with retry
            store = load_vector_store(brand_key)
            self.logger.info(f"Checking...")
            
            # Additional validation
            if not store or not _validate_store(store):
                raise RuntimeError("Store validation failed after loading")
                
            return store
        except Exception as e:
            self.logger.error(f"Vector store load failed: {str(e)}")
            # Config.windows_cleanup()
            raise RuntimeError(f"Vector store initialization failed: {str(e)}")

    def _validate_vector_store(self) -> bool:
        """Validate the vector store is properly loaded"""
        if not self.vector_db:
            return False
        
        try:
            count = self.vector_db._collection.count()
            if count == 0:
                self.logger.debug("Warning: Vector store is empty")
            return True
        except Exception as e:
            self.logger.error(f"Vector store validation error: {str(e)}")
            return False

    def _cleanup_vector_stores(self):
        """Clean up all vector store directories"""
        try:
            if Config.PERSIST_DIRECTORY.exists():
                shutil.rmtree(Config.PERSIST_DIRECTORY)
                Config.PERSIST_DIRECTORY.mkdir(parents=True)
                self.logger.info("Vector stores cleaned up")
        except Exception as e:
            self.logger.error(f"Vector store cleanup failed: {str(e)}")

    def _load_with_timeout(self, func, timeout):
        """Helper for timeout operations with better error handling"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                result = future.result(timeout=timeout)
                if result is None:
                    raise ValueError("Operation returned None")
                return result
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(f"Operation timed out after {timeout} seconds")
            
    def _start_keepalive(self):
        """Start background thread to keep LLM alive with error handling"""
        def _keepalive_loop():
            while True:
                try:
                    if self.llm and self.__class__._is_ready:
                        self.llm.invoke("ping")
                        self.logger.info("Keep-alive ping sent")
                except Exception as e:
                    self.logger.debug(f"Keep-alive failed: {e}")
                time.sleep(240)  # Every 4 minutes

        threading.Thread(
            target=_keepalive_loop, 
            daemon=True,
            name="LLM-Keepalive"
        ).start()

    def warm_up(self, brand_key: Optional[str] = None, timeout: int = 300) -> bool:
        """Thread-safe warm-up with simplified validation to prevent recursion"""
        current_thread = threading.current_thread()
        brand_context = f"for brand {brand_key}" if brand_key else "for all brands"
        self.logger.info(f"Warm-up started {brand_context} in thread {current_thread.name}")
        
        try:
            # Phase 1: Vector Store Initialization
            vector_start = time.time()
            try:
                self.vector_db = self._safe_load_vector_store(brand_key)
                self.logger.info(f"[DOCS]: \n{self.vector_db}")
                if not self.vector_db or not hasattr(self.vector_db, '_collection'):
                    raise RuntimeError(f"Vector store initialization failed {brand_context}")
                    
                doc_count = self.vector_db._collection.count()
                brand_check = f"(brand: {brand_key})" if brand_key else "(all brands)"
                self.logger.info(f"Vector store loaded with {doc_count} docs {brand_check} in {time.time()-vector_start:.2f}s")
            except Exception as e:
                self.logger.error(f"Vector store init failed {brand_context}: {str(e)}")
                raise RuntimeError(f"Vector store initialization failed {brand_context}: {str(e)}")

            # Phase 2: LLM Initialization
            llm_start = time.time()
            try:
                self.llm = self._init_llm_with_retry(min(30, timeout//2))
                if not self.llm:
                    raise RuntimeError(f"LLM instance creation failed {brand_context}")
                
                # Simple ping test
                try:
                    response = str(self.llm.invoke("ping"))
                    self.logger.debug(f"LLM response {brand_context}: {response[:100]}...")
                except Exception as e:
                    raise RuntimeError(f"LLM ping failed {brand_context}: {str(e)}")
                    
                self.logger.info(f"LLM initialized {brand_context} in {time.time()-llm_start:.2f}s")
            except Exception as e:
                self.logger.error(f"LLM init failed {brand_context}: {str(e)}")
                raise RuntimeError(f"LLM initialization failed {brand_context}: {str(e)}")

            # Phase 3: Retriever Initialization
            retriever_start = time.time()
            try:
                # Explicitly pass brand_key to retriever creation
                self.retriever = create_retriever(self.vector_db, self.llm, brand_key=brand_key)
                if not self.retriever:
                    raise RuntimeError(f"Retriever creation failed {brand_context}")
                self.logger.info(f"Retriever created {brand_context} in {time.time()-retriever_start:.2f}s")
            except Exception as e:
                self.logger.error(f"Retriever creation failed {brand_context}: {str(e)}")
                raise

            # Phase 4: Chain Initialization
            chain_start = time.time()
            try:
                with self._initialization_lock:
                    self.chain = create_rag_chain(self.llm, brand_key)
                    if not self.chain:
                        raise RuntimeError(f"Chain creation failed {brand_context}")
                        
                self.logger.info(f"Chain created {brand_context} in {time.time()-chain_start:.2f}s")
            except Exception as e:
                self.logger.error(f"Chain creation failed {brand_context}: {str(e)}")
                raise RuntimeError(f"Chain initialization failed {brand_context}: {str(e)}")

            # Final readiness check
            self.__class__._is_ready = True
            self.create_cache_file()
            total_time = time.time() - vector_start
            self.logger.info(f"Warm-up completed successfully {brand_context} in {total_time:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"Warm-up failed {brand_context}: {str(e)}", exc_info=True)
            self._cleanup()
            raise RuntimeError(f"Warm-up failed {brand_context}: {str(e)}")
        
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
            self.vector_db,
            self.llm,
            self.retriever,
            self.chain
        ]
        
        return all(comp is not None for comp in required_components)

# Singleton instance
rag_system = RAGSystem()