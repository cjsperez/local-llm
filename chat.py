import json
import time
import inspect
import sys
import asyncio
import torch
import os
import uuid
import warnings
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Union, Optional, List, AsyncGenerator, Any
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
import psutil  # For resource monitoring
import threading
import platform
import re

from langchain_ollama import ChatOllama
from core import rag_system
from config import Config
from data_loader import DocumentLoader
from vector_store import VectorStoreManager
from langchain_core.documents.base import Document
from async_lru import alru_cache

from utils.brand_utils import (
    get_brand_config,
)

load_and_prepare_documents = DocumentLoader.load_and_prepare_documents

start_total = time.time()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(name)s - %(levelname)s]: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Message(BaseModel):
    id: str
    content: str
    role: str  # "user" or "assistant"
    timestamp: str
    brand: Optional[str] = None


class ConversationThread:
    def __init__(self, user_id: Optional[str] = None):
        self.thread_id = str(uuid.uuid4())
        self.user_id = user_id
        self.messages: List[Message] = []
        self.context_history: List[Dict] = []  # Store previous context documents
        self.created_at = time.strftime('%Y-%m-%d %H:%M:%S')
        self.lock = asyncio.Lock()
        self.context_memory_window = 3  # Only keep last 3 contexts
        self.max_history_length = 6  # Last 3 exchanges
        self.state = {
            "has_greeted": False,
            "last_questions": []
        }
        
    async def add_message(self, role: str, content: str, context: Optional[List] = None , brand_key: Optional[str] = None):
        async with self.lock:
            # Prune old messages if needed
            if len(self.messages) > self.max_history_length:
                self.messages = self.messages[-self.max_history_length:]
            
            if context and len(self.context_history) > self.context_memory_window:
                self.context_history = self.context_history[-self.context_memory_window:]
            message = Message(
                id=str(uuid.uuid4()),
                content=content,
                role=role,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                brand=brand_key
            )
            self.messages.append(message)

             # Auto-update summary for important user facts
            if role == "user":
                self._update_summary(content)
                self.state["last_questions"].append(content)
                if len(self.state["last_questions"]) > 3:
                    self.state["last_questions"].pop(0)

            if context:
                self.context_history.append({
                    "timestamp": message.timestamp,
                    "context": context
                })
            return message
    def _update_summary(self, user_message: str):
        """Extract and store key facts more robustly"""
        # Name detection
        name_patterns = [
            "my name is",
            "I'm called",
            "you can call me"
        ]
        
        for pattern in name_patterns:
            if pattern in user_message.lower():
                name = user_message.split(pattern)[-1].split(".")[0].strip()
                if 0 < len(name) < 50:
                    # Store as a permanent fact in the thread
                    if not hasattr(self, "user_facts"):
                        self.user_facts = {}
                    self.user_facts["name"] = name
                    break
    
    def get_recent_context(self, max_contexts=3) -> List:
        """Get recent context documents for continuity"""
        return [ctx["context"] for ctx in self.context_history[-max_contexts:]]


class QueryProcessor:
    def __init__(self):
        self._init_lock = asyncio.Lock()
        self._is_warming = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.active_threads: Dict[str, ConversationThread] = {}
        self._threads_lock = asyncio.Lock()
        self._user_thread_map: Dict[str, str] = {}  # Maps user_id to thread_id
        self._load_threads()
        self.state = {
            "has_greeted": False,
            "last_questions": []
        }
        

    def _load_threads(self):
        """Load persisted threads from storage"""
        try:
            if Config.THREADS_FILE.exists():
                with open(Config.THREADS_FILE, "r") as f:  # Fixed: Use Config.THREADS_FILE
                    threads_data = json.load(f)
                    for thread_id, data in threads_data.items():
                        thread = ConversationThread(user_id=data["user_id"])
                        thread.messages = [Message(**msg) for msg in data["messages"]]
                        self.active_threads[thread_id] = thread
        except Exception as e:
            logger.error(f"Error loading threads: {e}")

    def _save_threads(self):
        """Persist threads to storage"""
        try:
            with open(Config.THREADS_FILE, "w") as f:  # Fixed: Use Config.THREADS_FILE
                json.dump({
                    thread_id: {
                        "user_id": thread.user_id,
                        "messages": [msg.dict() for msg in thread.messages]
                    }
                    for thread_id, thread in self.active_threads.items()
                }, f, indent=2)  # Added indent for better readability
        except Exception as e:
            logger.error(f"Error saving threads: {e}")

    async def _create_new_processor(self):
        """Create a fresh processor instance for each request"""
        return QueryProcessor()
        
    async def get_or_create_thread(self, user_id: Optional[str] = None) -> str:
        """Get or create a thread for the user"""
        async with self._threads_lock:
            if user_id and user_id in self._user_thread_map:
                return self._user_thread_map[user_id]
            
            thread = ConversationThread(user_id=user_id)
            self.active_threads[thread.thread_id] = thread
            if user_id:
                self._user_thread_map[user_id] = thread.thread_id
            return thread.thread_id
        
    async def get_previous_questions(self, thread_id: str) -> List[str]:
        """Get previous questions from the thread"""
        thread = await self.get_thread(thread_id)
        if not thread:
            return []
        return [msg.content for msg in thread.messages if msg.role == "user"]
    
    async def get_thread(self, thread_id: str) -> ConversationThread:
        """Get thread with proper validation"""
        async with self._threads_lock:
            # logger.info(f"Active threads: {list(self.active_threads.keys())}")
            logger.info(f"User thread map: {self._user_thread_map}")
            if not thread_id:
                raise ValueError("Thread ID cannot be empty")
                
            # Check memory first
            thread = self.active_threads.get(thread_id)
            if thread:
                return thread
                
            # Fallback to disk if not in memory
            try:
                if Config.THREADS_FILE.exists():
                    with open(Config.THREADS_FILE, "r") as f:
                        threads_data = json.load(f)
                        if thread_id in threads_data:
                            data = threads_data[thread_id]
                            thread = ConversationThread(user_id=data["user_id"])
                            thread.messages = [Message(**msg) for msg in data["messages"]]
                            self.active_threads[thread_id] = thread
                            return thread
            except Exception as e:
                logger.error(f"Error reloading thread from disk: {e}")
                
            raise ValueError(f"Thread {thread_id} not found")
    
    async def get_threads_for_user(self, user_id: str) -> List[str]:
        """Optional: Get all thread IDs for a user"""
        async with self._threads_lock:
            return [
                t.thread_id for t in self.active_threads.values() 
                if t.user_id == user_id
            ]

    def _format_input(self, inputs: Dict) -> Dict:
        """Format inputs for the RAG chain with proper thread handling"""
        input_data = {
            "question": inputs["question"],
            "context": inputs["context"],
            "conversation_history": "No previous conversation",
            "user_name": "",
            "is_first_message": True,  # Default to True, will be updated if thread exists
        }
        # logger.info(f"[MAIN INPUT]: {inputs}\n")
        # Only process thread if it exists in inputs
        if "thread" in inputs and inputs["thread"]:
            thread = inputs["thread"]
            
            # Build conversation history
            history_lines = []
            for msg in thread.messages[-6:]:  # Last 2 exchanges
                # logger.info(f"{msg}")
                history_lines.append({
                    "role": msg.role,
                    "text": msg.content,
                    "timestamp": msg.timestamp,
                    "brand":inputs["brand"]
                })
            
            # Update input data with thread information
            input_data.update({
                "conversation_history": json.dumps(history_lines),
                "is_first_message": len(thread.messages) <= 1,  # True only for first message
                "user_name": thread.user_facts.get("name", "") if hasattr(thread, "user_facts") else ""
            })
            # logger.info(f"[Context]: {inputs['context']}")
            logger.info(f"[Query]: {input_data['question']}")
            logger.info(f"[Conversation history]: {input_data['conversation_history']}")
        return input_data

    async def _ensure_initialized(self, brand_key: Optional[str] = None, max_retries: int = 3) -> bool:
        async with self._init_lock:
            if rag_system.is_ready():
                return True
                
            if self._is_warming:
                try:
                    await asyncio.wait_for(
                        self._wait_for_warmup(), 
                        timeout=60
                    )
                    if not rag_system.is_ready():
                        raise RuntimeError("System not ready after warmup wait")
                    return True
                except asyncio.TimeoutError:
                    logger.error("Deadlock detected - restarting initialization")
                    self._is_warming = False
                    rag_system._cleanup()
                    if max_retries <= 0:
                        raise RuntimeError("Max retries exceeded")
                    return await self._ensure_initialized(brand_key, max_retries-1)

            self._is_warming = True
            try:
                logger.info("Starting RAG system initialization...")
                
                # Initialize vector stores with retry
                try:
                    await asyncio.wait_for(
                        self._initialize_vector_stores(),
                        timeout=30  # Increased timeout
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError("Vector store init timed out")

                # Warm up RAG system
                try:
                    loop = asyncio.get_running_loop()
                    warm_up_task = loop.run_in_executor(
                        None,
                        lambda: rag_system.warm_up(brand_key, timeout=90)  # Increased timeout
                    )
                    await asyncio.wait_for(warm_up_task, timeout=120)  # Increased timeout
                except asyncio.TimeoutError:
                    logger.error("Warm-up timed out")
                    raise RuntimeError("System initialization timed out")
                except Exception as e:
                    logger.error(f"Warm-up failed: {str(e)}")
                    raise

                if not rag_system.is_ready():
                    logger.info("Loading...")
                    raise RuntimeError("System not ready after warm-up")

                return True
                
            except Exception as e:
                logger.error(f"Initialization failed: {str(e)}")
                self._is_warming = False
                rag_system._cleanup()
                raise
            finally:
                self._is_warming = False

    async def _initialize_vector_stores(self) -> bool:
        """Initialize vector stores with retry logic"""
        for attempt in range(3):
            try:
                logger.info(f"Initializing vector stores (attempt {attempt+1}/3)")
                if Config.RECREATE_STORE:
                    VectorStoreManager.delete_vector_store()
                    
                # Load brand stores
                brands = DocumentLoader.get_all_brands()
                for brand in brands:
                    logger.info(f"Brand: {brand}")
                    VectorStoreManager.get_vector_store(
                        brand_key=brand,
                        create_if_missing=True
                    )
                    
                return True
            except Exception as e:
                logger.error(f"Vector store init attempt {attempt+1} failed: {str(e)}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return False

    async def _wait_for_warmup(self):
        """Wait for ongoing warm-up to complete"""
        while self._is_warming:
            await asyncio.sleep(0.1)
            logger.debug("Waiting for warm-up to complete...")

    @alru_cache(maxsize=500)
    async def process_query(
    self, 
    question: str, 
    thread_id: Optional[str] = None, 
    user_id: Optional[str] = None, 
    brand_key: Optional[str] = None,
    reset: bool = False
) -> Dict[str, Union[str, float]]:
        """Process a query with brand-specific context"""
        if reset:
            thread = await self.get_thread(thread_id)
            if thread:
                thread.context_history = []
                thread.messages = [m for m in thread.messages if m.role == "user"]

        response = {
            "status": "success",
            "result": "",
            "processing_time": 0.0,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "thread_id": thread_id,
            "brand": brand_key  # Track brand in response
        }
        start_time = time.perf_counter()

        try:
            # Get or create thread FIRST
            if not thread_id:
                thread_id = await self.get_or_create_thread(user_id=user_id)
            thread = await self.get_thread(thread_id)
            if not thread:
                raise RuntimeError("Thread not found")

            # Store brand in thread if provided
            if brand_key and hasattr(thread, 'brand'):
                thread.brand = brand_key

            # Check for previous questions
            previous_questions = await self.get_previous_questions(thread_id)
            
            # Add user message to thread
            await thread.add_message("user", question, brand_key)
            self._save_threads()
            await self._ensure_initialized(brand_key)
            # Ensure system is ready with brand-specific setup
            # await self._ensure_initialized(brand_key=brand_key)
            
            logger.info(f"Processing query for brand {brand_key} in thread {thread_id}: '{question[:50]}...'")
            logger.debug(f"Previous questions in thread: {previous_questions}")

            # Step 1: Retrieve context with brand filtering
            context_docs = await self._retrieve_context(question, brand_key=brand_key)
            retrieval_time = time.perf_counter() - start_time

            logger.info(f"Retrieved {len(context_docs)} relevant documents in {retrieval_time:.2f}s")

            # Step 2: Generate brand-specific response
            chain_input = self._format_input(
                {
                    "question": question,
                    "context": context_docs,
                    "thread": thread,
                    "brand": brand_key  # Pass brand to formatter
                }
            )
            
            result_chunks = []
            async for chunk in self._generate_response(chain_input, stream=False, brand_key=brand_key):
                result_chunks.append(chunk)
            result = "".join(result_chunks).strip()
            generation_time = time.perf_counter() - start_time - retrieval_time
            
            response.update({
                "result": result,
                "processing_time": time.perf_counter() - start_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "documents_retrieved": len(context_docs),
                "model": Config.LLM_MODEL,
                "thread_id": thread_id,
                "brand": brand_key  # Ensure brand is included
            })

            # Apply brand-specific post-processing
            if response["result"] and brand_key:
                brand_config = get_brand_config(brand_key)
                if brand_config:
                    # Remove brand-specific welcome phrases
                    welcome_phrases = [
                        f"Welcome to {brand_config.get('display_name', '')} support",
                        brand_config.get('off_topic_response', '')
                    ]
                    for phrase in welcome_phrases:
                        if phrase and response["result"].startswith(phrase):
                            response["result"] = response["result"][len(phrase):].lstrip()
                    
                    # Apply brand-specific corrections
                    corrections = brand_config.get('corrections', {})
                    for wrong, correct in corrections.items():
                        response["result"] = response["result"].replace(wrong, correct)
            
            # Add assistant message to thread
            await thread.add_message("assistant", response["result"], brand_key)
            self._save_threads()
            
            # Log the complete updated conversation history
            history_log = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "brand": msg.brand
                } 
                for msg in thread.messages
            ]
            logger.info(f"Updated conversation history: {json.dumps(history_log, indent=2)}")
            
            logger.info(
                f"[TIMING] Brand: {brand_key} | "
                f"Total: {response['processing_time']:.2f}s | "
                f"Retrieval: {retrieval_time:.2f}s | "
                f"Generation: {generation_time:.2f}s | "
                f"Docs: {len(context_docs)} | "
                f"Thread: {thread_id}"
            )
            
            return response
            
        except Exception as e:
            error_type = type(e).__name__
            return {
                "status": "error",
                "result": None,
                "error": f"{error_type}: {str(e)}",
                "processing_time": time.perf_counter() - start_time,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "thread_id": thread_id,
                "brand": brand_key,
                "diagnostics": {
                    "ollama_status": "Check with: curl http://localhost:11434/api/tags",
                    "vector_store": str(Config.PERSIST_DIRECTORY),
                    "embedding_model": Config.EMBEDDING_MODEL,
                    "llm_model": Config.LLM_MODEL,
                    "brand_config": f"Check with: GET /brands/{brand_key}" if brand_key else "No brand specified"
                }
            }
        
    async def stream_query(
        self, 
        question: str, 
        thread_id: Optional[str] = None, 
        user_id: Optional[str] = None,
        brand_key: Optional[str] = None,
        reset: bool = False
    ):
        """Stream query response chunks with brand-specific context and full feature parity with process_query"""
        last_yield_time = time.time()
        start_time = time.perf_counter()
        response_metadata = {
            "status": "in_progress",
            "processing_time": 0.0,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "thread_id": thread_id,
            "brand": brand_key,
            "documents_retrieved": 0,
            "model": Config.LLM_MODEL
        }

        try:
            # Handle thread reset if requested
            if reset:
                thread = await self.get_thread(thread_id)
                if thread:
                    thread.context_history = []
                    thread.messages = [m for m in thread.messages if m.role == "user"]
            
            # Get or create thread FIRST
            if not thread_id:
                thread_id = await self.get_or_create_thread(user_id=user_id)
                response_metadata["thread_id"] = thread_id
            logger.info(f"[THREAD ID - Chat]: {thread_id}")
            thread = await self.get_thread(thread_id)
            if not thread:
                raise RuntimeError("Thread not found")

            # Store brand in thread if provided
            if brand_key and hasattr(thread, 'brand'):
                thread.brand = brand_key

            # Check for previous questions
            previous_questions = await self.get_previous_questions(thread_id)
            
            # Add user message to thread
            await thread.add_message("user", question, brand_key)
            self._save_threads()
            
            # Ensure system is ready with brand-specific setup
            await self._ensure_initialized(brand_key)
            
            logger.info(f"Streaming query for brand {brand_key} in thread {thread_id}: '{question[:50]}...'")
            logger.debug(f"Previous questions in thread: {previous_questions}")

            # Step 1: Retrieve context with brand filtering
            context_docs = await self._retrieve_context(question, brand_key=brand_key)
            retrieval_time = time.perf_counter() - start_time
            response_metadata.update({
                "documents_retrieved": len(context_docs),
                "retrieval_time": retrieval_time
            })

            logger.info(f"Retrieved {len(context_docs)} relevant documents in {retrieval_time:.2f}s")

            # Step 2: Generate brand-specific response
            chain_input = self._format_input({
                "question": question,
                "context": context_docs,
                "thread": thread,
                "brand": brand_key  # Pass brand to formatter
            })

            # Collect all response chunks for post-processing
            full_response_chunks = []
            
            # Stream response chunks
            async for chunk in self._generate_response(chain_input, stream=True, brand_key=brand_key):
                if hasattr(chunk, 'content'):
                    chunk_content = chunk.content
                else:
                    chunk_content = str(chunk)
                
                
                
                # Send each chunk to client with metadata
                chunk_data = {
                    "text": chunk_content,
                    "metadata": response_metadata
                }
                yield f"data: {json.dumps(chunk_data)}"
                full_response_chunks.append(chunk_content)

            # Combine all chunks into full response
            assistant_response = "".join(full_response_chunks).strip()
            generation_time = time.perf_counter() - start_time - retrieval_time
            
            # Update final metadata
            response_metadata.update({
                "status": "success",
                "processing_time": time.perf_counter() - start_time,
                "generation_time": generation_time
            })

            # Apply brand-specific post-processing
            if assistant_response and brand_key:
                brand_config = get_brand_config(brand_key)
                if brand_config:
                    # Remove brand-specific welcome phrases
                    welcome_phrases = [
                        f"Welcome to {brand_config.get('display_name', '')} support",
                        brand_config.get('off_topic_response', '')
                    ]
                    for phrase in welcome_phrases:
                        if phrase and assistant_response.startswith(phrase):
                            assistant_response = assistant_response[len(phrase):].lstrip()
                    
                    # Apply brand-specific corrections
                    corrections = brand_config.get('corrections', {})
                    for wrong, correct in corrections.items():
                        assistant_response = assistant_response.replace(wrong, correct)

            # Add assistant message to thread
            await thread.add_message("assistant", assistant_response, brand_key)
            self._save_threads()
            # Log the complete updated conversation history
            history_log = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "brand": msg.brand
                } 
                for msg in thread.messages
            ]
            logger.info(f"Updated conversation history: {json.dumps(history_log, indent=2)}")
            
            logger.info(
                f"[TIMING] Brand: {brand_key} | "
                f"Total: {response_metadata['processing_time']:.2f}s | "
                f"Retrieval: {retrieval_time:.2f}s | "
                f"Generation: {generation_time:.2f}s | "
                f"Docs: {len(context_docs)} | "
                f"Thread: {thread_id}"
            )
            
            # Final completion message with full metadata
            yield f"data: {json.dumps({'status': 'complete', 'metadata': response_metadata})}"

        except Exception as e:
            error_type = type(e).__name__
            error_metadata = {
                "status": "error",
                "error": f"{error_type}: {str(e)}",
                "processing_time": time.perf_counter() - start_time,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "thread_id": thread_id,
                "brand": brand_key,
                "diagnostics": {
                    "ollama_status": "Check with: curl http://localhost:11434/api/tags",
                    "vector_store": str(Config.PERSIST_DIRECTORY),
                    "embedding_model": Config.EMBEDDING_MODEL,
                    "llm_model": Config.LLM_MODEL,
                    "brand_config": f"Check with: GET /brands/{brand_key}" if brand_key else "No brand specified"
                }
            }
            yield f"data: {json.dumps(error_metadata)}"

    async def _retrieve_context(self, question: str, brand_key: Optional[str] = None) -> List[Dict]:
        """Retrieve and rank relevant documents based on the question.
        
        Args:
            question: The query string to search for
            brand_key: Optional brand identifier to filter collections
            
        Returns:
            List of formatted documents sorted by relevance
        
        Raises:
            RuntimeError: If retriever is not initialized
        """
        start_time = time.time()
        logger.info(f"Starting context retrieval for question: '{question}'")
        
        try:
            # Validate retriever initialization
            if not hasattr(rag_system, 'retriever') or not rag_system.retriever:
                logger.error("Retriever not initialized in RAG system")
                raise RuntimeError("Retriever not initialized")
            
            # Prepare retrieval config
            retrieval_config = {}
            if brand_key:
                collection_name = Config.get_collection_name(brand_key)
                logger.info(f"Filtering for brand: {brand_key}, collection: {collection_name}")
                retrieval_config = {"configurable": {"collection_name": collection_name}}
            
            # Execute retrieval (async if available)
            retriever = rag_system.retriever
            invoke_method = retriever.ainvoke if hasattr(retriever, 'ainvoke') else retriever.invoke
            if inspect.iscoroutinefunction(invoke_method):
                result = await invoke_method(question, config=retrieval_config)
            else:
                result = invoke_method(question, config=retrieval_config)
            
            if not result:
                logger.warning("Retriever returned empty results")
                return []
            
            # Process and score documents
            query_lower = question.lower()
            query_words = set(word for word in query_lower.split() if len(word) > 2)  # Ignore short words
            
            scored_docs = []
            for doc in result:
                # Extract content and metadata
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = getattr(doc, 'metadata', {})
                elif isinstance(doc, dict):
                    content = doc.get('page_content', str(doc))
                    metadata = doc.get('metadata', {})
                else:
                    content = str(doc)
                    metadata = {}
                
                # Calculate relevance score
                question = metadata.get('question', '').lower()
                
                # Score based on question and content matches
                question_score = sum(1 for word in query_words if word in question) * 2
                
                # Bonus for exact phrase match
                if query_lower in question:
                    question_score += 5

                    
                total_score = question_score 
                
                scored_docs.append({
                    'content': content,
                    'metadata': metadata,
                    'score': total_score,
                    'question_score': question_score,
                })
            
            # Sort by score (descending) and limit to top 5
            scored_docs.sort(key=lambda x: -x['score'])
            top_docs = scored_docs[:5]
            # logger.info(
            #     f"[TOP 5 RESULT]: {top_docs}"
            # )
            # Log performance metrics
            retrieval_time = time.time() - start_time
            logger.info(
                f"Retrieved {len(top_docs)} documents in {retrieval_time:.2f}s | "
                f"Top score: {top_docs[0]['score']} (question: {top_docs[0]['question_score']}, "
                f"Top question: '{top_docs[0]['metadata'].get('question', 'N/A')}'"
            )
            
            # Format final output
            return [{
                'content': doc['content'],
                'metadata': doc['metadata'],
                'relevance_score': doc['score'],
                'question_score': doc['question_score'],
            } for doc in top_docs]
            
        except Exception as e:
            logger.error(f"Error during context retrieval: {str(e)}", exc_info=True)
            raise

        except Exception as e:
            logger.error(
                f"Context retrieval failed after {time.time() - start_time:.2f}s\n"
                f"Error: {str(e)}\n"
                f"Brand: {brand_key}\n"
                f"Question: {question[:100]}...",
                exc_info=True
            )
            raise RuntimeError(f"Context retrieval failed: {str(e)}")


    async def _generate_response(
        self, 
        chain_input: Dict, 
        stream: bool = False, 
        brand_key: Optional[str] = None,
        max_retries: int = 2,
        timeout: int = 120
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """
        Generate response with enhanced error handling, brand-specific processing, and reliability features
        
        Args:
            chain_input: Dictionary containing input parameters for the generation
            stream: Whether to stream the response or return it complete
            brand_key: Optional brand identifier for brand-specific processing
            max_retries: Maximum number of retry attempts for generation
            timeout: Timeout in seconds for the generation operation
            
        Yields:
            Response chunks (when streaming) or complete response (when not streaming)
        """
        start_time = time.perf_counter()
        fallback_response = self._get_brand_fallback(brand_key)
        attempt = 0
        
        try:
            # Validate chain input
            if not chain_input or "question" not in chain_input:
                raise ValueError("Invalid chain input - missing required 'question' field")
                
            if not chain_input.get("context"):
                logger.warning("No context provided for generation - response may be less accurate")

            # Get brand-specific configuration
            brand_config = get_brand_config(brand_key) if brand_key else None

            while attempt <= max_retries:
                logger.info(f"[CHAIN INPUT]: {chain_input}")
                if "context" in chain_input:
                    # Ensure scores are properly transferred from doc level to metadata
                    fixed_context = []
                    for doc in chain_input["context"]:
                        if isinstance(doc, dict):
                            # Create a copy to avoid modifying original
                            fixed_doc = doc.copy()
                            metadata = fixed_doc.get('metadata', {}).copy()
                            
                            # Transfer scores from doc level to metadata if they exist
                            if 'relevance_score' in fixed_doc:
                                metadata['relevance_score'] = float(fixed_doc['relevance_score'])
                            if 'question_score' in fixed_doc:
                                metadata['question_score'] = float(fixed_doc['question_score'])
                            
                            fixed_doc['metadata'] = metadata
                            fixed_context.append(fixed_doc)
                        else:
                            fixed_context.append(doc)
                    
                    chain_input["context"] = fixed_context
                
                # logger.info(f"Fixed context metadata: {[doc.get('metadata', {}) for doc in chain_input.get('context', [])]}")
                try:
                    if stream:
                        # Streaming response with timeout and retry support
                        async for chunk in self._stream_with_timeout(
                            self._stream_response(chain_input, brand_key),
                            timeout=timeout
                        ):
                            logger.info(f"=> text: {chunk}" )
                            yield chunk
                        break  # Success - exit retry loop
                    else:
                        logger.info(f"\n\nhere?\n\n")
                        # Non-streaming response with timeout
                        # try:
                        #     result = await asyncio.wait_for(
                        #         self._generate_complete_response(chain_input, brand_config, llm_params),
                        #         timeout=timeout
                        #     )
                            
                        #     # Validate and process response
                        #     processed = self._process_response(result, brand_key)
                            
                        #     # Apply brand-specific formatting if needed
                        #     if brand_config and brand_config.get("response_format"):
                        #         processed = self._apply_response_format(processed, brand_config)
                            
                        #     yield processed
                        #     break  # Success - exit retry loop
                            
                        # except asyncio.TimeoutError:
                        #     logger.warning(f"Generation timeout after {timeout}s (attempt {attempt + 1})")
                        #     attempt += 1
                        #     if attempt > max_retries:
                        #         raise RuntimeError("Max retries exceeded due to timeout")
                        #     continue
                            
                except Exception as e:
                    logger.error(f"Generation attempt {attempt + 1} failed: {str(e)}")
                    attempt += 1
                    if attempt > max_retries:
                        raise  # Re-raise if we've exhausted retries
                    await asyncio.sleep(1 * attempt)  # Exponential backoff
                    
        except Exception as e:
            logger.error(f"Final generation failure after {attempt} attempts: {str(e)}")
            error_info = {
                "error": str(e),
                "attempts": attempt,
                "brand": brand_key,
                "fallback_used": True
            }
            
            # Enhanced fallback with error context
            enhanced_fallback = f"{fallback_response}\n\n[Error: {str(e)}]" if self._include_errors_in_response else fallback_response
            yield enhanced_fallback
            
        finally:
            processing_time = time.perf_counter() - start_time
            logger.info(f"Generation completed in {processing_time:.2f} seconds after {attempt} attempts")
            
            # Log performance metrics
            self._log_generation_metrics({
                "success": attempt <= max_retries,
                "processing_time": processing_time,
                "attempts": attempt,
                "brand": brand_key,
                "streaming": stream,
                "input_length": len(chain_input.get("question", "")),
                "context_items": len(chain_input.get("context", []))
            })

    async def _stream_with_timeout(
        self,
        generator: AsyncGenerator,
        timeout: int
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """
        Wrapper for streaming generators that adds timeout per chunk and enforces total timeout
        """
        start_time = time.perf_counter()
        try:
            async for chunk in generator:
                elapsed = time.perf_counter() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise asyncio.TimeoutError("Stream generation exceeded total timeout")
                # Enforce timeout for the current chunk
                chunk = await asyncio.wait_for(asyncio.sleep(0, result=chunk), timeout=remaining)
                yield chunk
        except asyncio.TimeoutError:
            raise

    def _log_generation_metrics(self, metrics: Dict) -> None:
        """Log generation performance metrics for monitoring"""
        logger.info(
            f"Generation metrics: "
            f"success={metrics['success']} | "
            f"time={metrics['processing_time']:.2f}s | "
            f"attempts={metrics['attempts']} | "
            f"brand={metrics['brand']} | "
            f"streaming={metrics['streaming']} | "
            f"input_len={metrics['input_length']} | "
            f"context_items={metrics['context_items']}"
        )

    async def _stream_response(self, chain_input: Dict, brand_key: Optional[str] = None):
        buffer = ""
        try:
            async for chunk in rag_system.chain.astream(chain_input):
                try:
                    content = str(chunk)
                    # buffer += content
                    content = str(chunk)
                    if content.strip():
                        yield content
                            
                except Exception as chunk_error:
                    logger.warning(f"Chunk processing error: {chunk_error}")
                    yield f"[ERROR: {str(chunk_error)}]"
                    continue
            
            if buffer:
                yield buffer
                logger.info(f"Completed streaming. Buffer length: {len(buffer)}")

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {{\"error\": \"Streaming failed\", \"message\": \"{str(e)}\"}}\n\n"
            yield "data: {\"status\": \"error\"}\n\n"
            raise RuntimeError("Response generation aborted") from e
        
    async def _generate_complete_response(
        self,
        chain_input: Dict,
        brand_config: Optional[Dict] = None,
        llm_params: Optional[Dict] = None
    ) -> str:
        """
        Generate complete non-streaming response with enhanced processing
        
        Args:
            chain_input: Input dictionary for the generation
            brand_config: Brand-specific configuration
            llm_params: Parameters for the LLM (temperature, max_tokens, etc.)
            
        Returns:
            Fully processed response string
            
        Raises:
            ValueError: If response validation fails
            RuntimeError: For generation failures
        """
        llm_params = llm_params or {}
        response_formatter = brand_config.get("response_formatter") if brand_config else None
        
        try:
            # Generate response with configured parameters
            result = await rag_system.chain.ainvoke(
                chain_input,
                config={"configurable": {"llm_params": llm_params}}
            )
            logger.debug(f"Raw LLM response: {result[:500]}...")  # Log first 500 chars
            
            # Extract content from different response formats
            content = self._extract_response_content(result)
            
            # Apply brand-specific formatting if available
            if response_formatter:
                content = response_formatter(content)
                
            # Validate response quality
            if not self._validate_response(content):
                logger.error("Response validation failed")
                raise ValueError("Generated content failed quality checks")
                
            return content
            
        except Exception as e:
            logger.error(f"Non-streaming generation failed: {str(e)}", exc_info=True)
            fallback = self._get_brand_fallback(brand_config.get("key") if brand_config else None)
            return fallback

    def _extract_response_content(self, result: Union[str, dict, Any]) -> str:
        """
        Extract content from various LLM response formats
        
        Args:
            result: Raw response from LLM
            
        Returns:
            Extracted text content
        """
        if isinstance(result, str):
            return result
        if hasattr(result, 'content'):
            return result.content
        if isinstance(result, dict):
            if 'result' in result:
                return result['result']
            if 'text' in result:
                return result['text']
            if 'output' in result:
                return result['output']
        return str(result)

    def _validate_response(self, content: str) -> bool:
        """
        Validate response meets quality standards
        
        Args:
            content: Generated content to validate
            
        Returns:
            bool: True if content is valid
        """
        if not content.strip():
            return False
            
        # Check for common error patterns
        error_patterns = [
            "I don't know",
            "I can't answer",
            "I'm sorry",
            "I apologize",
            "as an AI language model"
        ]
        
        if any(pattern.lower() in content.lower() for pattern in error_patterns):
            return False
            
        # Minimum length check
        if len(content) < 10:  # At least 10 characters
            return False
            
        return True

    def _get_brand_fallback(self, brand_key: Optional[str] = None) -> str:
        """
        Get brand-specific fallback response
        
        Args:
            brand_key: Optional brand identifier
            
        Returns:
            Appropriate fallback message
        """
        default = "We're experiencing high demand. Please try again shortly."
        if not brand_key:
            return default
            
        brand_config = get_brand_config(brand_key)
        if not brand_config:
            return default
            
        return brand_config.get("fallback_response", default)

    def _process_response(self, content: str, validator: Optional[callable], brand_key: str):
        """Validate and process the final response"""
        # Default validation
        if not content or "I'm having trouble" in content:
            raise ValueError("Empty or error response generated")
        
        # Brand-specific validation
        if validator and not validator(content):
            raise ValueError("Response failed brand validation")
        
        # Apply post-processing
        return self._apply_brand_postprocessing(content, brand_key)

    def _get_brand_fallback(self, brand_key: Optional[str]) -> str:
        """Get brand-specific fallback response"""
        if brand_key:
            brand_config = get_brand_config(brand_key)
            if brand_config and brand_config.get("fallback_response"):
                return brand_config.get("fallback_response")
        return "How can I assist with EV charging today?"

    def _apply_brand_postprocessing(self, content: str, brand_key: Optional[str]) -> str:
        """Apply brand-specific post-processing to response"""
        if not brand_key:
            return content
            
        brand_config = get_brand_config(brand_key)
        if not brand_config:
            return content
        
        # Remove brand-specific welcome phrases
        welcome_phrases = [
            f"Welcome to {brand_config.get('display_name', '')} support",
            brand_config.get('off_topic_response', '')
        ]
        for phrase in welcome_phrases:
            if phrase and content.startswith(phrase):
                content = content[len(phrase):].lstrip()
        
        # Apply brand-specific corrections
        corrections = brand_config.get('corrections', {})
        for wrong, correct in corrections.items():
            content = content.replace(wrong, correct)
        
        return content


async def health_check():
    """Comprehensive system health check"""
    checks = {
        "ollama_connection": False,
        "vector_store": False,
        "llm_model": False,
        "documents_available": False,
        "ollama_models": []
    }
    
    try:
        # Check Ollama connection and models
        try:
            import requests
            response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            checks["ollama_models"] = [m["name"] for m in models]
            checks["ollama_connection"] = True
            
            # Verify required models are available
            required_models = {
                Config.LLM_MODEL.lower(),
                Config.EMBEDDING_MODEL.lower()
            }
            available_models = {m.lower() for m in checks["ollama_models"]}
            checks["models_available"] = all(
                any(req in avail for avail in available_models)
                for req in required_models
            )
        except Exception as e:
            logger.error(f"Ollama connection check failed: {e}")
        
        # Check documents and vector store
        try:
            documents = DocumentLoader.load_and_prepare_documents()
            checks["documents_available"] = len(documents) > 0
            
            if checks["documents_available"]:
                # Let VectorStoreManager handle document loading internally
                vector_store = VectorStoreManager.get_vector_store(
                    brand_key="pnc",  # Using 'pnc' as shown in your logs
                    create_if_missing=True
                )
                checks["vector_store"] = vector_store is not None and hasattr(vector_store, '_collection')
        except Exception as e:
            logger.error(f"Document/vector store check failed: {e}")
            checks["documents_available"] = False
            checks["vector_store"] = False
        
        # Check LLM model
        if checks["ollama_connection"] and checks["models_available"]:
            try:
                from langchain_ollama import ChatOllama
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
                    num_thread=os.cpu_count(),  # More threads for CPU
                    top_k=20,  # More natural responses
                    top_p=0.9,  # Less restrictive
                    repeat_penalty=1.0,  # Good value
                    stop=["\nObservation:", "\n\tObservation:"],  # More robust stopping
                    keep_alive="10m",
                    num_ctx=512  # Larger context if model supports it
                )
                response = llm.invoke("ping")
                checks["llm_model"] = bool(response)
            except Exception as e:
                logger.error(f"LLM check failed: {e}")
        
        return {
            "status": "success" if all(checks.values()) else "partial",
            "checks": checks,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "missing_models": [
                model for model in [Config.LLM_MODEL, Config.EMBEDDING_MODEL]
                if model.lower() not in {m.lower() for m in checks["ollama_models"]}
            ]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "checks": checks,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }


async def main():
    """Main entry point with improved argument handling"""
    # Windows-specific output encoding
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    processor = QueryProcessor()
    thread_id = None
    user_id = None
    
    # Check for thread ID in arguments
    if "--thread" in sys.argv:
        thread_idx = sys.argv.index("--thread")
        if thread_idx + 1 < len(sys.argv):
            thread_id = sys.argv[thread_idx + 1]
            sys.argv.pop(thread_idx)
            sys.argv.pop(thread_idx)
    
    # Check for user ID in arguments (optional)
    if "--user" in sys.argv:
        user_idx = sys.argv.index("--user")
        if user_idx + 1 < len(sys.argv):
            user_id = sys.argv[user_idx + 1]
            sys.argv.pop(user_idx)
            sys.argv.pop(user_idx)
    
    # Handle different command modes
    if "--health-check" in sys.argv:
        result = await health_check()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "success" else 1)
    
    if "--init-only" in sys.argv:
        try:
            health = await health_check()
            if health["status"] != "success":
                print(json.dumps(health, indent=2))
                sys.exit(1)
                
            await processor._ensure_initialized()
            print(json.dumps({
                "status": "success",
                "message": "RAG system initialized",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "health": health["checks"]
            }))
            sys.exit(0)
        except Exception as e:
            print(json.dumps({
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }))
            sys.exit(1)

    # Normal query processing
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "error": "No question provided",
            "usage": {
                "query": "python chat.py 'Your question'",
                "health_check": "python chat.py --health-check",
                "init": "python chat.py --init-only"
            }
        }))
        sys.exit(1)

    question = " ".join(sys.argv[1:]).strip()
    if "--stream" in sys.argv:
        question = " ".join(arg for arg in sys.argv[1:] if not arg.startswith("--")).strip()
        async for chunk in processor.stream_query(question, thread_id, user_id):
            print(chunk, end='', flush=True)
    else:
        result = await processor.process_query(question, thread_id, user_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        logger.info(f"💡 Total startup+processing time: {time.time() - start_total:.2f}s")
        sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": f"System failure: {str(e)}",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "troubleshooting": {
                "ollama": "Run 'ollama serve' in another terminal",
                "vector_store": f"Check directory: {Config.PERSIST_DIRECTORY}",
                "documents": f"Verify documents exist in brand directories: {Config.DOCUMENTS_DIR}",
                "models": f"Ensure models are pulled: ollama pull {Config.LLM_MODEL} && ollama pull {Config.EMBEDDING_MODEL}"
            }
        }))
        sys.exit(1)


_processor = QueryProcessor()

async def process_query_async(question: str, thread_id: Optional[str] = None, user_id: Optional[str] = None):
    return await _processor.process_query(question, thread_id, user_id)

def process_query_sync(question: str, thread_id: Optional[str] = None, user_id: Optional[str] = None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(process_query_async(question, thread_id, user_id))