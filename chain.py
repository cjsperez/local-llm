from datetime import datetime
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from typing import Dict, Optional, Any, Union, AsyncIterator, Iterator, List, ClassVar
import logging
import sys
import re
from pathlib import Path
from utils.brand_utils import get_brand_config
from langchain_core.messages import AIMessageChunk
import codecs
from pydantic import Field


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(name)s - %(levelname)s]: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class TechnicalTermStreamer(StrOutputParser):
    word_boundary_re: ClassVar[re.Pattern] = re.compile(r'(\s+|(?<=\w)[,.:;](?=\s|$)|(?<=\d)\s*\.\s*(?=\d))')
    tech_terms: set[str] = Field(default_factory=set, exclude=True)
    known_corrections: set[str] = Field(default_factory=set, exclude=True)
    buffer: str = Field(default="", exclude=True)
    currency_buffer: List[str] = Field(default_factory=list, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffer = ""
        self.currency_buffer = []
        self.tech_terms = {
            'ev', 'qr', 'gcash', 'maya', 'parkncharge', 
            'gunplug', 'chargingstation', 'merchantportal',
            'fireisolator', 'firesuppressant', 'lithium-ion', 'electricvehicle'
        }
        self.known_corrections = {
            r"fire\s+is\s*ol\s*ator": "Fire Isolator",
            r"suppress\s+ant": "suppressant",
            r"lithium\s*[-]?\s*ion": "lithium-ion",
            r"electric\s+vehicle": "electric vehicle",
            r"\(\s*ev\s*\)": "(EV)"
        }

    def _extract_content(self, chunk: Any) -> str:
        logger.info(f"[DEBUG] _extract_content called with type: {type(chunk)}, value: {chunk!r}")
        """Robust content extraction from various formats"""
        if chunk is None:
            return ""
            
        if isinstance(chunk, str):
            return chunk
            
        if isinstance(chunk, dict):
            # Handle LLM response formats
            if 'content' in chunk:
                return str(chunk['content'])
            if 'text' in chunk:
                return str(chunk['text'])
            if 'message' in chunk and isinstance(chunk['message'], dict):
                return str(chunk['message'].get('content', ''))
            if 'choices' in chunk and isinstance(chunk['choices'], list):
                if chunk['choices'] and isinstance(chunk['choices'][0], dict):
                    return str(chunk['choices'][0].get('text', ''))
            return str(chunk)
            
        if hasattr(chunk, 'content'):
            return str(getattr(chunk, 'content', ''))
            
        return str(chunk)

    def _normalize_spacing(self, text: str) -> str:
        for pattern, replacement in self.known_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        text = re.sub(r'\s+([.,;:)](?:\s|$))', r'\1 ', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        text = re.sub(r'(\d+)\s*\.\s+', r'\1. ', text)
        text = re.sub(r'([.,;:])\s+', r'\1 ', text)
        text = re.sub(r'(\d)\s+([kK][Ww])', r'\1\2', text)
        text = re.sub(r'\be\s*\.\s*g\s*\.', 'e.g.', text)
        text = re.sub(r'\bi\s*\.\s*e\s*\.', 'i.e.', text)
        text = re.sub(r'g\s*[-]?\s*c\s*[-]?\s*a\s*[-]?\s*s\s*[-]?\s*h', 'GCash', text, flags=re.IGNORECASE)
        text = re.sub(r'e\s*[-]?\s*w\s*[-]?\s*a\s*[-]?\s*l\s*[-]?\s*l\s*[-]?\s*e\s*[-]?\s*t', 'e-wallet', text, flags=re.IGNORECASE)
        text = re.sub(r'g\s*[-]?\s*u\s*[-]?\s*n\s*[-]?\s*p\s*[-]?\s*l\s*[-]?\s*u\s*[-]?\s*g', 'GunPlug', text, flags=re.IGNORECASE)
        return text
    
    def _log_chunk_structure(self, chunk: Any, prefix: str = ""):
        """Logs the detailed structure of an incoming chunk"""
        if isinstance(chunk, dict):
            logger.info(f"{prefix}Chunk is dict with keys: {list(chunk.keys())}")
            for k, v in chunk.items():
                if isinstance(v, (str, int, float, bool)):
                    logger.info(f"{prefix}  {k}: {v!r}")
                elif isinstance(v, (list, tuple)):
                    logger.info(f"{prefix}  {k}: list[{len(v)}]")
                elif isinstance(v, dict):
                    logger.info(f"{prefix}  {k}: dict[{len(v)}]")
                    self._log_chunk_structure(v, prefix + "    ")
                else:
                    logger.info(f"{prefix}  {k}: {type(v)}")
        elif hasattr(chunk, '__dict__'):
            logger.info(f"{prefix}Chunk is object with attributes: {vars(chunk)}")
        else:
            logger.info(f"{prefix}Chunk is {type(chunk)}: {chunk!r}")

    def _log_state(self):
        """Logs the current state of the streamer"""
        logger.info(f"Streamer state - buffer: {self.buffer!r}")
        logger.info(f"Streamer state - currency_buffer: {self.currency_buffer!r}")

    def _flush_currency_buffer(self, force: bool = False) -> tuple[str, bool]:
        joined = "".join(self.currency_buffer).strip()
        currency_pattern = re.compile(
            r'(?:₱|PHP|P)?\s*(\d{1,3})(?:[ ,]?(\d{3}))(?:[ ,]?(\d{3}))?'
        )

        match = currency_pattern.match(joined)
        if match:
            number = "".join(filter(None, match.groups()))
            try:
                formatted = f"{int(number):,} pesos"
                self.currency_buffer.clear()
                return formatted + " ", True
            except:
                return "", False

        if force:
            flushed = "".join(self.currency_buffer)
            self.currency_buffer.clear()
            return flushed, True

        return "", False

    def _process_buffer(self) -> tuple[List[str], str]:
        logger.info(f"[DEBUG] _process_buffer called. self.buffer type: {type(self.buffer)}, value: {self.buffer!r}")
        try:
            buf = self._extract_content(self.buffer)
            logger.info(f"[DEBUG] _extract_content returned type: {type(buf)}, value: {buf!r}")
            self.buffer = self._normalize_spacing(buf)
            logger.info(f"[DEBUG] After normalization, buffer type: {type(self.buffer)}, value: {self.buffer!r}")
            parts = self.word_boundary_re.split(self.buffer)
            logger.info(f"[DEBUG] word_boundary_re.split result type: {type(parts)}, value: {parts!r}")
            if not parts:
                return [], ""
            complete_parts = parts[:-1]
            remainder = parts[-1] if parts else ""
            return complete_parts, remainder
        except Exception as e:
            logger.error(f"Buffer processing error: {str(e)}")
            logger.error(f"[DEBUG] Exception buffer type: {type(self.buffer)}, value: {self.buffer!r}")
            return [], str(self.buffer)

    async def atransform_stream(self, input: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Asynchronous version of word-by-word streaming with minimal buffering"""
        buffer = ""
        async for chunk in input:
            content = self._extract_content(chunk)
            if not content:
                continue
                
            buffer += content
            
            # Process word boundaries incrementally
            while True:
                match = self.word_boundary_re.search(buffer)
                if not match:
                    break
                    
                # Extract up to the boundary
                word_end = match.end()
                word_text = buffer[:word_end]
                buffer = buffer[word_end:]
                
                # Process and yield the word
                processed = self._normalize_spacing(word_text)
                if processed:
                    yield processed
        
        # Yield any remaining content
        if buffer:
            yield self._normalize_spacing(buffer)
            
    def _process_line(self, text: str) -> str:
        """Process individual lines with technical term handling"""
        processed = text
        for term in self.tech_terms:
            if term in processed.lower():
                processed = processed.replace(term, f"**{term.upper()}**")
        return processed

    def transform_stream(self, input: Iterator[Any]) -> Iterator[str]:
        """Synchronous version of word-by-word streaming"""
        buffer = ""
        for chunk in input:
            content = self._extract_content(chunk)
            if not content:
                continue
                
            buffer += content
            
            # Process word boundaries incrementally
            while True:
                match = self.word_boundary_re.search(buffer)
                if not match:
                    break
                    
                word_end = match.end()
                word_text = buffer[:word_end]
                buffer = buffer[word_end:]
                
                processed = self._normalize_spacing(word_text)
                if processed:
                    yield processed
        
        if buffer:
            yield self._normalize_spacing(buffer)

    def stream(self, input: Any) -> Iterator[str]:
        """Synchronous streaming entry point"""
        logger.info(f"[DEBUG] stream called with input type: {type(input)}")
        # Convert single input to iterator if needed
        if not isinstance(input, Iterator):
            input = iter([input])
        return self.transform_stream(input)
    
    async def astream(self, input: AsyncIterator[Any]) -> AsyncIterator[dict]:
        """Asynchronous streaming with immediate chunk emission"""
        thread_id = None
        async for chunk in input:
            # Extract content and thread_id
            content = self._extract_content(chunk)
            if isinstance(chunk, dict) and "thread_id" in chunk:
                thread_id = chunk["thread_id"]
            
            if not content:
                continue
                
            # Immediate yield with minimal processing
            yield {
                'text': content,
                'status': 'in_progress',
                'timestamp': datetime.now().isoformat(),
                'thread_id': thread_id
            }
        
        # Final completion message
        yield {
            'status': 'complete',
            'timestamp': datetime.now().isoformat(),
            'thread_id': thread_id
        }

def create_rag_chain(llm, retriever=None, brand_key: Optional[str] = None) -> Any:
    """
    Creates a RAG chain that optionally incorporates a retriever.
    Maintains strict response rules and call center agent tone.
    """
    brand_config = get_brand_config(brand_key) if brand_key else {}
    
    SYSTEM_PROMPT = """You represent {brand_name} as their customer support assistant. 
        You have access to:
        1. CONTEXT — factual information relevant to the current question.
        2. CONVERSATION HISTORY — previous exchanges in this chat.

        RULES:
        1. Use CONTEXT for factual accuracy.
        2. Use CONVERSATION HISTORY to understand what was already said, avoid repeating, and maintain continuity.
        3. If the user is asking for more details about something mentioned earlier, refer back to the history before answering.
        4. If the answer is not in CONTEXT or cannot be inferred from history, say: "I'm sorry, I don't have that information. If you have any question related to {brand_name}, feel free to ask!"
        5. Never mention CONTEXT or CONVERSATION_HISTORY explicitly.
        6. Keep responses natural and professional.
        
        ONLY answer what was asked.
        """

    
    
    HUMAN_PROMPT = """CONTEXT:
    {context}

    QUESTION: {question}

    SHORT ANSWER:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT)
    ])

    def format_docs(docs):
        """Format documents for context injection with robust error handling"""
        if not docs:
            return "No relevant documents available"
        
        # Ensure docs is a list
        if not isinstance(docs, (list, tuple)):
            docs = [docs]
        
        formatted = []
        for doc in docs[:3]:  # Only use top 3 most relevant docs
            try:
                # Handle Document objects and dicts
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = getattr(doc, 'metadata', {})
                elif isinstance(doc, dict):
                    # Defensive extraction
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    # If content is still a dict, convert to string
                    if isinstance(content, dict):
                        content = str(content)
                else:
                    content = str(doc)
                    metadata = {}
                
                # Ensure content is a string
                if not isinstance(content, str):
                    content = str(content)
                
                # Clean and format the content
                if content:
                    content = re.sub(r'\b(doc(ument)?\s*\d+)\b', '', content, flags=re.IGNORECASE)
                    source = metadata.get('source', '')
                    if source:
                        content = f"{content}\n(Source: {Path(source).name})"
                    formatted.append(content)
            except Exception as e:
                logger.error(f"Error formatting document: {str(e)}")
                continue
        
        # Always return a string
        return "\n\n".join(formatted) if formatted else "No valid document content found"
    
    def validate_input(input_dict: Dict) -> Dict:
        """Ensure input has correct structure"""
        if not isinstance(input_dict, dict):
            raise ValueError("Input must be a dictionary")
        
        # Normalize context to always be a list
        if 'context' not in input_dict:
            input_dict['context'] = []
        elif isinstance(input_dict['context'], dict):
            input_dict['context'] = [input_dict['context']]
        elif not isinstance(input_dict['context'], (list, tuple)):
            input_dict['context'] = [input_dict['context']]
            
        return input_dict
    
    def ensure_prompt_vars_are_strings(input_dict: Dict) -> Dict:
        """Ensure all prompt variables are strings before passing to prompt."""
        for key in ["brand_name", "question", "context", "conversation_history"]:
            if key in input_dict and not isinstance(input_dict[key], str):
                input_dict[key] = str(input_dict[key])
        return input_dict
    
    def extract_query(input_dict):
        """Extract query text from input dictionary"""
        if isinstance(input_dict, dict):
            return input_dict.get("question") or input_dict.get("query") or str(input_dict)
        return str(input_dict)

    base_chain = (
        RunnableLambda(validate_input)
        | {
            "brand_name": lambda x: brand_config.get("display_name", "our service"),
            "question": itemgetter("question"),
            "context": lambda x: str(format_docs(x["context"])) if not isinstance(x["context"], str) else x["context"],
            "conversation_history": lambda x: str(x.get("conversation_history", "No history")),
        }
        | RunnableLambda(ensure_prompt_vars_are_strings)
        | prompt
        | llm.with_config({"run_name": "ollama_llm"})  # Ensure streaming is enabled
        | RunnableLambda(lambda x: x.content if hasattr(x, 'content') else str(x))
        | TechnicalTermStreamer()
    )

    if retriever:
        # Wrap the retriever to handle dict inputs
        query_aware_retriever = (
            RunnablePassthrough() 
            | RunnableLambda(extract_query)
            | retriever
        )
        
        return {
            "context": query_aware_retriever,
            "question": RunnablePassthrough()
        } | base_chain
        
    return base_chain