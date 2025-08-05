from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from typing import Dict, Optional, Any, Union, AsyncIterator, Iterator, List, ClassVar
import logging
import re
from utils.brand_utils import get_brand_config
from langchain_core.messages import AIMessageChunk
import codecs
from pydantic import Field




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
        return text.strip()

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
        self.buffer = self._normalize_spacing(self.buffer)

        custom_pattern = r"|".join(map(re.escape, self.tech_terms))
        token_pattern = rf"\b(?:{custom_pattern}|[A-Za-z][A-Za-z0-9\-]*[.,;:]?)\b"

        matches = list(re.finditer(token_pattern, self.buffer, flags=re.IGNORECASE))
        tokens = [m.group() for m in matches]

        if tokens:
            last_match_end = matches[-1].end()
            complete_tokens = tokens[:-1] if last_match_end < len(self.buffer) else tokens
            remainder = self.buffer[last_match_end:] if last_match_end < len(self.buffer) else ""
            return complete_tokens, remainder

        return [], self.buffer

    def transform_stream(self, input: Iterator[Any]) -> Iterator[str]:
        self.buffer = ""
        self.currency_buffer = []

        for chunk in input:
            content = str(getattr(chunk, 'content', chunk))
            if content:
                self.currency_buffer.append(content)
                merged_text, flushed = self._flush_currency_buffer()
                if flushed:
                    self.buffer += merged_text
                    complete, remaining = self._process_buffer()
                    for word in complete:
                        yield word
                    self.buffer = remaining

        if self.currency_buffer:
            merged_text, _ = self._flush_currency_buffer(force=True)
            self.buffer += merged_text
        if self.buffer:
            cleaned = self._normalize_spacing(self.buffer)
            yield cleaned

    async def atransform_stream(self, input: AsyncIterator[Any]) -> AsyncIterator[str]:
        self.buffer = ""
        self.currency_buffer = []

        async for chunk in input:
            content = str(getattr(chunk, 'content', chunk))
            if content:
                self.currency_buffer.append(content)
                merged_text, flushed = self._flush_currency_buffer()
                if flushed:
                    self.buffer += merged_text
                    complete, remaining = self._process_buffer()
                    for word in complete:
                        yield word
                    self.buffer = remaining

        if self.currency_buffer:
            merged_text, _ = self._flush_currency_buffer(force=True)
            self.buffer += merged_text
        if self.buffer:
            cleaned = self._normalize_spacing(self.buffer)
            yield cleaned

    def stream(self, input: Any) -> Iterator[str]:
        return self.transform_stream(super().stream(input))

    async def astream(self, input: Any) -> AsyncIterator[str]:
        async for chunk in super().astream(input):
            async for word in self.atransform_stream([chunk]):
                yield word


def create_rag_chain( llm, brand_key: Optional[str] = None) -> Any:
    """
    Creates a strict RAG chain that:
    - Only answers with exact location info when asked
    - Never includes app instructions in location responses
    - Maintains call center agent tone
    """
    brand_config = get_brand_config(brand_key) if brand_key else {}
    
    SYSTEM_PROMPT = """You represent {brand_name} as their customer support assistant. You will receive relevant reference content in CONTEXT and must respond naturally and professionally.

    RULES:
    1. Use only content from CONTEXT if it answers the question exactly, if the answer is not in the context do not try to answer the question randomly.
    2. Do not mention any documents, files, or instructions.
    3. Never add steps, app instructions, or tech details unless directly mentioned in the context.
    4. Do not include your own reasoning.
    5. Never mention "based on the context", "based on the documents", or similar phrases."""
    
    HUMAN_PROMPT = """CONTEXT:
    {context}

    QUESTION: {question}

    SHORT ANSWER:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT)
    ])

    

    def format_docs(docs):
        """Universal document formatter that prioritizes docs with highest (relevance_score + question_score)"""
        formatted = []
        scored_docs = []

        for doc in docs:
            try:
                # Initialize variables
                content = ""
                metadata = {}
                doc_scores = {}
                
                # Handle LangChain Document objects
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = getattr(doc, 'metadata', {})
                    doc_scores = {
                        'relevance_score': getattr(doc, 'relevance_score', None),
                        'question_score': getattr(doc, 'question_score', None)
                    }
                
                # Handle dictionary inputs
                elif isinstance(doc, dict):
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    doc_scores = {
                        'relevance_score': doc.get('relevance_score'),
                        'question_score': doc.get('question_score')
                    }
                
                # Calculate total score (default to 0 if missing)
                rel_score = float(doc_scores.get('relevance_score') or 0)
                question_score = float(doc_scores.get('question_score') or 0)
                total_score = rel_score + question_score

                # Clean content
                content = re.sub(r'\b(doc(ument)?\s*\d+)\b', '', content, flags=re.IGNORECASE).strip()
                
                # Store doc with its score for sorting
                scored_docs.append({
                    'content': content,
                    'total_score': total_score
                })
                
            except Exception as e:
                logger.error(f"Error formatting doc of type {type(doc)}: {str(e)}")
                continue
        
        # Sort docs by total_score (highest first) and take top 3
        scored_docs.sort(key=lambda x: x['total_score'], reverse=True)
        top_docs = scored_docs[:3]
        
        # Extract and format the content
        formatted = [doc['content'] for doc in top_docs]
        
        
        return "\n\n".join(formatted) if formatted else "No valid documents available"

    return (
        {
            "brand_name": lambda x: brand_config.get("display_name", "our service"),
            "question": itemgetter("question"),
            "context": itemgetter("context") | RunnableLambda(format_docs),
            "conversation_history": lambda x: x.get("conversation_history", "No history"),
        }
        | prompt
        | llm
        | TechnicalTermStreamer()
    )