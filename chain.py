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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffer = ""
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

    def format_currency_values(self, text: str) -> str:
        preserved = {}
        patterns_to_preserve = [
            r'\b\d{1,3}(?:,\d{3})*\s*(?:kW|kilowatt[s]?|meters?|feet)\b',
            r'\b\d{1,3}(?:,\d{3})*\s*x\s*\d{1,3}(?:,\d{3})*\b'
        ]
        for i, pat in enumerate(patterns_to_preserve):
            for j, m in enumerate(re.finditer(pat, text, flags=re.IGNORECASE)):
                placeholder = f"__PRESERVED_{i}_{j}__"
                preserved[placeholder] = m.group(0)
                text = text.replace(m.group(0), placeholder)

        currency_pattern = re.compile(
            r'(?<![a-zA-Z0-9-])(?:\u20b1|PHP|P|pesos)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\b',
            flags=re.IGNORECASE
        )

        def replacer(match):
            try:
                number = float(match.group(1).replace(",", ""))
                return f"{number:,.0f} pesos"
            except:
                return match.group(0)

        text = currency_pattern.sub(replacer, text)
        for k, v in preserved.items():
            text = text.replace(k, v)
        return text

    def _process_buffer(self) -> tuple[List[str], str]:
        self.buffer = self._normalize_spacing(self.buffer)
        self.buffer = self.format_currency_values(self.buffer)

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
        for chunk in input:
            content = str(getattr(chunk, 'content', chunk))
            if content:
                self.buffer += content
                complete, remaining = self._process_buffer()
                for word in complete:
                    yield word
                self.buffer = remaining
        if self.buffer:
            cleaned = self._normalize_spacing(self.buffer)
            yield self.format_currency_values(cleaned)

    async def atransform_stream(self, input: AsyncIterator[Any]) -> AsyncIterator[str]:
        self.buffer = ""
        async for chunk in input:
            content = str(getattr(chunk, 'content', chunk))
            if content:
                self.buffer += content
                complete, remaining = self._process_buffer()
                for word in complete:
                    yield word
                self.buffer = remaining
        if self.buffer:
            cleaned = self._normalize_spacing(self.buffer)
            yield self.format_currency_values(cleaned)

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
    
    SYSTEM_PROMPT = """You are a customer support assistant for {brand_name}. You will receive relevant reference content in CONTEXT and must respond naturally and professionally.

    RULES:
    1. Use only content from CONTEXT if it answers the question exactly.
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
        """Universal document formatter that handles all input types"""
        formatted = []
        for i, doc in enumerate(docs[:3]):  # Only top 3 docs
            try:
                # Initialize variables
                content = ""
                metadata = {}
                doc_scores = {}
                # logger.info(f"\n[FORMATTING]:\n\n{doc}")
                
                # Handle LangChain Document objects
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = getattr(doc, 'metadata', {})
                    # Check for scores in both Document attributes and metadata
                    doc_scores = {
                        'relevance_score': getattr(doc, 'relevance_score', None),
                        'question_score': getattr(doc, 'question_score', None)
                    }
                
                # Handle dictionary inputs
                elif isinstance(doc, dict):
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    # Check for scores at both levels
                    doc_scores = {
                        'relevance_score': doc.get('relevance_score'),
                        'question_score': doc.get('question_score')
                    }
                
                # Get final scores (priority: doc level > metadata level > default 0)
                rel_score = float(
                    doc_scores.get('relevance_score') or 
                    metadata.get('relevance_score') or 
                    0
                )
                question_score = float(
                    doc_scores.get('question_score') or 
                    metadata.get('question_score') or 
                    0
                )
                total_score = min(10, (rel_score + question_score))

                # Clean and format content
                content = re.sub(r'\b(doc(ument)?\s*\d+)\b', '', content, flags=re.IGNORECASE).strip()
                
                # Format source information
                source = metadata.get('source', 'unknown')
                source_name = source.split('/')[-1].split('\\')[-1] if source != 'unknown' else 'unknown'
                
                # Build the formatted entry
                formatted.append(
                    f"{content}\n"
                )
                
            except Exception as e:
                logger.error(f"Error formatting doc {i} of type {type(doc)}: {str(e)}")
                continue
        
        return "\n\n".join(formatted) if formatted else "No valid documents available"

    # def process_output(chunk: Any) -> str:
    #     """Simplified output processor that maintains streaming"""
    #     if isinstance(chunk, AIMessageChunk):
    #         content = chunk.content
    #         if isinstance(content, str) and any(char.isdigit() for char in content):
    #             content = format_currency_values(content)
    #         return content
    #     elif hasattr(chunk, 'content'):
    #         content = str(chunk.content)
    #         if any(char.isdigit() for char in content):
    #             content = format_currency_values(content)
    #         return content
    #     else:
    #         content = str(chunk)
    #         if any(char.isdigit() for char in content):
    #             content = format_currency_values(content)
    #         return content

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