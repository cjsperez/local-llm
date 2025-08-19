import json
from typing import List, Dict, Optional
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from config import Config
from pathlib import Path
import logging

from utils.brand_utils import (
    get_brand_config,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the following context to answer the question:

Context: {context}

Conversation History: {conversation_history}

Question: {question}

Guidelines:
1. Respond concisely (under {word_limit} words)
2. If unsure, say "I don't have enough information about that"
3. Never mention "documents" or "files"
"""

DEFAULT_SYSTEM_MESSAGE = """
You are a helpful AI assistant. Follow these rules:
1. Be polite and professional
2. Keep responses under {word_limit} words
3. If the question is unrelated to your knowledge, respond: "{off_topic_response}"
4. Correct these common misspellings: {corrections}
"""

class DocumentLoader:
    @staticmethod
    def validate_json_path(path: Path) -> bool:
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found at {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"JSON file at {path} is empty")
        return True
    
    @staticmethod
    def get_brand_document_paths(brand_key: str) -> List[Path]:
        """Safely get document paths for a brand with validation"""
        try:
            brand_key = brand_key.lower().strip()
            DocumentLoader.validate_json_path(Config.BRAND_DOCUMENTS_FILE)
            
            with open(Config.BRAND_DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                brand_map = json.load(f)
            
            if brand_key not in brand_map:
                raise ValueError(f"Brand '{brand_key}' not found in document mapping")
                
            return [Path(p) for p in brand_map[brand_key] if Path(p).exists()]
        except Exception as e:
            logger.error(f"Failed to get document paths: {str(e)}")
            raise
    
    def _get_template_for_brand(self, brand_key: Optional[str] = None) -> str:
        """Get prompt template from brand config"""
        if not brand_key:
            return DEFAULT_PROMPT_TEMPLATE
            
        try:
            brand_config = get_brand_config(brand_key)
            return brand_config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        except Exception as e:
            logger.warning(f"Failed to get template for {brand_key}: {str(e)}")
            return DEFAULT_PROMPT_TEMPLATE

    def _get_system_message(self, brand_key: Optional[str] = None) -> str:
        """Get system message from brand config"""
        if not brand_key:
            return DEFAULT_SYSTEM_MESSAGE
            
        try:
            brand_config = get_brand_config(brand_key)
            # Format with brand details
            return brand_config.get("system_message", DEFAULT_SYSTEM_MESSAGE).format(
                display_name=brand_config.get("display_name", ""),
                corrections=json.dumps(brand_config.get("corrections", {})),
                word_limit=brand_config.get("word_limit", 30),
                off_topic_response=brand_config.get("off_topic_response", "")
            )
        except Exception as e:
            logger.warning(f"Failed to get system message for {brand_key}: {str(e)}")
            return DEFAULT_SYSTEM_MESSAGE

    @staticmethod
    def process_item(brand_key: str, source_path: str, item: Dict) -> Optional[Dict]:
        """More robust document processing with score preservation"""
        try:
            if not isinstance(item, dict):
                logger.warning(f"Invalid document format in {source_path}")
                return None
                
            required_fields = {'question', 'text', 'category'}
            if not required_fields.issubset(item.keys()):
                logger.warning(f"Missing required fields in {source_path}")
                return None
                
            # Ensure brand is properly set and not 'None'
            item_brand = str(item.get('brand', brand_key)).lower()
            if item_brand in ['none', 'null', '']:
                item_brand = brand_key
                
            return {
                'content': str(item.get('text', '')),
                'question': str(item.get('question', 'Unquestiond')),
                'category': str(item.get('category', 'General')),
                'source': source_path,
                'brand': item_brand,  # Use corrected brand value
                'metadata': {
                    **{
                        k: str(v) if not isinstance(v, (dict, list)) else json.dumps(v)
                        for k, v in item.items()
                        if k not in required_fields
                    },
                    'brand': item_brand  # Ensure brand is in metadata too
                }
            }
        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            return None

    @staticmethod
    def load_json_data(brand_key: Optional[str] = None) -> List[Dict]:
        """Load and process JSON data with improved brand handling"""
        try:
            if brand_key is None:
                # Load all brands but preserve their original brand metadata
                with open(Config.BRAND_DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                    brand_map = json.load(f)
                
                all_processed = []
                for current_brand in brand_map.keys():
                    try:
                        processed = DocumentLoader.load_json_data(current_brand)
                        all_processed.extend(processed)
                    except Exception as e:
                        logger.error(f"Failed to load {current_brand}: {str(e)}")
                        continue
                
                return all_processed
            
            # Process single brand case
            brand_key = str(brand_key).strip().lower()
            DocumentLoader.validate_json_path(Config.BRAND_DOCUMENTS_FILE)

            with open(Config.BRAND_DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                brand_map = json.load(f)

            doc_paths = brand_map.get(brand_key, [])
            all_processed = []
            
            for path_str in doc_paths:
                path = Path(path_str)
                try:
                    with open(path, 'r', encoding='utf-8') as doc_file:
                        content = json.load(doc_file)
                        
                        processed_items = [
                            DocumentLoader.process_item(brand_key, str(path), item)
                            for item in content if isinstance(item, dict)
                        ]
                        all_processed.extend([p for p in processed_items if p is not None])
                        
                except Exception as e:
                    logger.error(f"Failed to process {path}: {str(e)}")
                    continue

            return all_processed
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise

    @staticmethod
    def chunk_documents(documents: List[Dict]) -> List[Document]:
        try:
            logger.info("Starting document chunking")
            if not documents:
                logger.warning("Empty documents list received")
                return []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""]
            )

            chroma_docs = []
            for doc_idx, doc in enumerate(documents):
                try:
                    # Ensure brand is properly set in metadata
                    metadata = doc.get('metadata', {})
                    brand = str(doc.get('brand', metadata.get('brand', 'unknown')))
                    if brand.lower() in ['none', 'null']:
                        brand = 'unknown'
                    
                    core_metadata = {
                        'question': str(doc.get('question', f'Document_{doc_idx}')),
                        'category': str(doc.get('category', 'General')),
                        'type': str(doc.get('type', 'faq')),
                        'source': str(doc.get('source', 'unknown')),
                        'brand': brand,  # Use corrected brand value
                    }
                    full_metadata = {**core_metadata, **metadata}
                    
                    try:
                        filtered_metadata = filter_complex_metadata(full_metadata)
                    except Exception:
                        filtered_metadata = core_metadata

                    content = str(doc.get('content', ''))
                    if not content:
                        continue

                    if len(content) > Config.CHUNK_SIZE * 1.5:
                        chunks = text_splitter.split_text(content)
                        for chunk in chunks:
                            chroma_docs.append(Document(
                                page_content=chunk,
                                metadata=filtered_metadata.copy()
                            ))
                    else:
                        chroma_docs.append(Document(
                            page_content=content,
                            metadata=filtered_metadata.copy()
                        ))
                except Exception as e:
                    logger.error(f"Error processing doc {doc_idx}: {str(e)}")

            logger.info(f"Created {len(chroma_docs)} documents")
            return chroma_docs
        except Exception as e:
            logger.error(f"Critical failure in chunk_documents: {str(e)}")
            return []

    @staticmethod
    def load_and_prepare_documents(brand_key: Optional[str] = None) -> List[Document]:
        """Load and prepare documents with proper brand metadata handling"""
        try:
            raw_docs = DocumentLoader.load_json_data(brand_key)
            chroma_docs = DocumentLoader.chunk_documents(raw_docs)
            
            # Ensure brand metadata is properly set
            for doc in chroma_docs:
                if 'brand' not in doc.metadata or not doc.metadata['brand']:
                    doc.metadata['brand'] = brand_key or 'unknown'
                    
            return chroma_docs
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise ValueError("No valid documents after processing")

    @classmethod
    def load_brand_documents(cls, brand_key: Optional[str] = None) -> List[Document]:
        try:
            logger.info(f"Loading documents for brand: {brand_key or 'all'}")
            all_docs = cls.load_and_prepare_documents(brand_key)
            if not brand_key:
                return all_docs
            filtered = [doc for doc in all_docs if doc.metadata.get("brand") == brand_key]
            logger.info(f"Found {len(filtered)} documents for brand {brand_key}")
            return filtered
        except Exception as e:
            logger.error(f"Failed to load brand documents: {str(e)}")
            raise

    @staticmethod
    def load_document_by_id(doc_id: str, all_docs: List[Dict]) -> Optional[Dict]:
        """Find a document by its ID in the document list"""
        return next((doc for doc in all_docs if doc.get('id') == doc_id or doc.get('hash') == doc_id), None)    
    
    @staticmethod
    def process_document(doc_info: Dict) -> List[Dict]:
        """Process a single document into the required format"""
        try:
            # Handle JSON file case
            if doc_info.get('extension') == '.json' and not doc_info.get('path', '').startswith('http'):
                with open(doc_info['path'], 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        return [{
                            'content': str(item.get('text', '')),
                            'question': str(item.get('question', '')),
                            'source': doc_info['path'],
                            'metadata': {
                                'category': str(item.get('category', '')),
                                'id': str(item.get('id', '')),
                                'brand': doc_info.get('brand', '')
                            }
                        } for item in content]
            
            # Handle FAQ case (second document in your example)
            if 'faqs' in doc_info:
                return [{
                    'content': str(faq.get('text', '')),
                    'question': str(faq.get('question', '')),
                    'source': doc_info.get('source', ''),
                    'metadata': {
                        'category': str(faq.get('category', '')),
                        'id': str(faq.get('id', '')),
                        'brand': doc_info.get('brand', '')
                    }
                } for faq in doc_info['faqs']]
                
            # Handle URL case
            if doc_info.get('path', '').startswith('http'):
                return [{
                    'content': f"URL content from {doc_info['path']}",
                    'source': doc_info['path'],
                    'metadata': {
                        'type': 'website',
                        'brand': doc_info.get('brand', '')
                    }
                }]
                
        except Exception as e:
            logger.error(f"Failed to process document {doc_info.get('path')}: {str(e)}")
        return []

    @classmethod
    def check_documents_exist(cls, brand_key: Optional[str] = None) -> bool:
        """Health check helper to verify document availability"""
        try:
            if brand_key:
                paths = cls.get_brand_document_paths(brand_key)
                return len(paths) > 0
            else:
                DocumentLoader.validate_json_path(Config.BRAND_DOCUMENTS_FILE)
                with open(Config.BRAND_DOCUMENTS_FILE) as f:
                    brand_map = json.load(f)
                return any(Path(p).exists() for paths in brand_map.values() for p in paths)
        except Exception:
            return False

    @classmethod
    def get_all_brands(cls) -> List[str]:
        """Get all available brands from both document paths and metadata"""
        try:
            # Get brands from document mapping file
            with open(Config.BRAND_DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                brand_map = json.load(f)
            file_brands = set(brand_map.keys())
            
            # Get brands from actual document metadata
            docs = cls.load_and_prepare_documents()
            meta_brands = {doc.metadata.get("brand") for doc in docs if doc.metadata.get("brand")}
            
            # Combine and deduplicate
            all_brands = sorted(file_brands.union(meta_brands))
            logger.info(f"Found {len(all_brands)} brands: {all_brands}")
            return all_brands
        except Exception as e:
            logger.error(f"Failed to get brands: {str(e)}")
            return []
