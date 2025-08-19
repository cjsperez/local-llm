from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
from config import Config
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    @staticmethod
    def validate_input_document(doc: Dict) -> bool:
        """Validate the structure of input documents"""
        required_fields = {'text', 'question', 'category'}
        return all(field in doc for field in required_fields)

    @staticmethod
    def split_documents(documents: List[Dict]) -> List[Document]:
        """Split and convert documents with enhanced error handling and logging."""
        if not documents:
            logger.warning("Empty document list received")
            return []

        # Initialize splitter with safe defaults
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True
        )

        chunks = []
        processed_count = 0
        error_count = 0

        for doc in documents:
            try:
                if not TextProcessor.validate_input_document(doc):
                    logger.warning(f"Skipping invalid document: {doc.get('question', 'Unquestiond')}")
                    error_count += 1
                    continue

                # Safely get optional fields
                pronounced = doc.get('pronounced', '')
                additional_metadata = {
                    k: v for k, v in doc.items()
                    if k not in {'text', 'question', 'category', 'pronounced'}
                }

                # Split the text
                doc_chunks = text_splitter.split_text(doc["text"])
                processed_count += 1

                # Create document chunks
                for chunk in doc_chunks:
                    chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            "question": doc["question"],
                            "category": doc["category"],
                            "pronounced": pronounced,
                            **additional_metadata
                        }
                    ))

            except Exception as e:
                logger.error(f"Error processing document {doc.get('question')}: {str(e)}")
                error_count += 1
                continue

        logger.info(
            f"Processed {processed_count} documents into {len(chunks)} chunks "
            f"({error_count} errors)"
        )
        return chunks

    @staticmethod
    def process_faq_documents(faqs: List[Dict]) -> List[Document]:
        """Special processing for FAQ-style documents"""
        processed = []
        for faq in faqs:
            try:
                if not all(k in faq for k in ['question', 'answer', 'category']):
                    continue

                processed.append(Document(
                    page_content=f"Q: {faq['question']}\nA: {faq['answer']}",
                    metadata={
                        "question": faq.get('question', 'FAQ'),
                        "category": faq['category'],
                        "type": "faq",
                        **{k: v for k, v in faq.items() 
                           if k not in ['question', 'answer', 'category']}
                    }
                ))
            except Exception as e:
                logger.error(f"Error processing FAQ: {str(e)}")
        return processed