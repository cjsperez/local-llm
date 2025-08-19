from fastapi import FastAPI, HTTPException, Request, UploadFile, File, BackgroundTasks, status, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import json
import asyncio
from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field, ValidationError
import logging
from datetime import datetime, timezone
import time
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
from chat import logger, QueryProcessor, health_check
import requests  # Add this import
import tempfile  # Add this import
from utils.brand_utils import (
    add_brand, 
    get_brand_config,
    get_all_brands,
    update_brand,
    delete_brand,
    brand_exists
)
import os
from config import Config
import uuid 
from data_loader import DocumentLoader
from langchain_core.documents.base import Document
from utils.doc_extractor import (
    logger as doc_extractor_logger,
    extract_text_from_pdf,
    extract_text_from_url,
    ask_ollama,
    clean_llm_output,
    generate_question_variations
)

# ================ Base Models ================
class BaseResponse(BaseModel):
    status: str = Field(..., example="success")
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class SuccessResponse(BaseResponse):
    data: Optional[Union[dict, list]] = None

class ErrorResponse(BaseResponse):
    error: Optional[str] = None
    error_details: Optional[Union[dict, list]] = None
    code: Optional[int] = None

class PaginatedResponse(SuccessResponse):
    total: int
    page: int
    per_page: int

# ================ Domain Models ================
class QueryRequest(BaseModel):
    question: str
    stream: Optional[bool] = False
    thread_id: Optional[str] = None
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    result: Optional[str] = None
    processing_time: float
    thread_id: Optional[str] = None
    diagnostics: Optional[dict] = None

class BrandRequest(BaseModel):
    key: str = Field(..., example="pnc")
    display_name: str
    products: list[str] = Field(..., example=["EV charging"])
    support_email: str
    word_limit: int = 30
    corrections: dict[str, str] = Field(..., example={"Park and charge": "ParkNCharge"})
    off_topic_response: str
    prompt_template: Optional[str] = None
    system_message: Optional[str] = None
    welcome_message: Optional[str] = None
    no_info_response: Optional[str] = None
    location_fallback: Optional[str] = None
    pricing_fallback: Optional[str] = None

class BrandResponse(BaseModel):
    key: str
    display_name: str
    products: list[str]
    support_email: str
    word_limit: int
    off_topic_response: str
    corrections: dict[str, str]

class DocumentInfo(BaseModel):
    filename: str
    path: str
    size: int
    extension: str
    hash: str
    upload_date: str
    status: str
    brand: str

class VectorStoreOperationRequest(BaseModel):
    brand_key: str
    file_ids: List[str] = Field(default_factory=list)
    recreate: Optional[bool] = False

class VectorStoreStatusResponse(BaseModel):
    exists: bool
    brand: str
    path: str
    collection: str
    document_count: Optional[int] = None

class DocumentCreateRequest(BaseModel):
    brand_key: str
    source_type: str = Field(..., pattern="^(pdf|url)$")
    source: str
    extract_faqs: bool = True
    model: str = "llama3"

class DocumentProcessResponse(BaseModel):
    document_id: str
    status: str
    file_path: Optional[str] = None
    faqs: Optional[List[dict]] = None
    error: Optional[str] = None



# ================ Application Setup ================
@asynccontextmanager
async def lifespan(app: FastAPI):
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not Config.DOCUMENT_LIST.exists():
        Config.DOCUMENT_LIST.write_text(json.dumps([]))
    
    processor = QueryProcessor()
    app.state.processor = processor
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Initializing RAG system (attempt {attempt}/{max_retries})")
            break
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Startup failed after {max_retries} attempts")
                raise
            await asyncio.sleep(5)
    
    yield
    
    logger.info("Shutting down...")
    try:
        await processor.shutdown()
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

app = FastAPI(
    question="RAG System API",
    description="Retrieval-Augmented Generation System API",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================ Helper Functions ================
def get_file_hash(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def update_document_list(file_info: dict):
    current_list = json.loads(Config.DOCUMENT_LIST.read_text())
    current_list.append(file_info)
    Config.DOCUMENT_LIST.write_text(json.dumps(current_list, indent=2))

def update_brand_documents(brand_key: str, document_path: str, action: str = "add"):
    try:
        with open(Config.BRAND_DOCUMENTS_FILE, 'r+') as f:
            brand_docs = json.load(f)
            if brand_key not in brand_docs:
                brand_docs[brand_key] = []
            
            if action == "add":
                if document_path not in brand_docs[brand_key]:
                    brand_docs[brand_key].append(document_path)
            elif action == "remove":
                brand_docs[brand_key] = [d for d in brand_docs[brand_key] if d != document_path]
            
            f.seek(0)
            json.dump(brand_docs, f, indent=2)
            f.truncate()
    except Exception as e:
        logger.error(f"Brand document update error: {str(e)}")

# Add the new function here
def remove_from_brand_documents(brand_key: str, file_path: Path) -> bool:
    """Remove a file path from BRAND_DOCUMENTS_FILE"""
    if not hasattr(Config, 'BRAND_DOCUMENTS_FILE') or not Config.BRAND_DOCUMENTS_FILE.exists():
        return False
        
    try:
        with open(Config.BRAND_DOCUMENTS_FILE, 'r+') as f:
            brand_docs = json.load(f)
            if brand_key in brand_docs:
                file_path_str = str(file_path)
                brand_docs[brand_key] = [path for path in brand_docs[brand_key] if path != file_path_str]
                f.seek(0)
                json.dump(brand_docs, f, indent=2)
                f.truncate()
                return True
        return False
    except Exception as e:
        logger.error(f"Error updating BRAND_DOCUMENTS_FILE: {str(e)}")
        return False

def raw_dicts_to_documents(raw_docs: list[dict]) -> list[Document]:
    """Convert list of raw dicts to LangChain Document objects using DocumentLoader.chunk_documents."""
    return DocumentLoader.chunk_documents(raw_docs)

# ================ Brand Endpoints ================
# BRAND CREATION
@app.post("/brands", 
          response_model=SuccessResponse,
          status_code=status.HTTP_201_CREATED,
          responses={409: {"model": ErrorResponse}})
async def create_brand(data: BrandRequest):
    """
    Create a new brand with complete configuration including:
    - Brand name corrections
    - Prompt templates
    - System messages
    - Response configurations
    """
    try:
        if brand_exists(data.key):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=ErrorResponse(
                    status="error",
                    message="Brand already exists",
                    error=f"Brand key '{data.key}' is taken"
                ).model_dump()
            )
            
        # Build complete brand configuration
        brand_config = {
            "display_name": data.display_name,
            "products": data.products,
            "support_email": data.support_email,
            "word_limit": data.word_limit,
            "corrections": data.corrections,
            "off_topic_response": data.off_topic_response,
            "prompt_template": data.prompt_template or Config.DEFAULT_PROMPT_TEMPLATE,
            "system_message": data.system_message or Config.DEFAULT_SYSTEM_MESSAGE,
            "welcome_message": data.welcome_message or f"Welcome to {data.display_name} support!",
            "no_info_response": data.no_info_response or "I couldn't find information about that.",
            "location_fallback": data.location_fallback or "Location information not available",
            "pricing_fallback": data.pricing_fallback or "Pricing information not available"
        }
        
        # Validate required fields
        required_fields = ['display_name', 'products', 'support_email',]
        for field in required_fields:
            if not brand_config.get(field):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ErrorResponse(
                        status="error",
                        message="Missing required field",
                        error=f"Field '{field}' is required"
                    ).model_dump()
                )
        
        # Add the brand to configuration
        add_brand(data.key, brand_config)
        
        # Create brand directory if it doesn't exist
        brand_dir = Config.get_brand_dir(data.key)
        brand_dir.mkdir(exist_ok=True)
        
        return SuccessResponse(
            status="success",
            message="Brand created successfully",
            data={"key": data.key, **brand_config}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Brand creation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                status="error",
                message="Brand creation failed",
                error=str(e),
                error_details={"trace": str(e.__traceback__) if Config.DEBUG else None}
            ).model_dump()
        )

@app.get("/brands", response_model=SuccessResponse)
async def list_brands():
    try:
        brands = get_all_brands()
        return SuccessResponse(
            status="success",
            data=[{"key": k, **v} for k, v in brands.items()]
        )
    except Exception as e:
        logger.error(f"Brand listing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                status="error",
                message="Failed to list brands",
                error=str(e)
            ).model_dump()
        )

@app.get("/brands/{brand_key}", response_model=SuccessResponse)
async def get_brand(brand_key: str):
    try:
        config = get_brand_config(brand_key)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    status="error",
                    message="Brand not found",
                    error=f"Brand '{brand_key}' doesn't exist"
                ).model_dump()
            )
        return SuccessResponse(
            status="success",
            data={"key": brand_key, **config}
        )
    except Exception as e:
        logger.error(f"Brand retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                status="error",
                message="Brand retrieval failed",
                error=str(e)
            ).model_dump()
        )

@app.put("/brands/{brand_key}", 
         response_model=SuccessResponse,
         responses={
             400: {"model": ErrorResponse},
             404: {"model": ErrorResponse},
             409: {"model": ErrorResponse},
             500: {"model": ErrorResponse}
         })
async def update_brand_endpoint(brand_key: str, data: BrandRequest):
    try:
        if brand_key != data.key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    status="error",
                    message="Brand key mismatch",
                    error="URL and payload brand keys don't match"
                ).model_dump()
            )
            
        if not brand_exists(brand_key):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    status="error",
                    message="Brand not found",
                    error=f"Brand '{brand_key}' doesn't exist"
                ).model_dump()
            )
            
        # Build complete brand configuration matching creation endpoint
        brand_config = {
            "display_name": data.display_name,
            "products": data.products,
            "support_email": data.support_email,
            "word_limit": data.word_limit,
            "corrections": data.corrections,
            "off_topic_response": data.off_topic_response,
            "prompt_template": data.prompt_template or Config.DEFAULT_PROMPT_TEMPLATE,
            "system_message": data.system_message or Config.DEFAULT_SYSTEM_MESSAGE,
            "welcome_message": data.welcome_message or f"Welcome to {data.display_name} support!",
            "no_info_response": data.no_info_response or "I couldn't find information about that.",
            "location_fallback": data.location_fallback or "Location information not available",
            "pricing_fallback": data.pricing_fallback or "Pricing information not available"
        }
        
        # Validate required fields (same as creation)
        required_fields = ['display_name', 'products', 'support_email', 'corrections']
        for field in required_fields:
            if not brand_config.get(field):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ErrorResponse(
                        status="error",
                        message="Missing required field",
                        error=f"Field '{field}' is required"
                    ).model_dump()
                )
        
        update_brand(data.key, brand_config)
        return SuccessResponse(
            status="success",
            message="Brand updated successfully",
            data={"key": data.key, **brand_config}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Brand update failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                status="error",
                message="Brand update failed",
                error=str(e),
                error_details={"trace": str(e.__traceback__) if Config.DEBUG else None}
            ).model_dump()
        )

@app.delete("/brands/{brand_key}", response_model=SuccessResponse)
async def delete_brand_endpoint(brand_key: str):
    try:
        if not brand_exists(brand_key):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    status="error",
                    message="Brand not found",
                    error=f"Brand '{brand_key}' doesn't exist"
                ).model_dump()
            )
            
        delete_brand(brand_key)
        return SuccessResponse(
            status="success",
            message=f"Brand '{brand_key}' deleted successfully"
        )
    except Exception as e:
        logger.error(f"Brand deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                status="error",
                message="Brand deletion failed",
                error=str(e)
            ).model_dump()
        )

# ================ Document Endpoints ================
@app.post("/documents/upload/{brand_key}", 
          response_model=SuccessResponse,
          status_code=status.HTTP_201_CREATED)
async def upload_file(brand_key: str, file: UploadFile = File(...), overwrite: bool = False):
    try:
        if not brand_exists(brand_key):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    status="error",
                    message="Brand not found",
                    error=f"Brand '{brand_key}' doesn't exist"
                ).model_dump()
            )
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    status="error",
                    message="Invalid file type",
                    error=f"Allowed extensions: {', '.join(Config.ALLOWED_EXTENSIONS)}"
                ).model_dump()
            )
        
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=ErrorResponse(
                    status="error",
                    message="File too large",
                    error=f"Max size: {Config.MAX_FILE_SIZE} bytes"
                ).model_dump()
            )
        
        brand_dir = Config.get_brand_dir(brand_key)
        brand_dir.mkdir(exist_ok=True)
        file_path = brand_dir / file.filename
        
        if file_path.exists() and not overwrite:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=ErrorResponse(
                    status="error",
                    message="File exists",
                    error="Set overwrite=true to replace"
                ).model_dump()
            )
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Generate a unique ID for the document
        document_id = str(uuid.uuid4())

        file_info = {
            "id": document_id,
            "filename": file.filename,
            "path": str(file_path),
            "size": file_size,
            "extension": file_ext,
            "hash": get_file_hash(file_path),
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "status": "uploaded",
            "brand": brand_key,
            "metadata": {"brand": brand_key} 
        }

        update_document_list(file_info)
        update_brand_documents(brand_key, str(file_path))

        return SuccessResponse(
            status="success",
            message="File uploaded successfully",
            data=file_info
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                status="error",
                message="File upload failed",
                error=str(e)
            ).model_dump()
        )

@app.get("/documents/{brand_key}", response_model=SuccessResponse)
async def list_documents(brand_key: str):
    if not brand_exists(brand_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                status="error",
                message="Brand not found",
                error=f"Brand '{brand_key}' doesn't exist"
            ).model_dump()
        )
    try:
        # Read documents only for the brand
        documents = json.loads(Config.DOCUMENT_LIST.read_text())
        brand_docs = [doc for doc in documents if doc.get("brand") == brand_key]
        return SuccessResponse(status="success", data=brand_docs)
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                status="error",
                message="Failed to list documents",
                error=str(e)
            ).model_dump()
        )
    
# ================ Document Processing Endpoints ================
@app.post("/documents/process", status_code=status.HTTP_202_ACCEPTED)
async def process_document(
    background_tasks: BackgroundTasks,
    brand_key: str = Form(...),
    source_type: str = Form(...),
    source: str = Form(None),  # only for 'url'
    extract_faqs: bool = Form(...),
    model: str = Form(...),
    file: UploadFile = File(None)  # only for 'pdf'
):
    """
    Accepts either a URL or a PDF file, depending on source_type.
    """
    try:
        # Validate source_type
        if source_type not in ("pdf", "url"):
            raise HTTPException(status_code=400, detail="Invalid source_type. Must be 'pdf' or 'url'.")

        # Ensure brand exists
        if not brand_exists(brand_key):
            raise HTTPException(status_code=404, detail=f"Brand '{brand_key}' not found")

        # Generate processing ID
        processing_id = str(uuid.uuid4())
        brand_dir = Config.get_brand_dir(brand_key)
        brand_dir.mkdir(exist_ok=True)

        # Handle PDF upload
        if source_type == "pdf":
            if not file:
                raise HTTPException(status_code=400, detail="Missing PDF file")
            ext = Path(file.filename).suffix.lower()
            if ext != ".pdf":
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")

            temp_path = brand_dir / f"{processing_id}_{file.filename}"
            with open(temp_path, "wb") as f_out:
                f_out.write(await file.read())
            source_to_use = str(temp_path)

        # Handle URL input
        elif source_type == "url":
            if not source:
                raise HTTPException(status_code=400, detail="Missing source URL")
            source_to_use = source

        # Launch background task
        background_tasks.add_task(
            _process_document_task,
            processing_id=processing_id,
            brand_key=brand_key,
            source_type=source_type,
            source=source_to_use,
            extract_faqs=extract_faqs,
            model=model
        )

        return {
            "status": "processing",
            "message": "Document processing started",
            "data": {
                "processing_id": processing_id,
                "brand": brand_key,
                "status_url": f"/documents/status/{processing_id}"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/documents/status/{processing_id}", response_model=SuccessResponse)
async def get_processing_status(processing_id: str):
    """
    Check the status of a document processing task
    """
    # In a real implementation, you'd track this in a database or cache
    # For this example, we'll just return a mock response
    return SuccessResponse(
        status="success",
        data={
            "processing_id": processing_id,
            "status": "completed",  # or "processing", "failed"
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
    )

@app.get("/documents/{brand_key}/{document_id}", response_model=SuccessResponse)
async def get_document(brand_key: str, document_id: str):
    try:
        # Load document list
        with open(Config.DOCUMENT_LIST, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # Find the specific document
        doc = next((d for d in documents if d.get("id") == document_id and d.get("brand") == brand_key), None)
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found for brand {brand_key}"
            )

        # Debug logging
        logger.info(f"Processing document with path: {doc.get('path')}")
        if "path" in doc:
            doc_path = Path(doc["path"])
            logger.info(f"Document path details - exists: {doc_path.exists()}, is_file: {doc_path.is_file()}")
            
            # Check for FAQ file
            if "faq_file" in doc:
                faq_path = Path(doc["faq_file"])
                logger.info(f"Looking for FAQ file at: {faq_path}")
                logger.info(f"FAQ path exists: {faq_path.exists()}")
                
                if faq_path.exists():
                    logger.info("FAQ file found, attempting to read...")
                    try:
                        with open(faq_path, "r", encoding="utf-8") as f:
                            faq_data = json.load(f)
                            
                            # Handle both list and dict formats safely
                            if isinstance(faq_data, dict):
                                if "faqs" in faq_data:
                                    # New format with metadata
                                    doc["faqs"] = faq_data["faqs"]
                                    logger.info(f"Found {len(faq_data['faqs'])} FAQs in structured format")
                                else:
                                    # Unexpected dict format
                                    doc["faqs"] = [faq_data]
                                    logger.warning("Found single FAQ in dict format")
                            elif isinstance(faq_data, list):
                                # Direct list of FAQs
                                doc["faqs"] = faq_data
                                logger.info(f"Found {len(faq_data)} FAQs in list format")
                            else:
                                logger.error(f"Unexpected FAQ data format: {type(faq_data)}")
                                doc["faqs_error"] = "Unexpected FAQ data format"
                            
                            # Safe logging of FAQ data
                            if "faqs" in doc:
                                sample = doc["faqs"][0] if len(doc["faqs"]) > 0 else "Empty FAQ list"
                                logger.info(f"FAQ data sample: {json.dumps(sample, indent=2)[:500]}...")
                    except Exception as e:
                        logger.error(f"Error reading FAQ file: {str(e)}", exc_info=True)
                        doc["faqs_error"] = str(e)
            else:
                logger.info("No FAQ file reference found in document metadata")

        return SuccessResponse(
            status="success",
            data=doc
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}"
        )

@app.delete("/documents/{brand_key}/{document_id}", response_model=SuccessResponse)
async def delete_document(brand_key: str, document_id: str):
    """
    Delete a document and all its associated files with enhanced error handling
    """
    try:
        # 1. Verify document exists in registry
        documents = json.loads(Config.DOCUMENT_LIST.read_text())
        doc = next((d for d in documents if d.get("id") == document_id and d.get("brand") == brand_key), None)
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found for brand {brand_key}"
            )

        # 2. Prepare file paths to delete
        brand_dir = Config.get_brand_dir(brand_key)
        files_to_delete = set()
        
        # Add explicitly referenced files
        if "path" in doc:
            files_to_delete.add(Path(doc["path"]))
        
        # Add all files matching document ID pattern
        files_to_delete.update(brand_dir.glob(f"{document_id}*"))
        
        # Add known companion files (from your example)
        companion_files = [
            f"{document_id}_faqs.json",
            f"{document_id}_llm_raw.json",
            f"{document_id}_metadata.json",
            f"{document_id}_raw_text.txt"
        ]
        files_to_delete.update(brand_dir / f for f in companion_files)
        
        # 3. Delete files with verification
        deletion_report = []
        for file_path in files_to_delete:
            try:
                if file_path.exists():
                    file_path.unlink()
                    # Verify deletion
                    if file_path.exists():
                        raise RuntimeError(f"File still exists after deletion: {file_path}")
                    deletion_report.append({
                        "file": str(file_path),
                        "status": "deleted",
                        "size": file_path.stat().st_size if file_path.exists() else 0
                    })
                    logger.info(f"Successfully deleted {file_path}")
                else:
                    deletion_report.append({
                        "file": str(file_path),
                        "status": "not_found"
                    })
            except Exception as e:
                deletion_report.append({
                    "file": str(file_path),
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"Failed to delete {file_path}: {str(e)}")

        # 4. Update document registries
        # Remove from main document list
        updated_docs = [d for d in documents if not (d.get("id") == document_id and d.get("brand") == brand_key)]
        Config.DOCUMENT_LIST.write_text(json.dumps(updated_docs, indent=2))
        
        # Remove from brand-specific index
        remove_from_brand_documents(brand_key, document_id)

        # 5. Verify at least the main file was deleted
        main_file_deleted = any(
            report["status"] == "deleted" and str(doc.get("path", "")) in report["file"]
            for report in deletion_report
        )
        
        if not main_file_deleted and "path" in doc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Main document file {doc['path']} could not be deleted"
            )

        return SuccessResponse(
            status="success",
            message=f"Document cleanup completed for {document_id}",
            data={
                "document_id": document_id,
                "deletion_report": deletion_report,
                "files_deleted": sum(1 for r in deletion_report if r["status"] == "deleted"),
                "files_failed": sum(1 for r in deletion_report if r["status"] == "failed")
            }
        )
        
    except Exception as e:
        logger.error(f"Document deletion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ================ Query Endpoints ================
@app.post("/query", response_model=SuccessResponse)
async def query_endpoint(request: Request):
    try:
        processor = request.app.state.processor
        data = await request.json()
        
        if not (question := data.get("question")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    status="error",
                    message="Question required",
                    error="Missing 'question' field"
                ).model_dump()
            )
        
        start_time = time.time()
        thread_id = data.get("thread_id") or await processor.get_or_create_thread()
        user_id = data.get("user_id")
        brand_key = data.get("brand_key")  # Get brand_key from request

        result = await processor.process_query(
            question=question,
            thread_id=thread_id,
            user_id=user_id,
            brand_key=brand_key  # Pass brand_key to processor
        )

        if isinstance(result, dict) and result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorResponse(
                    status="error",
                    message="Query processing failed",
                    error=result.get("error"),
                    error_details=result.get("diagnostics")
                ).model_dump()
            )

        return SuccessResponse(
            status="success",
            data={
                "result": result if isinstance(result, str) else result.get("result"),
                "processing_time": round(time.time() - start_time, 3),
                "thread_id": thread_id,
                "brand": brand_key,  # Include brand_key in response
                "diagnostics": result.get("diagnostics") if isinstance(result, dict) else None
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                status="error",
                message="Query processing failed",
                error=str(e)
            ).model_dump()
        )
    
@app.post("/query/stream")
async def stream_query_endpoint(request: Request):
    data = await request.json()
    question = data.get("question", "")
    thread_id = data.get("thread_id")
    brand_key = data.get("brand_key", "default")

    async def generate():
        if not question:
            yield format_sse_message({
                'error': 'Question required',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            })
            return

        logger.info(f"[THREAD ID]: {thread_id}")

        try:
            processor = request.app.state.processor
            async for chunk in processor.stream_query(
                question=question,
                thread_id=thread_id,
                brand_key=brand_key
            ):
                # Handle the final metadata chunk
                if isinstance(chunk, str) and chunk.startswith('{"status":'):
                    try:
                        metadata = json.loads(chunk.strip())
                        # Flatten the metadata directly into the SSE message
                        yield format_sse_message({
                            'status': 'completed',
                            'processing_time': metadata.get('processing_time'),
                            'timestamp': metadata.get('timestamp'),
                            'thread_id': metadata.get('thread_id'),
                            'brand': metadata.get('brand'),
                            'documents_retrieved': metadata.get('documents_retrieved'),
                            'model': metadata.get('model'),
                            'retrieval_time': metadata.get('retrieval_time'),
                            'generation_time': metadata.get('generation_time')
                        })
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse metadata chunk: {chunk}")
                        continue
                # Handle normal content chunks
                elif isinstance(chunk, dict):
                    if 'thread_id' not in chunk and thread_id:
                        chunk['thread_id'] = thread_id

                    if chunk.get('status') == 'complete':
                        yield format_sse_message({
                            'status': 'completed',
                            'timestamp': chunk.get('timestamp', datetime.now().isoformat()),
                            'thread_id': chunk.get('thread_id', thread_id)
                        })
                    elif 'text' in chunk:
                        yield format_sse_message(chunk)
                else:
                    yield format_sse_message({
                        'text': str(chunk),
                        'status': 'in_progress',
                        'timestamp': datetime.now().isoformat(),
                        'thread_id': thread_id
                    })

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            yield format_sse_message({
                'error': 'Streaming failed',
                'message': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'thread_id': thread_id
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

def format_sse_message(data: dict) -> str:
    """Format data as Server-Sent Event"""
    return f"data: {json.dumps(data)}\n\n"



# ================ Vector Store Endpoints ================
@app.post("/vector_store/create", response_model=SuccessResponse)
async def create_vector_store(
    background_tasks: BackgroundTasks,
    request: VectorStoreOperationRequest
):
    """Create vector store with selected documents by file IDs"""
    logger.info(f"Request received - brand: {request.brand_key}, file_ids: {len(request.file_ids)}")

    try:
        from vector_store import VectorStoreManager
        from data_loader import DocumentLoader
        import datetime

        # Get all documents for the brand
        all_docs = json.loads(Config.DOCUMENT_LIST.read_text())
        brand_docs = [doc for doc in all_docs if doc.get("brand") == request.brand_key]

        # Filter documents by requested IDs if specified
        if request.file_ids:
            brand_docs = [
                doc for doc in brand_docs 
                if doc.get("id") in request.file_ids 
                or doc.get("hash") in request.file_ids
            ]

        if not brand_docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No matching documents found for brand '{request.brand_key}'"
            )

        # Load and process the selected documents
        processed_docs = []
        for doc_info in brand_docs:
            try:
                docs = DocumentLoader.process_document(doc_info)
                if docs:
                    processed_docs.extend(DocumentLoader.chunk_documents(docs))
            except Exception as e:
                logger.warning(f"Failed to process document {doc_info.get('path')}: {str(e)}")
                continue

        if not processed_docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No documents could be processed"
            )

        # Get current time for modification timestamp
        current_time = datetime.datetime.now().isoformat()
        
        background_tasks.add_task(
            _create_store_task,
            documents=processed_docs,
            brand_key=request.brand_key,
            recreate=request.recreate,
            modification_time=current_time  # Pass to the background task
        )

        return SuccessResponse(
            status="processing",
            message="Vector store creation started with selected documents",
            data={
                "brand": request.brand_key,
                "document_count": len(processed_docs),
                "file_ids": [doc.get("id") or doc.get("hash") for doc in brand_docs],
                "recreate": request.recreate,
                "last_modified": current_time  # Include in response
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/vector_store/status", response_model=SuccessResponse)
async def get_vector_store_status(brand_key: Optional[str] = None):
    try:
        from vector_store import VectorStoreManager
        from pathlib import Path
        import datetime
        
        config = VectorStoreManager._get_store_config(brand_key)
        exists = config.persist_path.exists()
        
        response = {
            "exists": exists,
            "brand": brand_key,
            "path": str(config.persist_path),
            "collection": config.collection_name,
            "last_modified": None  # Initialize as None
        }
        
        if exists:
            try:
                # Get last modified time from the file system
                modified_time = config.persist_path.stat().st_mtime
                response["last_modified"] = datetime.datetime.fromtimestamp(modified_time).isoformat()
                
                store = VectorStoreManager.get_vector_store(brand_key)
                if store and hasattr(store, '_collection'):
                    response["document_count"] = store._collection.count()
            except Exception as e:
                logger.warning(f"Couldn't get full store details: {str(e)}")
                
        return SuccessResponse(
            status="success",
            data=response
        )
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                status="error",
                message="Status check failed",
                error=str(e)
            ).model_dump()
        )

@app.delete("/vector_store/delete", response_model=SuccessResponse)
async def delete_vector_store(brand_key: Optional[str] = None):
    try:
        from vector_store import VectorStoreManager
        
        if not brand_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="brand_key is required"
            )
            
        success = await VectorStoreManager.clear_vector_store(brand_key)
        
        return SuccessResponse(
            status="success" if success else "error",
            message="Vector store deleted" if success else "Deletion failed",
            data={
                "brand": brand_key,
                "deleted": success
            }
        )
    except Exception as e:
        logger.error(f"Deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
@app.post("/vector_store/rebuild", response_model=SuccessResponse)
async def rebuild_vector_store(request: VectorStoreOperationRequest):
    """Force rebuild a vector store with selected documents"""
    try:
        from vector_store import VectorStoreManager
        from data_loader import DocumentLoader
        import datetime
        import os
        
        logger.info(f"Starting vector store rebuild for brand: {request.brand_key}")
        
        # Get current timestamp for modification time
        current_time = datetime.datetime.now().isoformat()
        
        # Load all documents directly from document_list.json
        with open(Config.DOCUMENT_LIST, 'r', encoding='utf-8') as f:
            all_docs = json.load(f)
        
        # Filter by brand and then by IDs if specified
        brand_docs = [doc for doc in all_docs if doc.get("brand") == request.brand_key]
        
        if request.file_ids:
            brand_docs = [
                doc for doc in brand_docs 
                if doc.get("id") in request.file_ids 
                or doc.get("hash") in request.file_ids
            ]

        if not brand_docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No matching documents found for brand '{request.brand_key}'"
            )

        # Force clean first with enhanced error handling
        logger.info("Cleaning up existing stores...")
        try:
            if request.recreate:
                logger.info("Force deleting existing vector store...")
                if not VectorStoreManager.clear_vector_store(request.brand_key):
                    raise RuntimeError("Failed to delete existing vector store")
            else:
                logger.info("Retaining existing store folder for overwrite (safe mode)")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Cleanup failed: {str(e)}"
            )

        # Process documents using the unified processor
        processed_docs = []
        for doc_info in brand_docs:
            try:
                docs = DocumentLoader.process_document(doc_info)
                if docs:
                    processed_docs.extend(DocumentLoader.chunk_documents(docs))
            except Exception as e:
                logger.warning(f"Failed to process document {doc_info.get('path')}: {str(e)}")
                continue

        if not processed_docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No documents could be processed"
            )

        logger.info(f"Loaded {len(processed_docs)} documents")
        
        # Recreate store with enhanced error handling
        logger.info("Creating vector store...")
        try:
            store = await VectorStoreManager.create_vector_store(
                documents=processed_docs,
                brand_key=request.brand_key, 
                recreate=True  # Force rebuild
            )
        except Exception as e:
            logger.error(f"Store creation failed: {str(e)}")
            # Check if this is a permission error
            if "readonly database" in str(e).lower():
                # Try to fix permissions and retry once
                try:
                    store_path = Config.get_brand_vector_store_path(request.brand_key)
                    os.chmod(store_path, 0o777)
                    logger.info(f"Retrying after setting permissions on {store_path}")
                    store = await VectorStoreManager.create_vector_store(
                        documents=processed_docs,
                        brand_key=request.brand_key,
                        recreate=True
                    )
                except Exception as retry_error:
                    logger.error(f"Retry failed: {str(retry_error)}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to create store after permission fix: {str(retry_error)}"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Store creation failed: {str(e)}"
                )
        
        # Explicitly set modification time
        try:
            config = VectorStoreManager._get_store_config(request.brand_key)
            if config.persist_path.exists():
                mod_time = datetime.datetime.fromisoformat(current_time).timestamp()
                os.utime(config.persist_path, (mod_time, mod_time))
                logger.info(f"Set rebuild modification time: {current_time}")
        except Exception as e:
            logger.warning(f"Couldn't set modification time: {str(e)}")

        store_info = {
            "document_count": len(processed_docs),
            "brand": request.brand_key,
            "file_ids": [doc.get("id") or doc.get("hash") for doc in brand_docs],
            "last_modified": current_time
        }
        
        try:
            store_info["path"] = str(store._persist_directory)
        except AttributeError:
            store_info["path"] = "unknown"
        
        return SuccessResponse(
            status="success",
            message=f"Rebuilt store for {request.brand_key}",
            data=store_info
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rebuild failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
@app.get("/vector_store/health")
async def vector_store_health(brand_key: str):  # Now requires brand_key
    from vector_store import VectorStoreManager
    store = VectorStoreManager.get_vector_store(brand_key)
    
    return {
        "exists": store is not None,
        "document_count": store._collection.count() if store else 0,
        "brand": brand_key
    }

# ================ Health Check ================
@app.get("/health", response_model=SuccessResponse)
async def get_health():
    try:
        health_info = await health_check()
        return SuccessResponse(
            status="success",
            data=health_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                status="error",
                message="Health check failed",
                error=str(e)
            ).model_dump()
        )

# ================ Helper Functions ================
async def _create_store_task(
    documents: List[Document], 
    brand_key: str = None, 
    recreate: bool = False,
    modification_time: Optional[str] = None
):
    """
    Background task to create/recreate a vector store with comprehensive error handling.
    
    Args:
        documents: List of documents to store
        brand_key: Brand identifier for the vector store
        recreate: Whether to force recreation of the store
        modification_time: Optional explicit timestamp to set as last modified
    """
    from vector_store import VectorStoreManager
    import datetime
    import os
    logger = logging.getLogger('vector_store_task')
    
    max_retries = 3
    retry_delay = 2  # seconds
    last_exception = None
    
    for attempt in range(1, max_retries + 1):
        try:
            # Clean up existing store if recreation requested
            if recreate and attempt == 1:
                logger.info(f"Cleaning existing store for brand '{brand_key}'")
                VectorStoreManager.clear_vector_store(brand_key)
            
            logger.info(f"Attempt {attempt}/{max_retries} to create vector store for '{brand_key}'")
            
            # Create the vector store
            store = await VectorStoreManager.create_vector_store(
                documents=documents,
                brand_key=brand_key,
                recreate=recreate
            )
            
            # Validate the store was created properly
            if not store or not hasattr(store, '_collection'):
                raise RuntimeError("Store creation returned invalid object")
            
            # Set modification time if provided
            if modification_time:
                try:
                    config = VectorStoreManager._get_store_config(brand_key)
                    if config.persist_path.exists():
                        mod_time = datetime.datetime.fromisoformat(modification_time).timestamp()
                        os.utime(config.persist_path, (mod_time, mod_time))
                        logger.info(f"Set explicit modification time: {modification_time}")
                except Exception as e:
                    logger.warning(f"Couldn't set modification time: {str(e)}")
            
            doc_count = store._collection.count()
            logger.info(f"Successfully created vector store for '{brand_key}' with {doc_count} documents")
            return
            
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt} failed: {str(e)}")
            
            # Special handling for database schema errors
            if "no such table: tenants" in str(e):
                logger.info("Detected database schema issue, performing cleanup...")
                VectorStoreManager.clear_vector_store(brand_key)
            
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
    
    # If we get here, all retries failed
    error_msg = f"Failed to create vector store for '{brand_key}' after {max_retries} attempts"
    logger.error(error_msg, exc_info=last_exception)
    raise RuntimeError(f"{error_msg}: {str(last_exception)}")

async def _process_document_task(
    processing_id: str,
    brand_key: str,
    source_type: str,
    source: str,
    extract_faqs: bool,
    model: str
):
    """Enhanced document processing with comprehensive error handling and debugging"""
    from utils.doc_extractor import (
        extract_text_from_pdf,
        extract_text_from_url,
        ask_ollama,
        clean_llm_output,
        validate_qa_structure,
        logger as doc_logger
    )
    
    temp_file = None
    doc_info = {
        "id": processing_id,
        "brand": brand_key,
        "source_type": source_type,
        "source": source,
        "status": "processing",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "extract_faqs": extract_faqs,
        "model": model,
        "debug": {}  # Container for debug information
    }
    created_files = []  # Track all created files

    def cleanup_files(force=False):
        """Remove created files with option to preserve for debugging"""
        if not force and doc_info.get("status") in ("failed", "partially_completed"):
            doc_logger.info("Preserving files for debugging")
            return
            
        for file_path in created_files:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    doc_logger.debug(f"Deleted file: {file_path}")
            except Exception as e:
                doc_logger.error(f"Failed to delete {file_path}: {str(e)}")
        
        # Update tracking files
        for file_path, update_fn in [
            (Config.DATA_DIR / "brand_documents.json", 
             lambda data: [p for p in data.get(brand_key, []) if str(processing_id) not in p]),
            (Config.DOCUMENT_LIST,
             lambda data: [d for d in data if d.get("id") != processing_id])
        ]:
            try:
                if file_path.exists():
                    with open(file_path, "r+") as f:
                        data = json.load(f)
                        updated = update_fn(data)
                        if file_path.name == "brand_documents.json":
                            data[brand_key] = updated
                        else:
                            data = updated
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()
            except Exception as e:
                doc_logger.error(f"Failed to update {file_path.name}: {str(e)}")

    def get_expected_faq_count(text_length: int) -> tuple:
        """Return min and max expected FAQs based on text length with logging"""
        if text_length < 800:
            return (2, 4)
        elif text_length < 2000:
            return (4, 8)
        elif text_length < 5000:
            return (6, 12)
        else:
            return (8, 15)

    try:
        brand_dir = Config.get_brand_dir(brand_key)
        brand_dir.mkdir(exist_ok=True)
        doc_logger.info(f"Processing {source_type.upper()}: {source}")
        
        # Handle both source types with enhanced error handling
        if source_type == "pdf":
            if source.startswith("http"):
                temp_file = Path(tempfile.mktemp(suffix=".pdf"))
                with open(temp_file, "wb") as f:
                    response = requests.get(source, timeout=30)
                    response.raise_for_status()
                    f.write(response.content)
                source_path = temp_file
            else:
                source_path = Path(source)
                if not source_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {source_path}")
            
            filename = f"{processing_id}_{source_path.name}"
            dest_path = brand_dir / filename
            dest_path.write_bytes(source_path.read_bytes())
            created_files.append(str(dest_path))
            
            doc_info.update({
                "filename": filename,
                "path": str(dest_path),
                "size": dest_path.stat().st_size,
                "extension": ".pdf",
                "hash": get_file_hash(dest_path)
            })
            
            text = extract_text_from_pdf(str(dest_path))
        else:  # URL processing
            filename = f"{processing_id}.txt"
            dest_path = brand_dir / filename
            text = extract_text_from_url(source)
            
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(text)
            created_files.append(str(dest_path))
            
            doc_info.update({
                "filename": filename,
                "path": str(dest_path),
                "size": dest_path.stat().st_size,
                "extension": ".txt",
                "hash": get_file_hash(dest_path),
                "original_url": source
            })
        
        if not text:
            raise RuntimeError(f"{source_type.upper()} text extraction returned empty content")
        
        doc_logger.info(f"Extracted {len(text)} characters")
        doc_info["text_length"] = len(text)
        min_expected, max_expected = get_expected_faq_count(len(text))
        # doc_logger.info(f"Expecting {min_expected}-{max_expected} FAQs from this content")
        doc_info["debug"]["expected_faqs"] = f"{min_expected}-{max_expected}"

        # Enhanced FAQ Processing with debugging capabilities
        if extract_faqs and text:
            try:
                doc_logger.info("Extracting FAQs...")
                
                # Save raw text for debugging
                raw_text_file = brand_dir / f"{processing_id}_raw_text.txt"
                with open(raw_text_file, "w", encoding="utf-8") as f:
                    f.write(text)
                created_files.append(str(raw_text_file))
                doc_info["debug"]["raw_text_file"] = str(raw_text_file)
                
                # Get LLM response
                result = ask_ollama(text, model)
                if not result:
                    raise RuntimeError("Empty response from LLM")
                
                # Save raw LLM output
                raw_output_file = brand_dir / f"{processing_id}_llm_raw.json"
                with open(raw_output_file, "w", encoding="utf-8") as f:
                    json.dump({"output": result}, f, indent=2)
                created_files.append(str(raw_output_file))
                doc_info["debug"]["llm_raw_output"] = str(raw_output_file)
                
                # Parse and validate
                parsed_data = clean_llm_output(result)
                validation_result = validate_qa_structure(parsed_data, len(text))
                
                # Save validation details
                doc_info.update({
                    "faq_validation": {
                        "status": validation_result.get("status"),
                        "errors": validation_result.get("errors"),
                        "warnings": validation_result.get("warnings"),
                        "expected": validation_result.get("expected_range"),
                        "received": len(validation_result.get("faqs", []))
                    }
                })
                
                # Process valid FAQs
                valid_faqs = validation_result.get("faqs", [])
                if valid_faqs:
                    faq_file = brand_dir / f"{processing_id}_faqs.json"
                    with open(faq_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "faqs": valid_faqs,
                            "metadata": {
                                "source": source,
                                "processing_id": processing_id,
                                "validation": validation_result
                            }
                        }, f, indent=2, ensure_ascii=False)
                    created_files.append(str(faq_file))
                    
                    doc_info.update({
                        "faqs": valid_faqs,
                        "faq_count": len(valid_faqs),
                        "faq_file": str(faq_file),
                        "validation_status": validation_result.get("status")
                    })
                
                # Handle validation results
                if not validation_result.get("is_acceptable", False):
                    if valid_faqs:
                        doc_info["status"] = "partially_completed"
                        doc_logger.warning(
                            f"Partial success: Got {len(valid_faqs)} FAQs "
                            f"(expected {min_expected}-{max_expected}). "
                            f"Errors: {len(validation_result.get('errors', []))}"
                        )
                    else:
                        raise RuntimeError(
                            "FAQ extraction failed completely. " +
                            "\n".join(validation_result.get("errors", ["Unknown error"]))[:500]
                        )
                        
            except Exception as e:
                doc_logger.error(f"FAQ processing error: {str(e)}", exc_info=True)
                doc_info["debug"]["faq_error"] = str(e)
                
                if "faqs" not in doc_info:
                    raise RuntimeError(f"FAQ extraction failed: {str(e)}")
                
                # If we have partial results, continue with warning
                doc_info["status"] = "partially_completed"
                doc_logger.warning(f"Continuing with partial results after error: {str(e)}")

        # Finalize status if not already set
        if doc_info.get("status") == "processing":
            doc_info["status"] = "completed"

        # Create comprehensive metadata file
        metadata_file = brand_dir / f"{processing_id}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(doc_info, f, indent=2, ensure_ascii=False)
        created_files.append(str(metadata_file))
        doc_info["metadata_file"] = str(metadata_file)

    except Exception as e:
        doc_logger.error(f"Document processing failed: {str(e)}", exc_info=True)
        doc_info.update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "preserved_files": [str(f) for f in created_files]
        })
        
        # Only cleanup if we have no valuable content at all
        if "faqs" not in doc_info and "text_length" not in doc_info:
            cleanup_files(force=True)
        raise

    finally:
        # Cleanup temporary files
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                doc_logger.warning(f"Failed to delete temp file: {str(e)}")
        
        # Finalize timing info
        doc_info["end_time"] = datetime.now(timezone.utc).isoformat()
        if "start_time" in doc_info:
            doc_info["processing_time"] = round((
                datetime.fromisoformat(doc_info["end_time"]) - 
                datetime.fromisoformat(doc_info["start_time"])
            ).total_seconds(), 2)

        # Update tracking if successful
        if doc_info.get("status") in ("completed", "partially_completed"):
            try:
                update_document_list(doc_info)
                update_brand_documents(brand_key, doc_info.get("path"))
            except Exception as e:
                doc_logger.error(f"Failed to update tracking: {str(e)}")
        
        doc_logger.info(
            f"Processing completed with status: {doc_info.get('status', 'unknown')}. "
            f"Duration: {doc_info.get('processing_time', 0)}s"
        )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} "
        f"- {process_time:.2f}s"
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=4001,
        log_level="info",
        # timeout_keep_alive=300
    )