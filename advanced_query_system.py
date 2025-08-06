from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import json
import asyncio
import os
import logging
import time
import hashlib
import random
from datetime import datetime

# Import system components
from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from domain_handlers import get_domain_handler
from explainability import ExplanationEngine
from token_optimizer import TokenOptimizer
from llm_manager import LLMManager
from utils import setup_logger

# Setup logging
logger = setup_logger()

app = FastAPI(
    title="SuperDocAI: Intelligent Document Analysis System",
    version="1.0.0",
    description="Advanced document intelligence platform for semantic analysis across multiple domains"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    domain: Optional[str] = None
    explanation_level: Optional[str] = "standard"  # none, basic, standard, detailed
    output_format: Optional[str] = "json"  # json, markdown, html


class AnalysisResponse(BaseModel):
    answer: str
    confidence: float
    supporting_clauses: List[Dict[str, Any]]
    reasoning: str
    metadata: Dict[str, Any]


class DocumentUploadResponse(BaseModel):
    document_id: str
    document_type: str
    page_count: int
    token_count: int
    domain: str
    status: str
    processing_time: float


# Initialize system components
document_processor = DocumentProcessor()
embedding_engine = EmbeddingEngine()
explanation_engine = ExplanationEngine()
token_optimizer = TokenOptimizer()
llm_manager = LLMManager()


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    await embedding_engine.initialize()
    await llm_manager.initialize()
    logger.info("SuperDocAI system initialized and ready")


@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    domain: str = Form(None),
    api_key: Optional[str] = Header(None)
):
    """Upload and process a new document"""
    start_time = time.time()
    
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required")
    
    # Determine document type from extension
    file_extension = file.filename.split('.')[-1].lower()
    
    try:
        # Process document based on type
        document_content, metadata = await document_processor.process_document(file, file_extension)
        
        # Auto-detect domain if not specified
        if not domain:
            domain = await document_processor.detect_domain(document_content)
            
        # Generate document ID
        document_id = hashlib.md5(f"{file.filename}_{time.time()}".encode()).hexdigest()
        
        # Index document for vector search
        token_count = await embedding_engine.index_document(document_id, document_content, domain)
        
        # Get domain-specific handler for additional processing
        domain_handler = get_domain_handler(domain)
        await domain_handler.process_document(document_id, document_content, metadata)
        
        processing_time = time.time() - start_time
        
        return {
            "document_id": document_id,
            "document_type": file_extension,
            "page_count": metadata.get("page_count", 0),
            "token_count": token_count,
            "domain": domain,
            "status": "processed",
            "processing_time": round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.post("/api/v1/query", response_model=AnalysisResponse)
async def query_document(
    request: QueryRequest,
    api_key: Optional[str] = Header(None)
):
    """Query a document with natural language"""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required")
    
    try:
        start_time = time.time()
        
        # Get domain handler
        domain = request.domain
        if not domain and request.document_id:
            domain = await embedding_engine.get_document_domain(request.document_id)
        domain_handler = get_domain_handler(domain or "general")
        
        # Optimize the query for token efficiency
        optimized_query = await token_optimizer.optimize_query(request.query, domain)
        
        # Retrieve relevant document sections
        if request.document_id:
            # Semantic search in specific document
            relevant_sections = await embedding_engine.semantic_search(
                optimized_query, 
                document_id=request.document_id,
                top_k=5
            )
        else:
            # Cross-document semantic search
            relevant_sections = await embedding_engine.semantic_search(
                optimized_query,
                domain=domain,
                top_k=5
            )
        
        # Apply domain-specific processing
        processed_sections = await domain_handler.process_query_context(
            optimized_query, 
            relevant_sections
        )
        
        # Generate optimized prompt
        prompt, prompt_tokens = await token_optimizer.generate_optimized_prompt(
            optimized_query, 
            processed_sections,
            domain
        )
        
        # Call LLM for analysis
        llm_response = await llm_manager.analyze_query(prompt, domain)
        
        # Generate explanation
        explanation = await explanation_engine.generate_explanation(
            request.query,
            llm_response,
            processed_sections,
            level=request.explanation_level
        )
        
        # Format final response
        response = {
            "answer": llm_response.get("answer"),
            "confidence": llm_response.get("confidence", 0.7),
            "supporting_clauses": llm_response.get("supporting_clauses", []),
            "reasoning": explanation,
            "metadata": {
                "processing_time": round(time.time() - start_time, 2),
                "tokens_used": llm_response.get("tokens_used", 0),
                "prompt_tokens": prompt_tokens,
                "domain": domain,
                "relevant_sections_count": len(relevant_sections),
                "query_optimization": token_optimizer.get_optimization_stats()
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/api/v1/batch-query")
async def batch_query(
    queries: List[QueryRequest],
    api_key: Optional[str] = Header(None),
    max_concurrency: Optional[int] = 3
):
    """Process multiple queries with controlled concurrency"""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required")
    
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_single_query(query_request):
        async with semaphore:
            try:
                return await query_document(query_request, api_key=api_key)
            except Exception as e:
                return {"error": str(e)}
    
    # Process all queries with controlled concurrency
    tasks = [process_single_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    return {"results": results}


@app.get("/api/v1/system/status")
async def system_status():
    """Get system status and statistics"""
    return {
        "status": "operational",
        "version": app.version,
        "embedding_engine": await embedding_engine.get_status(),
        "llm_manager": await llm_manager.get_status(),
        "document_count": await document_processor.get_document_count(),
        "domains_available": list(await document_processor.get_available_domains()),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting SuperDocAI: Intelligent Document Analysis System")
    uvicorn.run(app, host="0.0.0.0", port=8000)