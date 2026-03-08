"""
Main FastAPI Application
Intelligent PDF Question Answering System
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import uuid
import logging

from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Question Answering System",
    description="AI-powered system to ask questions about your PDF documents",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_processor = PDFProcessor(chunk_size=500, chunk_overlap=100)
vector_store = VectorStore(persist_directory="./chroma_db")
llm_service = LLMService()

# Store document metadata (in production, use a database)
documents_store = {}

# Initialize vector store on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up PDF Q&A System...")
    try:
        vector_store.initialize()
        vector_store.create_collection("pdf_documents")
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize vector store: {e}")

# Request models
class QuestionRequest(BaseModel):
    document_id: str
    question: str
    n_results: Optional[int] = 5

class QuestionResponse(BaseModel):
    answer: str
    sources: List[dict]
    method: str
    document_id: str
    question: str

# Routes
@app.get("/")
async def root():
    return {
        "message": "PDF Question Answering System API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "ask": "/api/ask",
            "health": "/api/health",
            "documents": "/api/documents"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ollama_available": llm_service.check_ollama_available(),
        "vector_store": "initialized" if vector_store.collection else "not_initialized"
    }

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        logger.info(f"Processing PDF: {file.filename} ({len(content)} bytes)")
        
        # Process PDF
        processed_data = pdf_processor.process_pdf(content)
        
        # Add chunks to vector store
        vector_store.add_chunks(processed_data['chunks'], document_id)
        
        # Store document metadata
        documents_store[document_id] = {
            "id": document_id,
            "filename": file.filename,
            "num_chunks": processed_data['num_chunks'],
            "total_characters": processed_data['total_characters']
        }
        
        logger.info(f"Successfully processed document: {document_id}")
        
        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "num_chunks": processed_data['num_chunks'],
            "total_characters": processed_data['total_characters'],
            "message": "PDF uploaded and processed successfully. You can now ask questions!"
        }
        
    except ValueError as e:
        logger.error(f"PDF processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about an uploaded PDF document"""
    
    # Validate request
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if request.document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found. Please upload a PDF first.")
    
    try:
        logger.info(f"Question for document {request.document_id}: {request.question}")
        
        # Perform similarity search
        relevant_chunks = vector_store.similarity_search(
            query=request.question,
            n_results=request.n_results,
            document_id=request.document_id
        )
        
        # Generate answer using LLM
        result = llm_service.generate_answer(request.question, relevant_chunks)
        
        return {
            "success": True,
            "question": request.question,
            "document_id": request.document_id,
            **result
        }
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents"""
    documents = list(documents_store.values())
    return {
        "documents": documents,
        "count": len(documents)
    }

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings"""
    
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete from vector store
        vector_store.delete_document(document_id)
        
        # Remove from documents store
        del documents_store[document_id]
        
        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

