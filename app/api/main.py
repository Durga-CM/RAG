from typing import List
import uuid
import logging
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from app.models.schemas import (
    QueryRequest, 
    QueryResponse, 
    HealthResponse, 
    ChatMessage, 
    SessionResponse, 
    MessageRequest
)
from app.models.models import ChatHistory
from app.pipelines.rag_pipeline import RAGPipeline
from app.pipelines.structured_ingestion_pipeline import StructuredIngestionPipeline
from app.db.database import get_db
from app.services.history_service import HistoryService
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Universal JSON RAG API",
    description="Production REST API for the Universal JSON RAG Pipeline. "
                "Supports invoice, medical, HR, insurance_policy and any custom document type.",
    version="3.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons — initialized once at startup
rag_pipeline        = RAGPipeline()
ingestion_pipeline  = StructuredIngestionPipeline()


@app.get("/")
async def root():
    """Root endpoint to verify the API is accessible."""
    return {"message": "Universal JSON RAG API with RESTful Session Management is running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns service health status."""
    return HealthResponse(status="healthy")


# --- SESSION MANAGEMENT ENDPOINTS ---

@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new conversational session and return a unique session_id."""
    new_id = f"chat_{uuid.uuid4().hex[:8]}"
    logger.info(f"Created new session: {new_id}")
    return SessionResponse(session_id=new_id)


@app.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def query_session(session_id: str, request: MessageRequest, db: Session = Depends(get_db)):
    """
    Submit a query within a specific session. History is automatically managed.
    """
    try:
        logger.info(f"Processing session query: {session_id}")
        
        # 1. Fetch History
        raw_history = HistoryService.get_history(db, session_id, limit=10)
        history_text = HistoryService.format_history_for_llm(raw_history)
        
        # 2. Get Answer from Pipeline
        result = rag_pipeline.answer(request.query, history_text=history_text)
        
        # 3. Handle result format
        if isinstance(result, dict):
            answer = result.get("answer", "")
        else:
            answer = str(result)
            result = {"answer": answer}

        # 4. Save History
        HistoryService.save_message(db, session_id, "user", request.query)
        HistoryService.save_message(db, session_id, "assistant", answer)

        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error in session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/history", response_model=List[ChatMessage])
async def get_session_history(session_id: str, db: Session = Depends(get_db)):
    """Retrieve the full chat history for a specific session."""
    history = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp.asc()).all()
    
    for msg in history:
        msg.timestamp = msg.timestamp.isoformat()
        
    return history


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str, db: Session = Depends(get_db)):
    """Delete all history for a specific session."""
    try:
        HistoryService.clear_history(db, session_id)
        return {"message": f"Session {session_id} history has been cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- LEGACY & MAINTENANCE ENDPOINTS ---

@app.post("/query", response_model=QueryResponse, tags=["Legacy"])
async def query_rag_legacy(request: QueryRequest, db: Session = Depends(get_db)):
    """Legacy endpoint for single queries with optional session_id."""
    # Internally maps to the new session logic if session_id is provided
    if request.session_id:
        return await query_session(request.session_id, MessageRequest(query=request.query), db)
    
    # Generic query without history
    result = rag_pipeline.answer(request.query)
    if not isinstance(result, dict):
        result = {"answer": str(result)}
    return QueryResponse(**result)


@app.post("/ingest", tags=["Maintenance"])
async def trigger_ingestion(force_rebuild: bool = False):
    """Trigger document re-ingestion from dataset.json."""
    try:
        logger.info(f"Triggering ingestion (force_rebuild={force_rebuild})")
        summary = ingestion_pipeline.run(force_rebuild=force_rebuild)
        return summary
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
