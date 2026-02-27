from typing import List
import uuid
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.models.schemas import (
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



# --- SHARED STATE FOR SESSION CONTROL ---
active_stops = {}  # session_id -> asyncio.Event


@app.post("/sessions/{session_id}/stream")
async def stream_session(session_id: str, request: MessageRequest, db: Session = Depends(get_db)):
    """
    Streamed version of the query endpoint. 
    Supports interruption via client disconnect OR explicit /stop API.
    """
    import asyncio
    
    try:
        logger.info(f"Processing session query (STREAM): {session_id}")
        
        # Initialize stop trigger for this session
        stop_event = asyncio.Event()
        active_stops[session_id] = stop_event

        # 1. Fetch History
        raw_history = HistoryService.get_history(db, session_id, limit=10)
        history_text = HistoryService.format_history_for_llm(raw_history)
        
        # 2. Generator function
        async def event_generator():
            full_answer = ""
            # Save User message first
            HistoryService.save_message(db, session_id, "user", request.query)
            
            try:
                # Iterate through the pipeline stream
                for token in rag_pipeline.answer_stream(request.query, history_text=history_text):
                    # CHECK FOR STOP SIGNAL
                    if stop_event.is_set():
                        logger.warning(f"Stop signal received for session {session_id}")
                        yield "\n[STOPPED BY USER]"
                        break
                        
                    full_answer += token
                    yield token
                    # Tiny sleep to allow other tasks (like /stop) to run
                    await asyncio.sleep(0.01)
                    
            except Exception as stream_err:
                logger.error(f"Stream error: {str(stream_err)}")
                yield f"\n[STREAM_ERROR]: {str(stream_err)}"
            finally:
                # CLEANUP
                if session_id in active_stops:
                    del active_stops[session_id]
                
                # Save Assistant message once finished or interrupted
                if full_answer.strip():
                    HistoryService.save_message(db, session_id, "assistant", full_answer)
                    logger.info(f"Stream for {session_id} finalized. Saved history.")

        return StreamingResponse(event_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in stream session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/stop")
async def stop_session_generation(session_id: str):
    """
    Explicitly stop a running generation for a specific session.
    """
    if session_id in active_stops:
        active_stops[session_id].set()
        logger.info(f"Stop signal triggered for session: {session_id}")
        return {"message": "Stop signal sent successfully."}
    else:
        return {"message": "No active generation found for this session."}


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


# --- MAINTENANCE ENDPOINTS ---

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
