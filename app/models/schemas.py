from pydantic import BaseModel
from typing import List, Optional

class SessionResponse(BaseModel):
    session_id: str

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class MessageRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    detected_doc_type: Optional[str] = None
    sources: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str

    class Config:
        from_attributes = True
