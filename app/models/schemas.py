from pydantic import BaseModel
from typing import List, Optional

class SessionResponse(BaseModel):
    session_id: str

class MessageRequest(BaseModel):
    query: str

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str

    class Config:
        from_attributes = True
