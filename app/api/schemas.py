from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    detected_doc_type: Optional[str] = None
    sources: Optional[List[str]] = None
     

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
