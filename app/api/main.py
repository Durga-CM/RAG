from fastapi import FastAPI, HTTPException
from app.api.schemas import QueryRequest, QueryResponse, HealthResponse
from app.pipelines.rag_pipeline import RAGPipeline
from app.pipelines.structured_ingestion_pipeline import StructuredIngestionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Universal JSON RAG API",
    description="Production REST API for the Universal JSON RAG Pipeline. "
                "Supports invoice, medical, HR, insurance_policy and any custom document type.",
    version="2.0.0"
)

# Singletons — initialized once at startup
rag_pipeline        = RAGPipeline()
ingestion_pipeline  = StructuredIngestionPipeline()


@app.get("/")
async def root():
    """Root endpoint to verify the API is accessible."""
    return {"message": "Universal JSON RAG API is running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns service health status."""
    return HealthResponse(status="healthy")


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Submit a natural language query against all indexed documents.
    The pipeline automatically detects the document type and retrieves relevant context.
    """
    try:
        logger.info(f"Processing query: {request.query}")
        result = rag_pipeline.answer(request.query)
        if isinstance(result, dict):
            return QueryResponse(**result)
        return QueryResponse(answer=result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def trigger_ingestion(force_rebuild: bool = False):
    """
    Ingests documents from dataset.json into the vector store.
    - Set force_rebuild=true to delete the collection and start from scratch.
    - Set force_rebuild=false (default) to skip already indexed documents.
    """
    try:
        logger.info(f"Triggering structured ingestion (force_rebuild={force_rebuild})")
        summary = ingestion_pipeline.run(force_rebuild=force_rebuild)
        return summary
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
