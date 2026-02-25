"""
CLI entry point for structured JSON ingestion.
All logic lives in app/pipelines/structured_ingestion_pipeline.py
"""
from app.pipelines.structured_ingestion_pipeline import StructuredIngestionPipeline

if __name__ == "__main__":
    pipeline = StructuredIngestionPipeline()
    pipeline.run(force_rebuild=False)
