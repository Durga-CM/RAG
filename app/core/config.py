import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data_invoice"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "invoice_collection")

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
CLASSIFICATION_MODEL = os.getenv("LLM_MODEL", "llama3")

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Processing Configuration
VECTOR_SIZE = 768
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
