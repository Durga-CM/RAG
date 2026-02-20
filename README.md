# Universal JSON RAG Pipeline

A modular, production-ready **Retrieval-Augmented Generation (RAG)** system that can ingest and query **any structured JSON document** without requiring code changes per document type.

Supports invoices, medical reports, HR policies, insurance policies — and any new document type you add to the dataset.

---

## 🚀 Key Features

- **Universal JSON Ingestion** — Any JSON document is automatically chunked and embedded without hardcoded field names. Header fields and list items are handled generically.
- **Automatic Document Classification** — LLM-based classifier detects document type (`invoice`, `medical`, `hr`, `insurance_policy`) from content, not filenames.
- **Strict Type Filtering** — Retrieval is scoped to the relevant document type to prevent context mixing between domains.
- **Hybrid Extraction Mode** — Detects specific data requests (IDs, amounts, names) and uses a high-precision extraction pass before final generation.
- **Cross-domain Query Support** — Queries that span multiple document types are handled by searching across relevant collections.
- **FastAPI REST Interface** — Full REST API with Swagger UI for querying the RAG system programmatically.
- **Incremental Ingestion** — Only processes new or modified files; skips already-indexed documents.
- **UTF-8 Safe** — All file I/O uses explicit UTF-8 encoding to handle special characters correctly.

---

## 🛠 Prerequisites

1. **Ollama** — Install [Ollama](https://ollama.ai/) and pull the required models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. **Qdrant** — Run the Qdrant vector database via Docker:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Python 3.9+**

---

## 📥 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag_llm
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your model names and Qdrant connection details
   ```

---

## 📂 Project Structure

```
rag_llm/
├── run_api.py                # 🚀 FastAPI server entry point (production)
├── ingest_structured.py      # 🔧 CLI wrapper for one-time dataset ingestion
├── requirements.txt
├── .env.example
│
├── data_invoice/
│   └── dataset.json          # All documents in one JSON array (any doc types)
│
└── app/
    ├── api/
    │   ├── main.py           # FastAPI routes: /query  /ingest  /health
    │   └── schemas.py        # Pydantic request/response models
    │
    ├── pipelines/
    │   ├── rag_pipeline.py                  # End-to-end RAG orchestration
    │   └── structured_ingestion_pipeline.py # Universal JSON ingestion (used by API + CLI)
    │
    ├── services/
    │   ├── classification_service.py  # LLM-based doc type & query classifier
    │   ├── retrieval_service.py       # Vector search + score-based reranking
    │   ├── extraction_service.py      # High-precision field extraction
    │   ├── generation_service.py      # Final LLM answer generation
    │   ├── embedding_service.py       # Text embedding via Ollama
    │   └── vector_store_service.py    # Qdrant client wrapper
    │
    └── core/
        └── config.py         # Centralized config & environment variable loading
```

---

## 🏃 Running the Project

### Option 1 — Interactive CLI

```bash
# Step 1: Ingest your dataset (run once, or when dataset.json changes)
python ingest_structured.py

# Step 2: Start the chat interface
python main.py
```

### Option 2 — FastAPI REST Server

```bash
# Step 1: Ingest your dataset
python ingest_structured.py

# Step 2: Start the API server
python run_api.py
```

API will be available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 📡 API Endpoints

| Method | Endpoint  | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/health` | Health check                         |
| POST   | `/query`  | Submit a query to the RAG pipeline   |
| POST   | `/ingest` | Trigger document re-ingestion        |

### Example Query Request

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total premium payable in the insurance?"}'
```

### Example Response

```json
{
  "answer": "The total premium payable is ₹1,999.",
  "detected_doc_type": "insurance_policy",
  "sources": ["insurance_001.json"]
}
```

---

## 📄 Supported Document Types

| Type               | Example Queries                                              |
|--------------------|--------------------------------------------------------------|
| `invoice`          | "What is the grand total?", "List all items purchased"       |
| `medical`          | "What is the diagnosis for Ravi Kumar?", "List prescriptions"|
| `hr`               | "How many sick leave days in 2026?"                          |
| `insurance_policy` | "When does the policy expire?", "What is the IDV?"           |

> ➕ **Adding a new document type** requires only adding entries to `dataset.json` — no code changes needed.

---

## 🗂 Adding New Documents

Add any JSON document to `data_invoice/dataset.json` with a unique `doc_id` and `doc_type` field:

```json
{
  "doc_id": "legal_001",
  "doc_type": "legal",
  "case_number": "CAS-2026-001",
  "parties": { "plaintiff": "ABC Corp", "defendant": "XYZ Ltd" },
  "clauses": [
    { "title": "Non-Disclosure", "text": "..." }
  ]
}
```

Then re-run ingestion:
```bash
python ingest_structured.py
```

The universal chunker auto-discovers all fields — no hardcoding needed.
