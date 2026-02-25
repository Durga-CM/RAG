import json
import uuid
import ollama
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import (
    DATA_DIR, COLLECTION_NAME, EMBEDDING_MODEL,
    QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE
)

import os

# Dataset file lives inside DATA_DIR
DATASET_FILE = os.path.join(DATA_DIR, "dataset.json")


def recursive_semantic_serializer(data, parent_key=""):
    """
    Universally converts ANY JSON structure into a rich semantic text description.
    Works for any nesting depth — dicts, lists, scalars.
    """
    texts = []
    if isinstance(data, dict):
        for key, value in data.items():
            readable_key = key.replace("_", " ").title()
            full_context = f"{parent_key} {readable_key}" if parent_key else readable_key
            serialized_value = recursive_semantic_serializer(value, full_context)
            if serialized_value:
                texts.append(serialized_value)
    elif isinstance(data, list):
        for item in data:
            serialized_item = recursive_semantic_serializer(item, parent_key)
            if serialized_item:
                texts.append(f"- {serialized_item}")
    else:
        if data is not None and str(data).strip():
            return f"{parent_key}: {data}"
    return "\n".join(texts)


def chunk_document(doc):
    """
    Universally splits ANY JSON document into logical chunks for precise retrieval.
    - NO hardcoded field names or doc-type-specific logic.
    - Header chunk  : all top-level scalar + dict fields (auto-serialized).
    - List chunks   : each element inside any top-level list field becomes its
                      own named chunk (works for invoice items, HR sections,
                      medical prescriptions, or any future list fields).
    - Fallback      : if a doc has no lists and no header, serialize the whole doc.
    Adding a new document type requires ZERO changes here.
    """
    chunks = []
    doc_id   = doc.get("doc_id", str(uuid.uuid4()))
    doc_type = doc.get("doc_type", "general")

    base_metadata = {
        "doc_id":      doc_id,
        "file_id":     doc_id,
        "file_name":   f"{doc_id}.json",
        "doc_type":    doc_type,
        "source_file": f"{doc_id}.json",
        "raw_data":    doc,
    }

    # ── STEP 1: HEADER CHUNK (all non-list top-level fields) ─────────────────
    header_parts = []
    for key, value in doc.items():
        if isinstance(value, list):
            continue
        readable_key = key.replace("_", " ").title()
        serialized   = recursive_semantic_serializer(value, readable_key)
        if serialized:
            header_parts.append(serialized)

    if header_parts:
        chunks.append({
            "text":     "\n".join(header_parts),
            "metadata": {**base_metadata, "section": "header"},
        })

    # ── STEP 2: LIST ITEM CHUNKS (one chunk per element per list field) ───────
    for key, value in doc.items():
        if not isinstance(value, list) or not value:
            continue
        section_label = key.replace("_", " ").title()
        for idx, item in enumerate(value, start=1):
            item_text = recursive_semantic_serializer(item, section_label)
            if item_text:
                chunks.append({
                    "text":     item_text,
                    "metadata": {**base_metadata,
                                 "section":    key,
                                 "item_index": idx,
                                 "item_data":  item},
                })

    # ── STEP 3: FALLBACK (nothing produced above) ─────────────────────────────
    if not chunks:
        full_text = recursive_semantic_serializer(doc)
        chunks.append({
            "text":     full_text,
            "metadata": {**base_metadata, "section": "full_doc"},
        })

    return chunks


class StructuredIngestionPipeline:
    """
    Ingests all documents from dataset.json into Qdrant using the
    universal chunk_document strategy. Called from:
      - FastAPI  POST /ingest
      - CLI      python ingest_structured.py
    """

    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def run(self, force_rebuild: bool = False) -> dict:
        """
        Load dataset.json, chunk every document universally, embed and upsert.
        Supports incremental ingestion by skipping documents already in the vector store.
        """
        print(f"🚀 Loading dataset from {DATASET_FILE}...")

        if not os.path.exists(DATASET_FILE):
            return {"total_docs": 0, "total_chunks": 0, "message": f"Dataset file not found at {DATASET_FILE}"}

        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reset or ensure collection
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)

        if exists and force_rebuild:
            print(f"⚠️  Recreating collection '{COLLECTION_NAME}'...")
            self.client.delete_collection(COLLECTION_NAME)
            exists = False

        if not exists:
            print(f"🆕 Creating collection '{COLLECTION_NAME}'...")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )

        points = []
        total_chunks = 0
        new_docs_processed = 0

        for doc in data:
            doc_id = doc.get("doc_id", str(uuid.uuid4()))
            
            # Incremental Check: Skip if doc_id already exists in Qdrant
            if not force_rebuild:
                check_res = self.client.count(
                    collection_name=COLLECTION_NAME,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id),
                            )
                        ]
                    )
                )
                if check_res.count > 0:
                    # Document already processed — skip
                    continue

            print(f"   🔹 Processing: {doc_id} ({doc.get('doc_type', 'general')})")
            chunks = chunk_document(doc)

            for chunk in chunks:
                emb = ollama.embeddings(
                    model=EMBEDDING_MODEL,
                    prompt=f"search_document: {chunk['text']}"
                )["embedding"]

                payload = chunk["metadata"]
                payload["text"] = chunk["text"]

                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload=payload
                ))
                total_chunks += 1
            
            new_docs_processed += 1

        if points:
            # Batch upsert all new points
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            msg = f"Successfully indexed {total_chunks} chunks from {new_docs_processed} new documents."
        else:
            msg = "No new documents to index."

        print(f"✅ {msg}")
        return {
            "total_docs":   len(data),
            "new_docs":     new_docs_processed,
            "total_chunks": total_chunks,
            "message":      msg
        }
