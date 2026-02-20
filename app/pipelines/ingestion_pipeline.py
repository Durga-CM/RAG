import os
import uuid
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models

from app.core.config import *
from app.services.embedding_service import EmbeddingService
from app.services.classification_service import ClassificationService
from app.services.vector_store_service import VectorStoreService

class IngestionPipeline:

    def __init__(self):
        self.embedding = EmbeddingService()
        self.classifier = ClassificationService()
        self.vector_store = VectorStoreService()

    def run(self, force_rebuild=False):
        if force_rebuild:
            self.vector_store.create_collection(COLLECTION_NAME)
        else:
            self.vector_store.ensure_collection(COLLECTION_NAME)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        print(f"📂 Scanning {DATA_DIR} for new documents...")

        new_files_count = 0
        for pdf_file in pdf_files:
            path = os.path.join(DATA_DIR, pdf_file)
            file_name_clean = os.path.splitext(pdf_file)[0]
            
            # 🔥 INCREMENTAL CHECK: Skip if file already exists in Vector DB
            if not force_rebuild and self.vector_store.file_exists(COLLECTION_NAME, file_name_clean):
                print(f"   ⏭️ Skipping '{file_name_clean}' (already indexed)")
                continue

            file_id = str(uuid.uuid4())
            new_files_count += 1
            
            print(f"\n📄 Processing new file: {file_name_clean}...")
            
            doc = fitz.open(path)
            first_page = doc[0].get_text() if len(doc) > 0 else ""
            
            print(f"   🔍 Analyzing content to detect document type...")
            doc_type = self.classifier.classify_document(first_page)
            print(f"   ✅ Detected type: {doc_type}")

            points = []

            for page_idx, page in enumerate(doc):
                text = page.get_text()
                if not text.strip(): continue
                
                chunks = splitter.split_text(text)
                embeddings = self.embedding.embed_documents(chunks)

                for idx, emb in enumerate(embeddings):
                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={
                            "text": chunks[idx],
                            "file_id": file_id,
                            "file_name": file_name_clean,
                            "doc_type": doc_type,
                            "page": page_idx + 1,
                            "chunk_index": idx
                        }
                    ))

            if points:
                self.vector_store.upsert(COLLECTION_NAME, points)
                print(f"   ✅ Indexed {len(points)} chunks for '{file_name_clean}'")

        if new_files_count == 0:
            print("\n✨ All documents are up to date. No new embeddings generated.")
        else:
            print(f"\n✨ Successfully indexed {new_files_count} new documents.")



