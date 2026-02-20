from app.services.embedding_service import EmbeddingService
from app.services.classification_service import ClassificationService
from app.services.vector_store_service import VectorStoreService
from app.services.retrieval_service import RetrievalService
from app.services.extraction_service import ExtractionService
from app.services.generation_service import GenerationService
from app.core.config import COLLECTION_NAME

class RAGPipeline:

    def __init__(self):
        self.embedding = EmbeddingService()
        self.classifier = ClassificationService()
        self.vector_store = VectorStoreService()
        self.retrieval = RetrievalService(self.embedding, self.vector_store)
        self.extractor = ExtractionService()
        self.generator = GenerationService()

    def answer(self, query):
        # STANDALONE STEP 0: Detect Document Type from Query
        detected_doc_type = self.classifier.classify_query(query)
        detected_filename = self.classifier.detect_filename_in_query(query)
        
        if detected_doc_type:
            print(f"🎯 Detected query type: {detected_doc_type}")
        if detected_filename:
            print(f"📄 Detected filename reference: {detected_filename}")

        # STANDALONE STEP 1-3: Handled by retrieval.retrieve
        # Fetch up to 3 files as per standalone default
        selected_files_meta = self.retrieval.retrieve(query, detected_doc_type, detected_filename, limit=3)

        if not selected_files_meta:
            return {
                "answer": "🤖 I couldn't find any relevant documents. Check if files are scanned/empty.",
                "sources": []
            }

        if len(selected_files_meta) == 1:
            print(f"🔍 Focused on file: {selected_files_meta[0]['file_name']} (Type: {selected_files_meta[0]['doc_type']})")
        else:
            print(f"🔍 Searching across {len(selected_files_meta)} relevant files:")
            for sf in selected_files_meta:
                print(f"   📄 {sf['file_name']} (Type: {sf['doc_type']}, Score: {sf['score']:.3f})")

        # STANDALONE STEP 4: Layer 2 Retrieval (Get Context from ALL Selected Files)
        all_contexts = []
        for sf in selected_files_meta:
            # Standalone uses client.scroll with limit 50 and sort by chunk_index
            chunks = self.vector_store.scroll_by_file(COLLECTION_NAME, sf['file_id'])
            # Sort: header first, then list chunks in original order (item_index)
            # Universal chunker uses 'item_index'; PDF pipeline uses 'chunk_index'
            chunks.sort(key=lambda x: (
                0 if x.payload.get('section') == 'header' else 1,
                x.payload.get('item_index', x.payload.get('chunk_index', 0))
            ))
            file_text = "\n".join([c.payload['text'] for c in chunks])
            
            all_contexts.append({
                'file_name': sf['file_name'],
                'content': file_text
            })

        # STANDALONE STEP 5: Hybrid Search (AI Extraction)
        combined_context = "\n\n".join([
            f"=== FILE: {ctx['file_name']} ===\n{ctx['content']}" 
            for ctx in all_contexts
        ])
        
        # Standalone limits context to 15000 in extract_metadata_llm call
        extracted_info = self.extractor.extract(combined_context, query)
        
        if extracted_info:
            print(f"✨ AI EXTRACTION: {extracted_info}")

        # STANDALONE STEP 6: LLM Generation
        answer_text = self.generator.generate(query, all_contexts, extracted_info)
        
        return {
            "answer": answer_text,
            "detected_doc_type": detected_doc_type,
            "sources": [ctx["file_name"] for ctx in all_contexts]
        }
