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



    def answer_stream(self, query, history_text=None):
        """
        Streaming version of the RAG pipeline.
        Hides the 'Source' line and appends metadata at the bottom.
        """
        # NEW: GREETING FILTER FOR STREAM
        is_greeting = query.lower().strip().strip('?!.') in ["hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"]
        
        if is_greeting:
            for token in self.generator.generate_stream(query, [], history_text=history_text):
                yield token
            return

        # STEP 1: Classification
        detected_doc_type = self.classifier.classify_query(query)
        detected_filename = self.classifier.detect_filename_in_query(query)
        
        # STEP 2: Retrieval
        selected_files_meta = self.retrieval.retrieve(query, detected_doc_type, detected_filename, limit=3)
        selected_files_meta.sort(key=lambda x: x['file_name'])

        if not selected_files_meta and detected_doc_type != "general":
            yield "🤖 I couldn't find any relevant documents."
            return

        all_contexts = []
        source_filenames = []
        for sf in selected_files_meta:
            source_filenames.append(sf['file_name'])
            chunks = self.vector_store.scroll_by_file(COLLECTION_NAME, sf['file_id'])
            chunks.sort(key=lambda x: (
                0 if x.payload.get('section') == 'header' else 1,
                x.payload.get('item_index', x.payload.get('chunk_index', 0))
            ))
            all_contexts.append({
                'file_name': sf['file_name'],
                'doc_type': sf['doc_type'],
                'content': "\n".join([c.payload['text'] for c in chunks])
            })

        # STEP 3: Simple Streaming
        for token in self.generator.generate_stream(query, all_contexts, history_text=history_text):
            yield token
