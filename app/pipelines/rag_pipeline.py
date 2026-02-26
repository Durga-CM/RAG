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

    def answer(self, query, history_text=None):
        # NEW: GREETING FILTER
        # Detect if the query is just a simple greeting to avoid pulling irrelevant documents.
        is_greeting = query.lower().strip().strip('?!.') in ["hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"]
        
        if is_greeting:
            print(f"👋 Detected greeting: {query}")
            answer_text = self.generator.generate(query, [], history_text=history_text)
            return {
                "answer": answer_text,
                "detected_doc_type": "general",
                "sources": []
            }

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

        # LOGICAL ORDERING: Sort files alphabetically so "ID 1" is always 
        # the first logical file (e.g., invoice_001). This prevents Source mismatches.
        selected_files_meta.sort(key=lambda x: x['file_name'])

        if not selected_files_meta:
            if detected_doc_type == "general":
                # For general queries, we allow zero documents
                pass
            else:
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
                'doc_type': sf['doc_type'],
                'content': file_text
            })

        # STANDALONE STEP 5: AI Extraction and Generation (CONSOLIDATED)
        answer_text = self.generator.generate(query, all_contexts, history_text=history_text)
        
        # --- SOURCE ALIGNMENT LOGIC ---
        all_retrieved_names = [ctx["file_name"] for ctx in all_contexts]
        aligned_sources = []
        clean_answer = answer_text
        
        # 1. Detect multiple Sources from LLM Response (for aggregate queries)
        import re
        source_pattern = r"(?i)Source:\s*([\d\s,]+)"
        source_match = re.search(source_pattern, answer_text)
        
        if source_match:
            try:
                # Handle comma-separated list like "Source: 1, 2"
                indices = [int(i.strip()) for i in source_match.group(1).replace(",", " ").split()]
                for idx_val in indices:
                    doc_idx = idx_val - 1
                    if 0 <= doc_idx < len(all_retrieved_names):
                        primary_source = all_retrieved_names[doc_idx]
                        if primary_source not in aligned_sources:
                            aligned_sources.append(primary_source)
            except Exception:
                pass
            
            # 2. STRIP the Source line from the final answer text
            clean_answer = re.sub(source_pattern, "", answer_text).strip()

        # --- NEW AGGRESSIVE CLEANUP ---
        # 1. Strip residual "Document N", "Ref N", "Reference N" in parentheses or brackets
        clean_answer = re.sub(r'(?i)\(?(Document|Ref|Reference|Internal_ID)\s*\d+\)?', '', clean_answer)
        clean_answer = re.sub(r'(?i)\[?(Document|Ref|Reference|Internal_ID)\s*\d+\]?', '', clean_answer)

        # 2. Strip any known technical filenames from the conversational text.
        for name in all_retrieved_names:
            # Strip full filename (e.g., insurance_001.json)
            clean_answer = clean_answer.replace(name, "").strip()
            # Strip ID-only name (e.g., insurance_001)
            id_name = name.replace(".json", "")
            if id_name in clean_answer:
                # We use a regex to ensure we only strip it if it's a standalone "tag" at the end
                clean_answer = re.sub(rf'\n*\s*{id_name}\s*$', '', clean_answer).strip()
        
        # Final cleanup factor
        clean_answer = clean_answer.replace("()", "").replace("[]", "").strip()
        
        # FALLBACK: If no tag found, use relevant source
        if not aligned_sources:
            if detected_doc_type == "general":
                aligned_sources = ["general"]
            elif all_retrieved_names:
                aligned_sources = [all_retrieved_names[0]]
            else:
                aligned_sources = []

        return {
            "answer": clean_answer,
            "detected_doc_type": detected_doc_type,
            "sources": aligned_sources  # Relevant file remains here
        }

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

        import re
        import json

        # STEP 3: Precise Streaming with Sliding Window Buffer
        full_answer = ""
        emitted_length = 0
        forbidden_tag = "SOURCE_ID"
        
        # We stay 10 characters behind the LLM to ensure we never emit partial tags
        for token in self.generator.generate_stream(query, all_contexts, history_text=history_text):
            full_answer += token
            
            # Find where the forbidden tag starts
            pos = full_answer.find(forbidden_tag)
            
            if pos == -1:
                # Tag not found. Yield text that is safely behind the possible start of the tag.
                # We leave the last N characters in the buffer to check in the next iteration.
                safe_length = len(full_answer) - len(forbidden_tag)
                if safe_length > emitted_length:
                    chunk_to_yield = full_answer[emitted_length:safe_length]
                    yield chunk_to_yield
                    emitted_length += len(chunk_to_yield)
            else:
                # Tag FOUND! Yield everything up to the tag, then stop.
                last_chunk = full_answer[emitted_length:pos]
                if last_chunk:
                    yield last_chunk
                break
        else:
            # The loop finished naturally WITHOUT hitting 'break' 
            # (i.e. NO SOURCE_ID was found). 
            # We must yield the remaining characters left in the buffer.
            if len(full_answer) > emitted_length:
                yield full_answer[emitted_length:]

        # STEP 4: Final Metadata
        import re
        source_pattern = r"SOURCE_ID:\s*([\d\s,]+)"
        match = re.search(source_pattern, full_answer)
        aligned_sources = []
        
        if match:
            try:
                indices = [int(i.strip()) for i in match.group(1).replace(",", " ").split()]
                for idx_val in indices:
                    doc_idx = idx_val - 1
                    if 0 <= doc_idx < len(source_filenames):
                        primary_source = source_filenames[doc_idx]
                        if primary_source not in aligned_sources:
                            aligned_sources.append(primary_source)
            except Exception:
                pass
        
        # Fallback
        if not aligned_sources:
            if detected_doc_type == "general":
                aligned_sources = ["general"]
            elif source_filenames:
                aligned_sources = [source_filenames[0]]
            else:
                aligned_sources = []
