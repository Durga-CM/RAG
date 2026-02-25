from qdrant_client.http import models
from app.core.config import COLLECTION_NAME

class RetrievalService:

    def __init__(self, embedding_service, vector_store):
        self.embedding = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query, doc_type=None, detected_filename=None, limit=3):
        """
        Retrieves relevant documents based on content similarity and LLM classification.
        Directly from invoice_rag_standalone.py ranking logic.
        """
        query_vector = self.embedding.embed_query(query)

        search_filter = None
        if doc_type and doc_type != "cross_category":
            search_filter = models.Filter(
                must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))]
            )

        # Standalone step 1: client.query_points with limit 10
        results = self.vector_store.search(
            COLLECTION_NAME,
            query_vector,
            search_filter,
            limit=10 
        )

        if not results:
            return []

        # Standalone step 2: Reranking & File Selection (MAX score)
        file_scores = {}
        file_names = {}
        file_types = {}
        
        for res in results:
            fid = res.payload['file_id']
            score = res.score
            
            # CRITICAL FIX: Use MAX score instead of SUM.
            file_scores[fid] = max(file_scores.get(fid, 0), score)
            file_names[fid] = res.payload['file_name']
            file_types[fid] = res.payload.get('doc_type', 'general')

        # ENHANCEMENT: Boost score if filename matches query
        if detected_filename:
            for fid, fname in file_names.items():
                if detected_filename.lower() in fname.lower():
                    file_scores[fid] *= 1.5
                    print(f"⭐ Boosted score for '{fname}' due to filename match")

        # Standalone step 3: Multi-Document Selection
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        best_score = sorted_files[0][1] if sorted_files else 0
        threshold = best_score * 0.5
        
        selected_files = []
        for fid, score in sorted_files:
            # Include if score is good enough OR explicitly mentioned
            if score >= threshold or (detected_filename and detected_filename.lower() in file_names[fid].lower()):
                selected_files.append({
                    'file_id': fid,
                    'file_name': file_names[fid],
                    'doc_type': file_types[fid],
                    'score': score
                })
        
        # Limit to top N files (standalone defaults to 3)
        return selected_files[:limit]
