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
        # DYNAMIC DEPTH: If no keyword/category, search much deeper (Global Search)
        is_global = doc_type is None or doc_type == "cross_category" or doc_type == "general"
        internal_limit = 50 if is_global else 15
        
        if not is_global:
            search_filter = models.Filter(
                must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))]
            )

        # Step 1: Query the vector store with dynamic depth
        results = self.vector_store.search(
            COLLECTION_NAME,
            query_vector,
            search_filter,
            limit=internal_limit 
        )

        if not results:
            return []

        # Step 2: Lexical Boosting (Literal Search)
        # If the query contains words that literally exist in the chunk, we boost it.
        # This fixes the "Deepak Raj" problem where semantic search might be too fuzzy.
        query_words = [w.lower() for w in query.replace("?", "").split() if len(w) > 2 and w.lower() not in ["who", "is", "the", "and", "what"]]
        
        # Standalone step 2: Reranking & File Selection (MAX score)
        file_scores = {}
        file_names = {}
        file_types = {}
        file_has_lexical_match = {} # NEW: Track literal matches
        
        for res in results:
            fid = res.payload['file_id']
            score = res.score
            text = res.payload.get('text', '').lower()

            # Apply granular lexical boost
            matches = [w for w in query_words if w in text]
            if matches:
                # DYNAMIC BOOST: Higher multiplier for more unique keyword matches
                # Each match adds 0.5 to the multiplier and 0.5 to the fixed bonus
                score = (score * (1.5 + 0.5 * len(matches))) + (0.5 * len(matches))
                file_has_lexical_match[fid] = True
                # print(f"🚀 Lexical Match for {fid}: '{len(matches)}' words. New Score: {score}")

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
        threshold = best_score * 0.4 # Slightly more relaxed threshold
        
        selected_files = []
        for fid, score in sorted_files:
            # CRITICAL: Always include if there was a lexical/literal match OR score is good enough
            is_lexical = file_has_lexical_match.get(fid, False)
            if score >= threshold or is_lexical:
                selected_files.append({
                    'file_id': fid,
                    'file_name': file_names[fid],
                    'doc_type': file_types[fid],
                    'score': score
                })
        
        # Limit to top N files (standalone defaults to 3)
        return selected_files[:limit]
