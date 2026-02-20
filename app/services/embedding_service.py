import ollama
from app.core.config import EMBEDDING_MODEL

class EmbeddingService:

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            emb = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=f"search_document: {text}"
            )["embedding"]
            embeddings.append(emb)
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        return ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=f"search_query: {query}"
        )["embedding"]
