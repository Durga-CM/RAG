from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE

class VectorStoreService:

    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def ensure_collection(self, name):
        """Creates collection only if it doesn't already exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == name for c in collections):
            print(f"📦 Creating new collection: {name}")
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )

    def file_exists(self, collection_name, file_name):
        """Checks if points with this file_name already exist in the collection."""
        results = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="file_name",
                    match=models.MatchValue(value=file_name)
                )]
            ),
            limit=1
        )[0]
        return len(results) > 0

    def create_collection(self, name):
        """DESTRUCTIVE: Deletes and recreates the collection."""
        print(f"⚠️ Recreating collection: {name}")
        collections = self.client.get_collections().collections
        if any(c.name == name for c in collections):
            self.client.delete_collection(name)
            
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )


    def upsert(self, collection_name, points):
        self.client.upsert(collection_name=collection_name, points=points)

    def search(self, collection_name, vector, query_filter=None, limit=10):
        return self.client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit
        ).points

    def scroll_by_file(self, collection_name, file_id):
        return self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="file_id",
                    match=models.MatchValue(value=file_id)
                )]
            ),
            limit=100
        )[0]
