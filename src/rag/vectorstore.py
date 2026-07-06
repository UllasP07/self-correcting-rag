"""Qdrant wrapper: store chunk vectors and search them.

A vector DB stores each chunk as (id, vector, payload). At query time we hand it
a query vector and it returns the nearest chunks by cosine similarity, plus a
score in [0, 1]. That score becomes important in Milestone 4 (CRAG): a low score
means "my documents probably don't contain the answer — go correct course."
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import settings


@dataclass
class Hit:
    """One search result."""
    text: str
    score: float
    source: str


class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if missing. `recreate=True` wipes it first."""
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                # Cosine distance is the standard choice for text embeddings.
                vectors_config=VectorParams(
                    size=settings.embed_dim, distance=Distance.COSINE
                ),
            )

    def upsert(self, texts: list[str], vectors: list[list[float]], source: str) -> int:
        """Insert chunks. Each point carries its text + source file in the payload."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"text": txt, "source": source},
            )
            for txt, vec in zip(texts, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query_vector: list[float], top_k: int | None = None) -> list[Hit]:
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k or settings.top_k,
            with_payload=True,
        ).points
        return [
            Hit(
                text=r.payload.get("text", ""),
                score=r.score,
                source=r.payload.get("source", "?"),
            )
            for r in results
        ]
