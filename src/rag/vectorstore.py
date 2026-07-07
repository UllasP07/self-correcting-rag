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
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import settings
from .errors import EmbedderMismatchError

# A fixed, reserved point that stores the *embedder fingerprint* — which
# provider/model/dim built this collection. It carries a unit vector (Qdrant
# rejects all-zero vectors under cosine) and a __meta__ flag so search excludes
# it. This is how we detect (and refuse) querying an index with a different
# embedder than the one that populated it.
_META_ID = "00000000-0000-0000-0000-000000000000"


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
        # Guard: zip() would silently drop chunks if these lengths diverge.
        if len(texts) != len(vectors):
            raise ValueError(
                f"upsert length mismatch: {len(texts)} texts vs {len(vectors)} vectors"
            )
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

    # --- Embedder fingerprint (Milestone 1.5 guard) ---

    def write_fingerprint(self, fp: dict) -> None:
        """Stamp the collection with the embedder that built it."""
        unit_vec = [1.0] + [0.0] * (int(fp["dim"]) - 1)
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=_META_ID, vector=unit_vec,
                                payload={"__meta__": True, **fp})],
        )

    def read_fingerprint(self) -> dict | None:
        """Return the stored embedder fingerprint, or None if unstamped."""
        try:
            pts = self.client.retrieve(
                collection_name=self.collection, ids=[_META_ID], with_payload=True
            )
        except Exception:  # noqa: BLE001 — collection may not exist yet
            return None
        if not pts:
            return None
        p = dict(pts[0].payload or {})
        p.pop("__meta__", None)
        return p or None

    def assert_embedder(self, current: dict) -> None:
        """Refuse to query if the current embedder != the one that built the index."""
        stored = self.read_fingerprint()
        if stored is None:
            return  # unstamped (older index) — nothing to check against
        key = ("provider", "model", "dim")
        if tuple(stored.get(k) for k in key) != tuple(current.get(k) for k in key):
            raise EmbedderMismatchError(
                "This index was built with embedder "
                f"{stored.get('provider')}/{stored.get('model')} "
                f"({stored.get('dim')}-dim), but you're querying with "
                f"{current.get('provider')}/{current.get('model')} "
                f"({current.get('dim')}-dim).\n"
                "Vector similarity across different embedders is meaningless. "
                "Re-ingest with:  python -m src.rag.ingest --recreate\n"
                "(or switch EMBED_PROVIDER/EMBED_DIM back to match the index)."
            )

    def search(self, query_vector: list[float], top_k: int | None = None) -> list[Hit]:
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k or settings.top_k,
            with_payload=True,
            # Exclude the reserved fingerprint point from results.
            query_filter=Filter(
                must_not=[FieldCondition(key="__meta__", match=MatchValue(value=True))]
            ),
        ).points
        return [
            Hit(
                text=r.payload.get("text", ""),
                score=r.score,
                source=r.payload.get("source", "?"),
            )
            for r in results
        ]
