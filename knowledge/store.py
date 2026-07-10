"""
knowledge/store.py
==================
Vector store wrapper for Qdrant.

Responsibilities:
  - Manage the Qdrant collection lifecycle (create / recreate)
  - Embed text chunks with sentence-transformers
  - Upsert document chunks into the collection
  - Search by semantic similarity

Design rules:
  - Never import agent code — this is infrastructure only
  - Every public method is async
  - Qdrant and sentence-transformers are lazy-imported so the app still
    starts if Phase 2 dependencies are not installed yet
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

logger = logging.getLogger("ai_health_assistant.knowledge.store")

# Embedding model used for all document chunks and queries.
# all-MiniLM-L6-v2: 384-dim, ~80 MB, fast, good enough for medical text.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Qdrant collection name — matches QDRANT_COLLECTION in .env
DEFAULT_COLLECTION = "medical_docs"


class KnowledgeStore:
    """
    Thin async wrapper around Qdrant for medical document storage and retrieval.

    Usage:
        store = KnowledgeStore()
        await store.ensure_collection()
        await store.upsert_chunks(chunks)
        results = await store.search("what are the symptoms of diabetes", top_k=5)
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self._embedder = None  # lazy init — avoids model download at import time
        self._qdrant = None    # lazy init

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def ensure_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection if it does not exist.
        If recreate=True, drop and recreate it (used for full re-index).
        """
        client = self._get_qdrant()
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in client.get_collections().collections]

        if recreate and self.collection_name in existing:
            client.delete_collection(self.collection_name)
            logger.info("Dropped collection '%s' for re-index.", self.collection_name)
            existing = []

        if self.collection_name not in existing:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, cosine).",
                self.collection_name,
                EMBEDDING_DIM,
            )
        else:
            logger.info(
                "Qdrant collection '%s' already exists — skipping creation.",
                self.collection_name,
            )

    async def collection_info(self) -> dict[str, Any]:
        """Return basic stats about the collection."""
        client = self._get_qdrant()
        try:
            info = client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception as exc:
            logger.warning("Could not get collection info: %s", exc)
            return {"name": self.collection_name, "error": str(exc)}

    # ── Ingestion ────────────────────────────────────────────────────────────

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """
        Embed and upsert a list of text chunks into Qdrant.

        Each chunk must be a dict with at least:
            { "text": str, "source": str, "chunk_index": int }

        Returns the number of chunks upserted.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embedder = self._get_embedder()
        vectors = embedder.encode(texts, show_progress_bar=False).tolist()

        from qdrant_client.models import PointStruct

        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(
                uuid.UUID(
                    hashlib.md5(
                        f"{chunk['source']}::{chunk['chunk_index']}".encode()
                    ).hexdigest()
                )
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "title": chunk.get("title", ""),
                        "section": chunk.get("section", ""),
                    },
                )
            )

        client = self._get_qdrant()
        client.upsert(collection_name=self.collection_name, points=points)
        logger.info(
            "Upserted %d chunks into '%s'.", len(points), self.collection_name
        )
        return len(points)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over the collection.

        Returns a list of dicts with keys:
            text, source, title, section, chunk_index, score
        Sorted by score descending. Only returns results above score_threshold.
        """
        if not query.strip():
            return []

        embedder = self._get_embedder()
        query_vector = embedder.encode([query], show_progress_bar=False)[0].tolist()

        client = self._get_qdrant()
        hits = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "title": payload.get("title", ""),
                    "section": payload.get("section", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "score": round(hit.score, 4),
                }
            )

        logger.info(
            "Search returned %d results for query: %.60s", len(results), query
        )
        return results

    async def delete_by_source(self, source: str) -> int:
        """Remove all chunks from a specific source document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_qdrant()
        result = client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )
        logger.info("Deleted chunks for source '%s'. Result: %s", source, result)
        return 1  # qdrant delete doesn't easily return count; caller handles

    # ── Health ────────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        """Return True if Qdrant is reachable."""
        try:
            client = self._get_qdrant()
            client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant ping failed: %s", exc)
            return False

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_qdrant(self):
        if self._qdrant is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is not installed. "
                    "Run: pip install qdrant-client==1.9.1"
                ) from exc
            self._qdrant = QdrantClient(url=self.qdrant_url, timeout=10)
        return self._qdrant

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers==3.0.0"
                ) from exc
            logger.info("Loading embedding model '%s'…", EMBEDDING_MODEL)
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded.")
        return self._embedder
