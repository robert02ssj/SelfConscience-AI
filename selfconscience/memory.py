"""
MemoriaEpisodica — ChromaDB-backed long-term memory for the AI entity.

Stores every interaction (user input, internal thought, external response)
as a vector embedding and provides similarity search to retrieve contextually
relevant past memories before each new response.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings


# Collection names inside ChromaDB
_INTERACTIONS_COLLECTION = "interactions"
_THOUGHTS_COLLECTION = "thoughts"


class MemoriaEpisodica:
    """Persistent episodic memory backed by a local ChromaDB instance.

    Parameters
    ----------
    db_path:
        Directory where ChromaDB will store its persistent data.
    embedding_function:
        Optional custom embedding function (useful for testing without
        network access).  When *None* (the default), ChromaDB uses its
        built-in default embedding model.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        embedding_function: Any | None = None,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._ef = embedding_function
        self._interactions = self._client.get_or_create_collection(
            name=_INTERACTIONS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,  # type: ignore[arg-type]
        )
        self._thoughts = self._client.get_or_create_collection(
            name=_THOUGHTS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_interaction(
        self,
        user_input: str,
        internal_thought: str,
        external_response: str,
        entity_state: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a full interaction turn and return its unique ID."""
        doc_id = str(uuid.uuid4())
        # Embed the concatenation of all relevant text so the memory is
        # searchable from multiple angles.
        document = (
            f"[Usuario]: {user_input}\n"
            f"[Pensamiento]: {internal_thought}\n"
            f"[Respuesta]: {external_response}"
        )
        metadata: dict[str, Any] = {
            "timestamp": time.time(),
            "user_input": user_input[:500],
            "internal_thought": internal_thought[:500],
            "external_response": external_response[:500],
            "type": "interaction",
        }
        if entity_state:
            metadata.update({f"state_{k}": v for k, v in entity_state.items()})
        if extra_metadata:
            metadata.update(extra_metadata)

        self._interactions.add(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata],
        )
        return doc_id

    def store_autonomous_thought(
        self,
        thought: str,
        mode: str = "reflection",
        entity_state: dict[str, Any] | None = None,
    ) -> str:
        """Persist a thought generated during Reflection or Sleep mode."""
        doc_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "timestamp": time.time(),
            "thought": thought[:1000],
            "mode": mode,
            "type": "autonomous_thought",
        }
        if entity_state:
            metadata.update({f"state_{k}": v for k, v in entity_state.items()})

        self._thoughts.add(
            ids=[doc_id],
            documents=[thought],
            metadatas=[metadata],
        )
        return doc_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search_similar_interactions(
        self, query: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Return the *n_results* most similar past interactions to *query*."""
        count = self._interactions.count()
        if count == 0:
            return []
        actual_n = min(n_results, count)
        results = self._interactions.query(
            query_texts=[query],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(results)

    def search_similar_thoughts(
        self, query: str, n_results: int = 3
    ) -> list[dict[str, Any]]:
        """Return the *n_results* most similar autonomous thoughts to *query*."""
        count = self._thoughts.count()
        if count == 0:
            return []
        actual_n = min(n_results, count)
        results = self._thoughts.query(
            query_texts=[query],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(results)

    def get_recent_interactions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the *limit* most recent interactions sorted by timestamp."""
        count = self._interactions.count()
        if count == 0:
            return []
        result = self._interactions.get(
            include=["documents", "metadatas"],
            limit=min(limit * 3, count),  # over-fetch and sort client-side
        )
        items = [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        items.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
        return items[:limit]

    def total_memories(self) -> dict[str, int]:
        return {
            "interactions": self._interactions.count(),
            "thoughts": self._thoughts.count(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten ChromaDB query results into a list of dicts."""
        output: list[dict[str, Any]] = []
        if not raw.get("ids") or not raw["ids"][0]:
            return output
        ids = raw["ids"][0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            output.append(
                {
                    "id": doc_id,
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )
        return output
