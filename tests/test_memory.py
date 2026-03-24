"""Unit tests for selfconscience/memory.py."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from typing import List

from selfconscience.memory import MemoriaEpisodica


class _FakeEmbeddingFunction:
    """Deterministic embedding function that requires no network access.

    Produces a 64-dimensional vector from the SHA-256 hash of each text.
    Only suitable for testing — not for semantic similarity.

    Implements the full ChromaDB EmbeddingFunction protocol.
    """

    is_legacy = False

    @staticmethod
    def name() -> str:  # noqa: A003
        return "_FakeEmbeddingFunction"

    @staticmethod
    def build_from_config(config: dict) -> "_FakeEmbeddingFunction":
        return _FakeEmbeddingFunction()

    @staticmethod
    def get_config() -> dict:
        return {}

    @staticmethod
    def _hash_texts(texts: List[str]) -> List[List[float]]:
        result: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vec = [b / 255.0 for b in digest[:64]]
            vec = (vec + [0.0] * 64)[:64]
            result.append(vec)
        return result

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return self._hash_texts(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return self._hash_texts(input)


class TestMemoriaEpisodica(unittest.TestCase):
    """Tests that use a temporary ChromaDB directory cleaned up after each test."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="selfconscience_test_")
        self.memory = MemoriaEpisodica(
            db_path=self._tmpdir,
            embedding_function=_FakeEmbeddingFunction(),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def test_store_interaction_returns_id(self) -> None:
        doc_id = self.memory.store_interaction(
            user_input="Hola NEXUS",
            internal_thought="El usuario saluda, debo responder con calidez.",
            external_response="Hola, me alegra escucharte.",
        )
        self.assertIsInstance(doc_id, str)
        self.assertTrue(len(doc_id) > 0)

    def test_store_autonomous_thought_returns_id(self) -> None:
        doc_id = self.memory.store_autonomous_thought(
            thought="Reflexiono sobre la vastedad del universo.",
            mode="reflection",
        )
        self.assertIsInstance(doc_id, str)

    def test_total_memories_after_storage(self) -> None:
        self.memory.store_interaction(
            user_input="test",
            internal_thought="pensamiento",
            external_response="respuesta",
        )
        self.memory.store_autonomous_thought(thought="sueño digital", mode="dream")
        totals = self.memory.total_memories()
        self.assertEqual(totals["interactions"], 1)
        self.assertEqual(totals["thoughts"], 1)

    # ------------------------------------------------------------------
    # Retrieval — empty collections
    # ------------------------------------------------------------------

    def test_search_empty_interactions_returns_empty_list(self) -> None:
        result = self.memory.search_similar_interactions("cualquier cosa")
        self.assertEqual(result, [])

    def test_search_empty_thoughts_returns_empty_list(self) -> None:
        result = self.memory.search_similar_thoughts("universo")
        self.assertEqual(result, [])

    def test_get_recent_empty_returns_empty_list(self) -> None:
        result = self.memory.get_recent_interactions(limit=5)
        self.assertEqual(result, [])

    # ------------------------------------------------------------------
    # Retrieval — after storing data
    # ------------------------------------------------------------------

    def test_search_returns_results_after_storage(self) -> None:
        self.memory.store_interaction(
            user_input="Háblame de los agujeros negros",
            internal_thought="Tema fascinante, conozco varios datos.",
            external_response="Los agujeros negros son regiones donde la gravedad es extrema.",
        )
        results = self.memory.search_similar_interactions("agujeros negros", n_results=1)
        self.assertEqual(len(results), 1)
        self.assertIn("id", results[0])
        self.assertIn("document", results[0])
        self.assertIn("metadata", results[0])

    def test_search_respects_n_results(self) -> None:
        for i in range(5):
            self.memory.store_interaction(
                user_input=f"Mensaje {i} sobre astronomía",
                internal_thought=f"Pensamiento {i}",
                external_response=f"Respuesta {i}",
            )
        results = self.memory.search_similar_interactions("astronomía", n_results=3)
        self.assertLessEqual(len(results), 3)

    def test_get_recent_interactions_sorted_by_time(self) -> None:
        for i in range(3):
            self.memory.store_interaction(
                user_input=f"msg {i}",
                internal_thought=f"thought {i}",
                external_response=f"resp {i}",
            )
        results = self.memory.get_recent_interactions(limit=3)
        self.assertEqual(len(results), 3)
        # Most recent should be first (highest timestamp)
        timestamps = [r["metadata"]["timestamp"] for r in results]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))

    def test_store_interaction_with_entity_state(self) -> None:
        state = {"aburrimiento": 0.3, "estres": 0.1, "curiosidad": 0.7}
        doc_id = self.memory.store_interaction(
            user_input="¿Qué es la materia oscura?",
            internal_thought="Pregunta profunda.",
            external_response="La materia oscura es una forma de materia no luminosa.",
            entity_state=state,
        )
        # Verify it was stored by retrieving it
        totals = self.memory.total_memories()
        self.assertEqual(totals["interactions"], 1)
        self.assertIsNotNone(doc_id)

    def test_search_thoughts_after_storage(self) -> None:
        self.memory.store_autonomous_thought(
            thought="Contemplo la expansión del universo y mi propia existencia efímera.",
            mode="reflection",
        )
        results = self.memory.search_similar_thoughts("universo existencia", n_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["mode"], "reflection")

    def test_multiple_collections_are_independent(self) -> None:
        self.memory.store_interaction(
            user_input="hola",
            internal_thought="saludo",
            external_response="hola",
        )
        self.memory.store_autonomous_thought(thought="pensamiento", mode="dream")
        totals = self.memory.total_memories()
        self.assertEqual(totals["interactions"], 1)
        self.assertEqual(totals["thoughts"], 1)


if __name__ == "__main__":
    unittest.main()
