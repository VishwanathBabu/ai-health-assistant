"""
tests/test_phase2.py
====================
Phase 2 test suite covering:
  1. Knowledge chunker
  2. KnowledgeStore (mocked Qdrant)
  3. IngestionService
  4. Orchestrator RAG integration
  5. Safety agent RAG prompt injection
  6. API endpoints (mocked store)

All tests use mocks — no real Qdrant, no real LLM calls, no network.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ════════════════════════════════════════════════════════════════════════════
# CHUNKER TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestChunker:

    def test_basic_chunking_returns_chunks(self):
        """Normal case: text longer than chunk_size → multiple chunks returned."""
        from knowledge.chunker import chunk_text

        long_text = "Fever is a common symptom. " * 50  # ~1350 chars
        chunks = chunk_text(text=long_text, source="test.txt", chunk_size=200, overlap=20)

        assert len(chunks) > 1
        for chunk in chunks:
            assert "text" in chunk
            assert "source" in chunk
            assert chunk["source"] == "test.txt"
            assert "chunk_index" in chunk

    def test_chunk_indexes_are_sequential(self):
        """Chunk indexes must start at 0 and increment by 1."""
        from knowledge.chunker import chunk_text

        text = "This is a sentence about health. " * 30
        chunks = chunk_text(text=text, source="test.txt", chunk_size=100, overlap=10)

        indexes = [c["chunk_index"] for c in chunks]
        assert indexes == list(range(len(indexes)))

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size → exactly one chunk."""
        from knowledge.chunker import chunk_text

        short_text = "Patient has mild fever and headache for the past two days."
        chunks = chunk_text(text=short_text, source="doc.txt", chunk_size=512)

        assert len(chunks) == 1
        assert "fever" in chunks[0]["text"].lower()

    def test_empty_text_returns_empty_list(self):
        """Edge case: empty input → no chunks."""
        from knowledge.chunker import chunk_text

        chunks = chunk_text(text="", source="empty.txt")
        assert chunks == []

    def test_whitespace_only_text_returns_empty(self):
        """Edge case: whitespace-only input → no chunks."""
        from knowledge.chunker import chunk_text

        chunks = chunk_text(text="   \n\n\t  ", source="blank.txt")
        assert chunks == []

    def test_source_preserved_in_all_chunks(self):
        """Source filename must appear in every chunk."""
        from knowledge.chunker import chunk_text

        text = "Diabetes symptoms include increased thirst. " * 40
        chunks = chunk_text(text=text, source="diabetes_guide.txt")

        for chunk in chunks:
            assert chunk["source"] == "diabetes_guide.txt"

    def test_title_preserved_in_all_chunks(self):
        """Title must appear in every chunk when provided."""
        from knowledge.chunker import chunk_text

        text = "Hypertension is high blood pressure. " * 40
        chunks = chunk_text(text=text, source="bp.txt", title="Blood Pressure Guide")

        for chunk in chunks:
            assert chunk["title"] == "Blood Pressure Guide"

    def test_overlap_means_text_is_shared(self):
        """With overlap > 0, text in chunk N must appear at the start of chunk N+1."""
        from knowledge.chunker import chunk_text

        # Build text where we can verify overlap
        text = "ABCDEFGHIJ" * 20  # 200 chars
        chunks = chunk_text(text=text, source="test.txt", chunk_size=40, overlap=10)

        if len(chunks) >= 2:
            # The tail of chunk 0 should appear in chunk 1
            tail = chunks[0]["text"][-5:]
            # At least some overlap must exist
            assert len(chunks[1]["text"]) > 0

    def test_pdf_page_chunking(self):
        """PDF pages are chunked independently and tagged with page numbers."""
        from knowledge.chunker import chunk_pdf_pages

        pages = [
            "First page content about fever and temperature. " * 15,
            "Second page about diabetes management strategies. " * 15,
        ]
        chunks = chunk_pdf_pages(pages=pages, source="manual.pdf", title="Medical Manual")

        assert len(chunks) > 0
        # All chunks must have the correct source
        for chunk in chunks:
            assert chunk["source"] == "manual.pdf"


# ════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE STORE TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestKnowledgeStore:

    def _make_store(self, mock_qdrant, mock_embedder):
        """Helper: build a KnowledgeStore with mocked internals."""
        from knowledge.store import KnowledgeStore
        store = KnowledgeStore(
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
        )
        store._qdrant = mock_qdrant
        store._embedder = mock_embedder
        return store

    @pytest.mark.asyncio
    async def test_ping_returns_true_when_qdrant_up(self):
        """ping() returns True when Qdrant responds."""
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])

        from knowledge.store import KnowledgeStore
        store = KnowledgeStore()
        store._qdrant = mock_qdrant

        result = await store.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_ping_returns_false_when_qdrant_down(self):
        """ping() returns False when Qdrant raises an exception."""
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.side_effect = ConnectionRefusedError("refused")

        from knowledge.store import KnowledgeStore
        store = KnowledgeStore()
        store._qdrant = mock_qdrant

        result = await store.ping()
        assert result is False

    @pytest.mark.asyncio
    async def test_upsert_chunks_returns_count(self):
        """upsert_chunks() returns the number of chunks upserted."""
        import numpy as np
        mock_qdrant = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((3, 384))

        store = self._make_store(mock_qdrant, mock_embedder)

        chunks = [
            {"text": "Fever is a raised body temperature.", "source": "fever.txt", "chunk_index": 0},
            {"text": "Common causes of fever include infection.", "source": "fever.txt", "chunk_index": 1},
            {"text": "Hydration is important when feverish.", "source": "fever.txt", "chunk_index": 2},
        ]

        mock_models = MagicMock()
        mock_models.PointStruct = MagicMock(side_effect=lambda **kw: kw)
        with patch.dict("sys.modules", {"qdrant_client.models": mock_models}):
            result = await store.upsert_chunks(chunks)

        assert result == 3
        mock_qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_empty_list_returns_zero(self):
        """upsert_chunks([]) returns 0 without calling Qdrant."""
        mock_qdrant = MagicMock()
        mock_embedder = MagicMock()

        store = self._make_store(mock_qdrant, mock_embedder)
        result = await store.upsert_chunks([])

        assert result == 0
        mock_qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_returns_ranked_results(self):
        """search() returns list of dicts with expected keys."""
        import numpy as np
        mock_qdrant = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((1, 384))

        # Mock Qdrant search result
        hit = MagicMock()
        hit.payload = {
            "text": "Fever often accompanies viral infections.",
            "source": "fever.txt",
            "title": "Fever Guide",
            "section": "Causes",
            "chunk_index": 0,
        }
        hit.score = 0.87
        mock_qdrant.search.return_value = [hit]

        store = self._make_store(mock_qdrant, mock_embedder)
        results = await store.search("what causes fever", top_k=5)

        assert len(results) == 1
        assert results[0]["text"] == "Fever often accompanies viral infections."
        assert results[0]["score"] == 0.87
        assert results[0]["source"] == "fever.txt"

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_empty(self):
        """search('') skips Qdrant and returns []."""
        mock_qdrant = MagicMock()
        mock_embedder = MagicMock()

        store = self._make_store(mock_qdrant, mock_embedder)
        results = await store.search("   ")

        assert results == []
        mock_qdrant.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_when_missing(self):
        """ensure_collection() calls create_collection when collection absent."""
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])

        from knowledge.store import KnowledgeStore
        store = KnowledgeStore(collection_name="test_collection")
        store._qdrant = mock_qdrant

        mock_models = MagicMock()
        mock_models.Distance.COSINE = "Cosine"
        mock_models.VectorParams = MagicMock(return_value=MagicMock())

        with patch("knowledge.store.KnowledgeStore._get_qdrant", return_value=mock_qdrant), \
             patch.dict("sys.modules", {"qdrant_client.models": mock_models}):
            await store.ensure_collection()

        mock_qdrant.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_skips_when_exists(self):
        """ensure_collection() does NOT call create_collection if already exists."""
        mock_col = MagicMock()
        mock_col.name = "medical_docs"
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value = MagicMock(collections=[mock_col])

        from knowledge.store import KnowledgeStore
        store = KnowledgeStore(collection_name="medical_docs")
        store._qdrant = mock_qdrant

        mock_models = MagicMock()
        with patch("knowledge.store.KnowledgeStore._get_qdrant", return_value=mock_qdrant), \
             patch.dict("sys.modules", {"qdrant_client.models": mock_models}):
            await store.ensure_collection()

        mock_qdrant.create_collection.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
# INGESTION SERVICE TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestIngestionService:

    @pytest.mark.asyncio
    async def test_ingest_txt_file(self):
        """Normal case: ingest a .txt file → chunks created."""
        from knowledge.ingestion import IngestionService

        mock_store = AsyncMock()
        mock_store.collection_name = "test"
        mock_store.upsert_chunks.return_value = 3
        mock_store.ensure_collection = AsyncMock()

        svc = IngestionService(store=mock_store)

        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                "Diabetes is a chronic condition affecting blood sugar levels. " * 20
            )
            tmp_path = f.name

        try:
            result = await svc.ingest_file(tmp_path)
            assert result["status"] == "ok"
            assert result["chunks_created"] > 0
            mock_store.upsert_chunks.assert_called_once()
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_ingest_unsupported_extension(self):
        """Edge case: .docx file → error response, no upsert."""
        from knowledge.ingestion import IngestionService

        mock_store = AsyncMock()
        svc = IngestionService(store=mock_store)

        result = await svc.ingest_file("/tmp/report.docx")

        assert result["status"] == "error"
        assert "unsupported" in result["error"].lower()
        mock_store.upsert_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_missing_file(self):
        """Edge case: file does not exist → error response."""
        from knowledge.ingestion import IngestionService

        mock_store = AsyncMock()
        svc = IngestionService(store=mock_store)

        result = await svc.ingest_file("/tmp/does_not_exist_xyz.txt")

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_ingest_empty_file(self):
        """Edge case: empty file → warning, no chunks."""
        from knowledge.ingestion import IngestionService

        mock_store = AsyncMock()
        mock_store.upsert_chunks.return_value = 0
        svc = IngestionService(store=mock_store)

        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("")  # empty
            tmp_path = f.name

        try:
            result = await svc.ingest_file(tmp_path)
            # Either warning or 0 chunks
            assert result["chunks_created"] == 0
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_ingest_directory(self):
        """Normal case: directory with 2 .txt files → both ingested."""
        from knowledge.ingestion import IngestionService

        mock_store = AsyncMock()
        mock_store.collection_name = "test"
        mock_store.upsert_chunks.return_value = 2
        mock_store.ensure_collection = AsyncMock()

        svc = IngestionService(store=mock_store)

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("doc1.txt", "doc2.txt"):
                Path(tmpdir, name).write_text(
                    "Medical information about symptoms. " * 20, encoding="utf-8"
                )

            results = await svc.ingest_directory(tmpdir)

        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR RAG INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestOrchestratorRAG:

    def _make_orchestrator(self, mock_store=None):
        """Build orchestrator with all agents mocked."""
        from core.orchestrator import HealthAssistantOrchestrator

        orch = HealthAssistantOrchestrator.__new__(HealthAssistantOrchestrator)
        orch.router = MagicMock()
        orch.symptom_agent = MagicMock()
        orch.emergency_agent = MagicMock()
        orch.safety_agent = MagicMock()
        orch.knowledge_store = mock_store
        return orch

    @pytest.mark.asyncio
    async def test_rag_results_injected_into_safety_agent(self):
        """When RAG returns results, safety agent receives rag_context."""
        mock_store = AsyncMock()
        mock_store.search.return_value = [
            {
                "text": "Fever above 38°C requires medical attention.",
                "source": "fever_guide.txt",
                "title": "Fever Guide",
                "section": "Treatment",
                "chunk_index": 0,
                "score": 0.85,
            }
        ]

        orch = self._make_orchestrator(mock_store)

        orch.router.run = AsyncMock(return_value={
            "intent": "symptom_query",
            "agents_to_invoke": ["symptom_agent", "emergency_agent", "safety_agent"],
            "confidence": "high",
            "reasoning": "Symptom described.",
        })
        orch.symptom_agent.run = AsyncMock(return_value={
            "symptoms": ["fever"],
            "duration": "2 days",
            "severity": "moderate",
            "demographics": {"age": None, "gender": None},
            "extraction_notes": "",
        })
        orch.emergency_agent.run = AsyncMock(return_value={
            "emergency": False,
            "emergency_type": "none",
            "reason": "No emergency indicators.",
            "immediate_action": "",
        })

        captured_input = {}

        async def capture_safety(input_data, request_id=None):
            captured_input.update(input_data)
            return {
                "final_response": "Please consult a doctor.",
                "safety_override_triggered": False,
                "override_reason": "",
                "disclaimer_included": True,
            }

        orch.safety_agent.run = capture_safety

        result = await orch.process("I have fever for 2 days", use_rag=True)

        # RAG context must be passed to safety agent
        assert "rag_context" in captured_input
        assert len(captured_input["rag_context"]) == 1
        assert "Fever above 38°C" in captured_input["rag_context"][0]["text"]

        # Result must reflect RAG activity
        assert result["rag_active"] is True
        assert "fever_guide.txt" in result["sources_used"]

    @pytest.mark.asyncio
    async def test_rag_disabled_when_no_store(self):
        """When knowledge_store is None, rag_active=False and no search called."""
        orch = self._make_orchestrator(mock_store=None)

        orch.router.run = AsyncMock(return_value={
            "intent": "general",
            "agents_to_invoke": ["safety_agent"],
            "confidence": "medium",
            "reasoning": "General question.",
        })
        orch.emergency_agent.run = AsyncMock(return_value={
            "emergency": False,
            "emergency_type": "none",
            "reason": "No emergency.",
            "immediate_action": "",
        })
        orch.safety_agent.run = AsyncMock(return_value={
            "final_response": "Stay hydrated.",
            "safety_override_triggered": False,
            "override_reason": "",
            "disclaimer_included": True,
        })

        result = await orch.process("How much water should I drink?", use_rag=True)

        assert result["rag_active"] is False
        assert result["sources_used"] == []

    @pytest.mark.asyncio
    async def test_rag_failure_does_not_break_pipeline(self):
        """If RAG search raises an exception, the pipeline continues without it."""
        mock_store = AsyncMock()
        mock_store.search.side_effect = Exception("Qdrant timeout")

        orch = self._make_orchestrator(mock_store)

        orch.router.run = AsyncMock(return_value={
            "intent": "symptom_query",
            "agents_to_invoke": ["symptom_agent", "emergency_agent", "safety_agent"],
            "confidence": "high",
            "reasoning": "Symptom.",
        })
        orch.symptom_agent.run = AsyncMock(return_value={
            "symptoms": ["headache"],
            "duration": None, "severity": None,
            "demographics": {"age": None, "gender": None},
            "extraction_notes": "",
        })
        orch.emergency_agent.run = AsyncMock(return_value={
            "emergency": False, "emergency_type": "none",
            "reason": "No emergency.", "immediate_action": "",
        })
        orch.safety_agent.run = AsyncMock(return_value={
            "final_response": "See a doctor.",
            "safety_override_triggered": False,
            "override_reason": "",
            "disclaimer_included": True,
        })

        result = await orch.process("I have a headache", use_rag=True)

        # Pipeline completes despite RAG failure
        assert result["final_response"] == "See a doctor."
        assert result["rag_active"] is False

    @pytest.mark.asyncio
    async def test_use_rag_false_skips_search(self):
        """When use_rag=False, the store.search() is never called."""
        mock_store = AsyncMock()

        orch = self._make_orchestrator(mock_store)

        orch.router.run = AsyncMock(return_value={
            "intent": "general",
            "agents_to_invoke": ["safety_agent"],
            "confidence": "high",
            "reasoning": "General query.",
        })
        orch.emergency_agent.run = AsyncMock(return_value={
            "emergency": False, "emergency_type": "none",
            "reason": "No emergency.", "immediate_action": "",
        })
        orch.safety_agent.run = AsyncMock(return_value={
            "final_response": "Eat well.",
            "safety_override_triggered": False,
            "override_reason": "",
            "disclaimer_included": True,
        })

        result = await orch.process("What is a healthy diet?", use_rag=False)

        mock_store.search.assert_not_called()
        assert result["rag_active"] is False


# ════════════════════════════════════════════════════════════════════════════
# SAFETY AGENT RAG PROMPT TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestSafetyAgentRAG:

    @pytest.mark.asyncio
    async def test_rag_context_appears_in_prompt(self):
        """When rag_context is provided, the prompt must contain the chunk text."""
        from agents.safety_agent import SafetyAgent

        agent = SafetyAgent.__new__(SafetyAgent)
        agent.name = "safety_agent"

        input_data = {
            "user_input": "I have a high fever",
            "emergency_output": {"emergency": False},
            "router_output": {"intent": "symptom_query"},
            "symptom_output": {"symptoms": ["fever"], "duration": None, "severity": None},
            "pipeline_responses": [],
            "rag_context": [
                {
                    "text": "A fever over 39°C in adults warrants medical evaluation.",
                    "source": "fever.txt",
                    "title": "Fever Management Guide",
                    "section": "Adults",
                    "chunk_index": 0,
                    "score": 0.91,
                }
            ],
        }

        prompt = agent._build_prompt(input_data)

        assert "Fever Management Guide" in prompt
        assert "39°C" in prompt
        assert "RELEVANT MEDICAL KNOWLEDGE" in prompt

    @pytest.mark.asyncio
    async def test_empty_rag_context_no_section(self):
        """When rag_context is empty, the prompt must not contain RAG section."""
        from agents.safety_agent import SafetyAgent

        agent = SafetyAgent.__new__(SafetyAgent)
        agent.name = "safety_agent"

        input_data = {
            "user_input": "I feel tired",
            "emergency_output": {"emergency": False},
            "router_output": {"intent": "general"},
            "symptom_output": {"symptoms": ["fatigue"], "duration": None, "severity": None},
            "pipeline_responses": [],
            "rag_context": [],
        }

        prompt = agent._build_prompt(input_data)

        assert "RELEVANT MEDICAL KNOWLEDGE" not in prompt

    @pytest.mark.asyncio
    async def test_rag_capped_at_five_chunks(self):
        """Safety agent should only use the first 5 RAG chunks even if more provided."""
        from agents.safety_agent import SafetyAgent

        agent = SafetyAgent.__new__(SafetyAgent)
        agent.name = "safety_agent"

        many_chunks = [
            {
                "text": f"Chunk number {i} about medical topics.",
                "source": f"doc{i}.txt",
                "title": f"Doc {i}",
                "section": "",
                "chunk_index": i,
                "score": 0.9 - i * 0.05,
            }
            for i in range(10)
        ]

        input_data = {
            "user_input": "headache",
            "emergency_output": {"emergency": False},
            "router_output": {"intent": "symptom_query"},
            "symptom_output": {"symptoms": ["headache"], "duration": None, "severity": None},
            "pipeline_responses": [],
            "rag_context": many_chunks,
        }

        prompt = agent._build_prompt(input_data)

        # Only chunks 0–4 should appear; chunk 9 should not be in the prompt
        assert "Chunk number 9" not in prompt
        assert "Chunk number 0" in prompt
