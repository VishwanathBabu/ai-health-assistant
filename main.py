"""
main.py
=======
FastAPI entry point for the AI Health Assistant (Phase 2: RAG).

Endpoints:
  POST /chat                → multi-agent pipeline with RAG context
  GET  /health              → liveness probe
  GET  /health/ready        → readiness probe (LLM + Qdrant)
  POST /documents/upload    → upload a medical document (.txt / .pdf / .md)
  POST /documents/index     → re-index all documents in the uploads folder
  GET  /knowledge/search    → semantic search against the vector store
  GET  /docs                → Swagger UI
"""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.orchestrator import HealthAssistantOrchestrator as Orchestrator

# ── Constants ────────────────────────────────────────────────────────────────

SAFE_ERROR_RESPONSE = (
    "Something went wrong. Please try again or consult a medical professional."
)

UPLOADS_DIR = Path("documents/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("ai_health_assistant.api")

# ── Lifespan ─────────────────────────────────────────────────────────────────

_orchestrator: Orchestrator | None = None
_knowledge_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _knowledge_store

    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        logger.info("Loading config from: %s", env_path)
    else:
        logger.warning(
            ".env not found at %s — environment variables must be set manually.",
            env_path,
        )

    from core.config import settings, LLMProvider

    provider = settings.llm_provider.value

    if settings.llm_provider != LLMProvider.OLLAMA:
        key = settings.active_api_key
        if not key:
            raise RuntimeError(
                f"No API key found for provider '{provider}'. "
                f"Set the correct key in your .env file and restart."
            )
        logger.info(
            "API key loaded: %s...%s (%d chars)",
            key[:8], key[-4:], len(key),
        )
    else:
        logger.info("Using local Ollama backend (no API key required).")
        logger.info(
            "Ollama URL: %s | model: %s",
            settings.ollama_base_url,
            settings.ollama_model,
        )

    logger.info("LLM provider: %s | model: %s", provider, settings.active_model)

    # ── Phase 2: Qdrant / knowledge store ────────────────────────────────────
    try:
        from knowledge.store import KnowledgeStore
        _knowledge_store = KnowledgeStore(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
        )
        qdrant_ok = await _knowledge_store.ping()
        if qdrant_ok:
            await _knowledge_store.ensure_collection()
            logger.info(
                "Qdrant connected at %s — collection '%s' ready.",
                settings.qdrant_url,
                settings.qdrant_collection,
            )
        else:
            logger.warning(
                "Qdrant NOT reachable at %s. "
                "RAG features will be unavailable. Chat still works.",
                settings.qdrant_url,
            )
            _knowledge_store = None
    except Exception as exc:
        logger.warning("Qdrant init error: %s — RAG disabled.", exc)
        _knowledge_store = None

    logger.info("Initialising orchestrator and agents…")
    _orchestrator = Orchestrator(knowledge_store=_knowledge_store)
    logger.info("Orchestrator ready — all agents initialised.")

    yield

    logger.info("Shutting down AI Health Assistant.")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Health Assistant",
    description=(
        "A multi-agent AI health assistant with RAG (Phase 2).\n\n"
        "⚕️ **Medical disclaimer**: This system does **not** provide diagnosis "
        "or prescription advice. Always consult a qualified healthcare professional."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's health-related question or symptom description.",
        examples=["I have had a headache and fever for 2 days."],
    )
    use_rag: bool = Field(
        default=True,
        description="Set false to skip knowledge retrieval and use LLM only.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"message": "I have a sore throat and mild fever.", "use_rag": True}
        }
    }


class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        description="The final safe, validated response from the health assistant.",
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="Document sources used for this response (RAG).",
    )
    rag_active: bool = Field(
        default=False,
        description="Whether RAG retrieval was used for this response.",
    )


# ── Global exception handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=200,
        content={"response": SAFE_ERROR_RESPONSE, "sources_used": [], "rag_active": False},
    )


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Liveness probe")
async def health_check():
    """Returns 200 OK as long as the server process is running."""
    from core.config import settings
    return {
        "status": "ok",
        "service": "AI Health Assistant",
        "version": "2.0.0",
        "provider": settings.llm_provider.value,
        "model": settings.active_model,
        "rag_enabled": _knowledge_store is not None,
    }


@app.get(
    "/health/ready",
    tags=["System"],
    summary="Readiness probe — validates LLM and Qdrant connectivity",
)
async def readiness_check():
    """
    Tests both the LLM and Qdrant (if configured).
    Returns 200 if the LLM is reachable, 503 otherwise.
    """
    from core.config import settings, LLMProvider

    TEST_PROMPT = "Reply with the single word: ok"
    llm_ok = False
    llm_error = None

    try:
        if settings.llm_provider == LLMProvider.OPENAI:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.active_api_key)
            resp = await client.chat.completions.create(
                model=settings.active_model,
                max_tokens=5,
                temperature=0,
                messages=[{"role": "user", "content": TEST_PROMPT}],
            )
            _ = resp.choices[0].message.content or ""

        elif settings.llm_provider == LLMProvider.ANTHROPIC:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.active_api_key)
            resp = await client.messages.create(
                model=settings.active_model,
                max_tokens=5,
                temperature=0,
                messages=[{"role": "user", "content": TEST_PROMPT}],
            )

        else:
            import httpx
            url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
            payload = {
                "model": settings.ollama_model,
                "prompt": TEST_PROMPT,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 10},
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()

        llm_ok = True

    except Exception as exc:
        llm_error = str(exc)
        logger.exception("Readiness check: LLM unreachable: %s", exc)

    # Qdrant check
    qdrant_ok = False
    qdrant_info = {}
    if _knowledge_store is not None:
        qdrant_ok = await _knowledge_store.ping()
        if qdrant_ok:
            qdrant_info = await _knowledge_store.collection_info()

    payload = {
        "llm_provider": settings.llm_provider.value,
        "model": settings.active_model,
        "llm_reachable": llm_ok,
        "qdrant_reachable": qdrant_ok,
        "qdrant_collection": qdrant_info,
    }

    if not llm_ok:
        payload["status"] = "not_ready"
        payload["llm_error"] = llm_error
        return JSONResponse(status_code=503, content=payload)

    payload["status"] = "ready"
    return payload


# ── Document endpoints ────────────────────────────────────────────────────────

@app.post(
    "/documents/upload",
    tags=["Documents"],
    summary="Upload a medical document (.txt, .pdf, .md)",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the server. The document is saved to the uploads
    folder but NOT yet indexed. Call POST /documents/index to build the
    vector store after uploading.

    Supported formats: .txt, .pdf, .md
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": (
                    f"File type '{suffix}' is not supported. "
                    f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
                ),
            },
        )

    dest = UPLOADS_DIR / file.filename
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        size_kb = dest.stat().st_size / 1024
        logger.info("Uploaded '%s' (%.1f KB).", file.filename, size_kb)
        return {
            "status": "uploaded",
            "filename": file.filename,
            "size_kb": round(size_kb, 2),
            "message": "File saved. Call POST /documents/index to build the vector store.",
        }
    except Exception as exc:
        logger.exception("Upload failed for '%s': %s", file.filename, exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@app.post(
    "/documents/index",
    tags=["Documents"],
    summary="Index all uploaded documents into the vector store",
)
async def index_documents(recreate: bool = Query(default=False)):
    """
    Reads all files from the uploads folder and upserts them into Qdrant.

    Set `recreate=true` to wipe the existing collection and rebuild from scratch.
    Leave as `false` (default) to add/update without deleting existing chunks.

    This endpoint can take 30–120 seconds depending on the number of documents
    and whether the embedding model needs to be downloaded on first run.
    """
    if _knowledge_store is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": (
                    "Qdrant is not connected. "
                    "Start Qdrant and restart the server."
                ),
            },
        )

    try:
        from knowledge.ingestion import IngestionService
        svc = IngestionService(store=_knowledge_store)
        results = await svc.ingest_directory(
            directory=UPLOADS_DIR,
            recreate_collection=recreate,
        )
        total_chunks = sum(r["chunks_created"] for r in results)
        ok_count = sum(1 for r in results if r["status"] == "ok")

        return {
            "status": "ok",
            "files_processed": len(results),
            "files_succeeded": ok_count,
            "total_chunks_indexed": total_chunks,
            "recreated": recreate,
            "results": results,
        }
    except Exception as exc:
        logger.exception("Indexing failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


# ── Knowledge endpoints ───────────────────────────────────────────────────────

@app.get(
    "/knowledge/search",
    tags=["Knowledge"],
    summary="Semantic search over the medical knowledge base",
)
async def knowledge_search(
    q: str = Query(..., min_length=2, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0),
):
    """
    Run a semantic similarity search against indexed medical documents.
    Returns the most relevant chunks ranked by cosine similarity score.
    """
    if _knowledge_store is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "Qdrant is not connected.",
            },
        )

    try:
        results = await _knowledge_store.search(
            query=q, top_k=top_k, score_threshold=threshold
        )
        return {
            "query": q,
            "results_count": len(results),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Search failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Assistant"],
    summary="Send a health question to the AI assistant",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Runs the full multi-agent pipeline with optional RAG retrieval.

    If Qdrant is connected and use_rag=true (default), relevant medical
    knowledge is retrieved and injected into the pipeline context before
    the LLM generates a response.
    """
    clean_message = request.message.strip()

    logger.info("POST /chat — message: %.120s", clean_message)

    if not clean_message:
        return ChatResponse(
            response=(
                "Your message appears to be empty. "
                "Please describe your symptoms or question."
            ),
            sources_used=[],
            rag_active=False,
        )

    try:
        use_rag = request.use_rag and _knowledge_store is not None
        result: dict = await _orchestrator.process(
            user_input=clean_message,
            use_rag=use_rag,
        )

        router_trace = result.get("pipeline_trace", {}).get("router", {})
        if (
            router_trace.get("intent") == "unknown"
            and router_trace.get("agents_to_invoke") == ["safety_agent"]
            and router_trace.get("reasoning", "").startswith("Router failed")
        ):
            logger.error(
                "PIPELINE FAILURE: RouterAgent returned fallback. "
                "Check Ollama/API setup. Message: %.80s",
                clean_message,
            )

        return ChatResponse(
            response=result["final_response"],
            sources_used=result.get("sources_used", []),
            rag_active=result.get("rag_active", False),
        )

    except Exception:
        logger.exception("Unhandled pipeline error for message: %.80s", clean_message)
        return ChatResponse(
            response=SAFE_ERROR_RESPONSE,
            sources_used=[],
            rag_active=False,
        )
