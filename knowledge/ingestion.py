"""
knowledge/ingestion.py
======================
Ingestion pipeline: raw file → chunks → Qdrant.

Supported file types:
  - .txt   (plain text)
  - .pdf   (text-based via pypdf; image-based via OCR fallback)
  - .md    (markdown treated as plain text)

PDF processing strategy:
  1. Try pypdf text extraction (fast, no dependencies).
  2. If the extracted text is empty or too short to be useful,
     automatically fall back to OCR (pytesseract + pdf2image).
  3. Clean the text, chunk it, embed it, and store in Qdrant.

OCR only runs when normal extraction fails. Text-based PDFs are never
sent through OCR, so there is no performance cost for normal files.

Design rules:
  - This module never touches the HTTP layer
  - All disk I/O is synchronous (files are small)
  - The KnowledgeStore is passed in (dependency injection; easier to test)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import pypdf  # noqa: F401  — imported at module level so tests can mock it
except ImportError:
    pypdf = None  # type: ignore[assignment]

from knowledge.chunker import chunk_text, chunk_pdf_pages
from knowledge.ocr import has_text_layer, ocr_pdf

logger = logging.getLogger("ai_health_assistant.knowledge.ingestion")

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}


class IngestionService:
    """
    Orchestrates the full document → vector pipeline.

    Usage:
        store = KnowledgeStore(...)
        await store.ensure_collection()
        svc = IngestionService(store)
        result = await svc.ingest_file("/path/to/document.pdf")
    """

    def __init__(self, store) -> None:
        self.store = store

    async def ingest_file(
        self,
        file_path: str | Path,
        title: str = "",
        recreate_collection: bool = False,
    ) -> dict[str, Any]:
        """
        Process a single file and upsert its chunks into Qdrant.

        Returns a summary dict:
            { source, title, chunks_created, status, error, ocr_used }
        """
        path = Path(file_path)
        source = path.name
        title = title or path.stem.replace("_", " ").replace("-", " ").title()

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return {
                "source": source,
                "title": title,
                "chunks_created": 0,
                "status": "error",
                "error": (
                    f"Unsupported file type '{path.suffix}'. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                ),
                "ocr_used": False,
            }

        if not path.exists():
            return {
                "source": source,
                "title": title,
                "chunks_created": 0,
                "status": "error",
                "error": f"File not found: {file_path}",
                "ocr_used": False,
            }

        try:
            ocr_used = False

            if path.suffix.lower() == ".pdf":
                chunks, ocr_used = self._process_pdf(path, source, title)
            else:
                chunks = self._process_text(path, source, title)

            if not chunks:
                return {
                    "source": source,
                    "title": title,
                    "chunks_created": 0,
                    "status": "warning",
                    "error": "File produced no usable chunks (too short or empty).",
                    "ocr_used": ocr_used,
                }

            n = await self.store.upsert_chunks(chunks)

            logger.info(
                "Stored %d vectors in Qdrant for '%s' (OCR used: %s).",
                n,
                source,
                ocr_used,
            )
            return {
                "source": source,
                "title": title,
                "chunks_created": n,
                "status": "ok",
                "error": None,
                "ocr_used": ocr_used,
            }

        except Exception as exc:
            logger.exception("Ingestion failed for '%s': %s", source, exc)
            return {
                "source": source,
                "title": title,
                "chunks_created": 0,
                "status": "error",
                "error": str(exc),
                "ocr_used": False,
            }

    async def ingest_directory(
        self,
        directory: str | Path,
        recreate_collection: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Ingest all supported files in a directory.
        Returns a list of per-file result dicts.
        """
        directory = Path(directory)
        if recreate_collection:
            await self.store.ensure_collection(recreate=True)
        else:
            await self.store.ensure_collection(recreate=False)

        results = []
        for file_path in sorted(directory.iterdir()):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                result = await self.ingest_file(file_path)
                results.append(result)
                logger.info(
                    "Ingested %s: %d chunks (%s, OCR: %s)",
                    file_path.name,
                    result["chunks_created"],
                    result["status"],
                    result.get("ocr_used", False),
                )

        total_chunks = sum(r["chunks_created"] for r in results)
        logger.info(
            "Directory ingestion complete: %d files, %d chunks total.",
            len(results),
            total_chunks,
        )
        return results

    # ── Private ───────────────────────────────────────────────────────────────

    def _process_text(self, path: Path, source: str, title: str) -> list[dict]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return chunk_text(text=text, source=source, title=title)

    def _process_pdf(
        self, path: Path, source: str, title: str
    ) -> tuple[list[dict], bool]:
        """
        Process a PDF file. Returns (chunks, ocr_used).

        Step 1: Try pypdf text extraction.
        Step 2: If text layer is absent or too thin, fall back to OCR.
        Step 3: Chunk whichever text we have.
        """
        import knowledge.ingestion as _self_module  # noqa: F401

        # ── Step 1: pypdf text extraction ─────────────────────────────────────
        logger.info("Extracting PDF text from '%s'...", source)

        _pypdf = pypdf  # use module-level import (allows test mocking)
        if _pypdf is None:
            raise ImportError(
                "pypdf is not installed. Run: pip install pypdf==4.2.0"
            )

        reader = _pypdf.PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")

        # ── Step 2: OCR fallback if text layer is empty ───────────────────────
        ocr_used = False
        if not has_text_layer(pages):
            logger.info(
                "No text layer detected in '%s'. Falling back to OCR...", source
            )
            try:
                pages = ocr_pdf(path)
                ocr_used = True
                logger.info("OCR complete for '%s'.", source)
            except Exception as exc:
                logger.error(
                    "OCR failed for '%s': %s. "
                    "Returning empty result — file will not be indexed.",
                    source,
                    exc,
                )
                return [], False
        else:
            logger.info("Text layer found in '%s'. Skipping OCR.", source)

        # ── Step 3: Chunk the text (same path for both text and OCR) ──────────
        chunks = chunk_pdf_pages(pages=pages, source=source, title=title)
        logger.info("Created %d chunks from '%s'.", len(chunks), source)
        return chunks, ocr_used
