"""
knowledge/chunker.py
====================
Splits raw document text into overlapping chunks suitable for embedding.

Design rules:
  - Pure Python — no AI or network calls
  - Deterministic output for the same input
  - Handles plain text and extracted PDF text
  - Returns typed dicts so the rest of the codebase has clear contracts
"""

from __future__ import annotations

import re
from typing import Any


# Default chunking parameters. Tuned for medical text.
# Chunk size of 512 characters catches most paragraph-length extracts.
# 64-character overlap preserves context across boundaries.
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64
MIN_CHUNK_LENGTH = 50  # Chunks shorter than this are dropped (e.g. empty pages)


def chunk_text(
    text: str,
    source: str,
    title: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping character-level chunks.

    Args:
        text:       Raw text content of the document.
        source:     Filename or identifier used for provenance tracking.
        title:      Document title (optional, stored in payload).
        chunk_size: Maximum characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of chunk dicts, each containing:
            text, source, title, section, chunk_index
    """
    # Normalise whitespace: collapse multiple blank lines, strip leading/trailing
    text = _normalise_whitespace(text)

    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            end = _find_break(text, start, end)

        raw_chunk = text[start:end].strip()

        if len(raw_chunk) >= MIN_CHUNK_LENGTH:
            # Detect section heading (heuristic: short line before chunk body)
            section = _extract_section(text, start)

            chunks.append(
                {
                    "text": raw_chunk,
                    "source": source,
                    "title": title,
                    "section": section,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

        # Move forward by chunk_size - overlap
        start += max(1, chunk_size - overlap)

    return chunks


def chunk_pdf_pages(
    pages: list[str],
    source: str,
    title: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Chunk a list of page-level strings (from PDF extraction).
    Each page is chunked independently so chunks never cross page boundaries.
    """
    all_chunks = []
    for page_index, page_text in enumerate(pages):
        page_chunks = chunk_text(
            text=page_text,
            source=source,
            title=title,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        # Tag chunks with their page number
        for chunk in page_chunks:
            chunk["section"] = chunk["section"] or f"Page {page_index + 1}"
        # Re-index chunk_index globally across pages
        for chunk in page_chunks:
            chunk["chunk_index"] = len(all_chunks) + chunk["chunk_index"]
        all_chunks.extend(page_chunks)

    return all_chunks


# ── Private helpers ──────────────────────────────────────────────────────────

def _normalise_whitespace(text: str) -> str:
    """Collapse runs of blank lines and strip outer whitespace."""
    # Reduce 3+ newlines to 2 (preserve paragraph structure)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs to single space (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _find_break(text: str, start: int, end: int) -> int:
    """
    Search backward from `end` for a good break point.
    Prefers paragraph > sentence > word boundaries.
    Falls back to `end` if nothing is found.
    """
    search_window = text[start:end]

    # Paragraph break
    idx = search_window.rfind("\n\n")
    if idx > 0:
        return start + idx + 2

    # Sentence end (. ! ?)
    for punct in (".", "!", "?"):
        idx = search_window.rfind(punct)
        if idx > chunk_size_fraction(len(search_window)):
            return start + idx + 1

    # Word boundary
    idx = search_window.rfind(" ")
    if idx > 0:
        return start + idx + 1

    return end


def chunk_size_fraction(length: int, fraction: float = 0.5) -> int:
    """Return `fraction` of length, used as a minimum break position."""
    return int(length * fraction)


def _extract_section(text: str, pos: int) -> str:
    """
    Heuristic: look backward up to 200 chars for a line that looks like a heading.
    A heading is a short line (< 80 chars) with no period at the end.
    """
    lookback = max(0, pos - 200)
    preceding = text[lookback:pos]
    lines = [l.strip() for l in preceding.split("\n") if l.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    if len(candidate) < 80 and not candidate.endswith("."):
        return candidate
    return ""
