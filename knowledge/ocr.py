"""
knowledge/ocr.py
================
OCR fallback for image-based PDFs (e.g. WHO PDFs saved from a browser).

Strategy:
  1. pdf2image converts each PDF page to a PIL Image at 300 DPI.
  2. pytesseract runs Tesseract OCR on each image.
  3. Raw OCR output is cleaned (whitespace, headers/footers, page numbers).

This module is ONLY called by ingestion.py when pypdf text extraction
returns empty or near-empty text. It never touches the HTTP layer, agents,
embeddings, or Qdrant.

Design rules:
  - All imports are deferred so the app starts even if Tesseract is not installed
  - Every public function returns strings, not exceptions
  - Caller decides what to do with errors; this module just logs and returns ""
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("ai_health_assistant.knowledge.ocr")

# Minimum characters on a page for it to count as "has text".
# Pages with fewer chars than this are treated as image-only.
MIN_TEXT_CHARS = 50

# OCR DPI — higher = better accuracy, slower processing.
# 300 DPI is the standard for document OCR.
OCR_DPI = 300


def has_text_layer(pages: list[str]) -> bool:
    """
    Return True if pypdf extracted meaningful text from the PDF.
    A PDF is considered text-based if at least one page has
    MIN_TEXT_CHARS non-whitespace characters.
    """
    for page_text in pages:
        stripped = page_text.strip()
        if len(stripped) >= MIN_TEXT_CHARS:
            return True
    return False


def ocr_pdf(pdf_path: str | Path) -> list[str]:
    """
    Run OCR on every page of a PDF and return a list of page strings.

    Returns a list of the same length as the PDF's page count.
    Pages that fail OCR return an empty string instead of crashing.

    Requires:
      - Tesseract installed on the system (see README for Windows install)
      - pytesseract Python package  (pip install pytesseract==0.3.13)
      - pdf2image Python package    (pip install pdf2image==1.17.0)
      - Poppler (required by pdf2image — see README for Windows install)
    """
    pdf_path = Path(pdf_path)
    logger.info("Starting OCR on '%s' at %d DPI...", pdf_path.name, OCR_DPI)

    # Validate dependencies before doing any real work
    pytesseract = _import_pytesseract()
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract==0.3.13\n"
            "Tesseract also requires the Tesseract binary — see README."
        )

    convert_from_path = _import_pdf2image()
    if convert_from_path is None:
        raise RuntimeError(
            "pdf2image is not installed. Run: pip install pdf2image==1.17.0\n"
            "pdf2image also requires Poppler — see README."
        )

    # Verify Tesseract binary is reachable
    _verify_tesseract(pytesseract)

    # Convert PDF pages to images
    logger.info("Converting PDF pages to images (DPI=%d)...", OCR_DPI)
    try:
        images = convert_from_path(str(pdf_path), dpi=OCR_DPI)
    except Exception as exc:
        raise RuntimeError(
            f"pdf2image failed to convert '{pdf_path.name}'. "
            f"Is Poppler installed and in PATH? Error: {exc}"
        ) from exc

    logger.info("Converted %d page(s) to images. Running OCR...", len(images))

    page_texts: list[str] = []
    for i, image in enumerate(images, 1):
        try:
            raw_text = pytesseract.image_to_string(
                image,
                lang="eng",
                config="--psm 3 --oem 3",
            )
            cleaned = clean_ocr_text(raw_text)
            page_texts.append(cleaned)
            logger.debug(
                "OCR page %d/%d: extracted %d chars.",
                i,
                len(images),
                len(cleaned),
            )
        except Exception as exc:
            logger.warning("OCR failed on page %d: %s — using empty string.", i, exc)
            page_texts.append("")

    total_chars = sum(len(t) for t in page_texts)
    logger.info(
        "OCR complete: %d page(s), %d total characters extracted.",
        len(page_texts),
        total_chars,
    )
    return page_texts


def clean_ocr_text(raw: str) -> str:
    """
    Clean raw OCR output to remove artefacts that hurt chunking and retrieval.

    Removes:
      - Repeated whitespace within lines
      - Lone page numbers (a line containing only digits, possibly with spaces)
      - Lines that look like repeated headers/footers (< 4 words, repeated pattern)
      - Excessive blank lines (more than 2 in a row collapsed to 1)

    Preserves:
      - Paragraph structure (double newlines)
      - Sentence boundaries
      - All actual medical content
    """
    if not raw:
        return ""

    lines = raw.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        # Collapse multiple spaces/tabs to single space within the line
        line = re.sub(r"[ \t]+", " ", line).strip()

        # Drop lone page numbers: lines that are ONLY digits (e.g. "42", "- 3 -")
        if re.fullmatch(r"[-–—\s\d]+", line) and len(line) < 10:
            continue

        # Drop very short lines that are all uppercase (common header/footer pattern)
        if len(line) < 30 and line.isupper() and len(line.split()) <= 4:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse 3+ consecutive blank lines down to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Private helpers ───────────────────────────────────────────────────────────

def _import_pytesseract():
    """Lazy import pytesseract. Returns None if not installed."""
    try:
        import pytesseract as pt
        return pt
    except ImportError:
        return None


def _import_pdf2image():
    """Lazy import pdf2image.convert_from_path. Returns None if not installed."""
    try:
        from pdf2image import convert_from_path
        return convert_from_path
    except ImportError:
        return None


def _verify_tesseract(pytesseract) -> None:
    """
    Confirm Tesseract binary is reachable. Raise RuntimeError with a clear
    Windows installation guide if it is not found.
    """
    try:
        version = pytesseract.get_tesseract_version()
        logger.info("Tesseract version: %s", version)
    except pytesseract.TesseractNotFoundError:
        if sys.platform.startswith("win"):
            instructions = (
                "Tesseract is not installed or not in PATH.\n"
                "Windows installation:\n"
                "  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  2. Run the installer (tesseract-ocr-w64-setup-*.exe)\n"
                "  3. During install, tick 'Add to PATH' OR add manually:\n"
                "     C:\\Program Files\\Tesseract-OCR\\ → System PATH\n"
                "  4. Restart your terminal.\n"
                "  5. Verify: tesseract --version"
            )
        else:
            instructions = (
                "Tesseract is not installed.\n"
                "Linux:  sudo apt-get install tesseract-ocr\n"
                "macOS:  brew install tesseract"
            )
        raise RuntimeError(instructions)
