"""
tests/test_ocr.py
=================
Tests for the OCR fallback pipeline in knowledge/ocr.py and knowledge/ingestion.py.

All tests use mocks — no real Tesseract, Poppler, or PDF files required.
Tests cover:
  1. Text layer detection (has_text_layer)
  2. OCR text cleaning (clean_ocr_text)
  3. OCR dispatch in ingestion (_process_pdf)
  4. OCR failure handling
  5. Mixed-mode ingestion (text PDF and image PDF in same directory)
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ════════════════════════════════════════════════════════════════════════════
# HAS_TEXT_LAYER TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestHasTextLayer:

    def test_page_with_sufficient_text_returns_true(self):
        """A page with 50+ non-whitespace characters is a text PDF."""
        from knowledge.ocr import has_text_layer

        pages = ["This page contains enough text to be considered a valid text layer."]
        assert has_text_layer(pages) is True

    def test_all_empty_pages_returns_false(self):
        """All-empty pages → image-based PDF, OCR needed."""
        from knowledge.ocr import has_text_layer

        pages = ["", "", "  \n  \t  "]
        assert has_text_layer(pages) is False

    def test_pages_below_min_threshold_returns_false(self):
        """Pages with very few chars (e.g. just a page number) → no text layer."""
        from knowledge.ocr import has_text_layer

        pages = ["42", "  ", "7"]  # Only page numbers
        assert has_text_layer(pages) is False

    def test_mixed_pages_one_with_text_returns_true(self):
        """If at least one page has enough text, the whole PDF counts as text-based."""
        from knowledge.ocr import has_text_layer

        pages = [
            "",  # empty page (cover)
            "This page has a detailed description of diabetes management protocols.",
            "",  # empty page (back cover)
        ]
        assert has_text_layer(pages) is True

    def test_empty_list_returns_false(self):
        """Edge case: no pages at all → False."""
        from knowledge.ocr import has_text_layer

        assert has_text_layer([]) is False

    def test_exactly_min_threshold_returns_true(self):
        """Exactly MIN_TEXT_CHARS characters on a page → counts as text."""
        from knowledge.ocr import has_text_layer, MIN_TEXT_CHARS

        pages = ["x" * MIN_TEXT_CHARS]
        assert has_text_layer(pages) is True

    def test_one_below_min_threshold_returns_false(self):
        """One character below MIN_TEXT_CHARS → not enough, treated as image."""
        from knowledge.ocr import has_text_layer, MIN_TEXT_CHARS

        pages = ["x" * (MIN_TEXT_CHARS - 1)]
        assert has_text_layer(pages) is False


# ════════════════════════════════════════════════════════════════════════════
# CLEAN_OCR_TEXT TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestCleanOcrText:

    def test_empty_string_returns_empty(self):
        """Empty input → empty output."""
        from knowledge.ocr import clean_ocr_text

        assert clean_ocr_text("") == ""

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces within a line are collapsed to one."""
        from knowledge.ocr import clean_ocr_text

        raw = "This   has   extra    spaces"
        cleaned = clean_ocr_text(raw)
        assert "  " not in cleaned

    def test_lone_page_number_removed(self):
        """A line with only a page number is removed."""
        from knowledge.ocr import clean_ocr_text

        raw = "Some content here.\n42\nMore content follows."
        cleaned = clean_ocr_text(raw)
        assert "\n42\n" not in cleaned
        assert "Some content here" in cleaned
        assert "More content follows" in cleaned

    def test_page_number_with_dashes_removed(self):
        """Page numbers formatted as '- 3 -' are removed."""
        from knowledge.ocr import clean_ocr_text

        raw = "Chapter text here.\n- 3 -\nNext chapter begins."
        cleaned = clean_ocr_text(raw)
        assert "- 3 -" not in cleaned

    def test_excessive_blank_lines_collapsed(self):
        """Three or more consecutive blank lines are collapsed to one."""
        from knowledge.ocr import clean_ocr_text

        raw = "Paragraph one.\n\n\n\n\nParagraph two."
        cleaned = clean_ocr_text(raw)
        assert "\n\n\n" not in cleaned
        assert "Paragraph one" in cleaned
        assert "Paragraph two" in cleaned

    def test_medical_content_preserved(self):
        """Real medical text must pass through unchanged except whitespace."""
        from knowledge.ocr import clean_ocr_text

        raw = (
            "Hypertension is defined as systolic blood pressure ≥ 140 mmHg "
            "or diastolic blood pressure ≥ 90 mmHg."
        )
        cleaned = clean_ocr_text(raw)
        assert "Hypertension" in cleaned
        assert "140 mmHg" in cleaned
        assert "90 mmHg" in cleaned

    def test_short_all_caps_line_removed(self):
        """Short all-caps lines (common headers/footers) are removed."""
        from knowledge.ocr import clean_ocr_text

        raw = "WORLD HEALTH ORGANIZATION\nDiabetes affects millions worldwide."
        cleaned = clean_ocr_text(raw)
        # The all-caps header line should be dropped
        assert "WORLD HEALTH ORGANIZATION" not in cleaned
        assert "Diabetes affects millions" in cleaned

    def test_long_all_caps_line_preserved(self):
        """Long all-caps lines (e.g. a title) should NOT be removed."""
        from knowledge.ocr import clean_ocr_text

        # 30+ characters, uppercase: should be preserved
        raw = "THIS IS A VERY LONG TITLE THAT EXCEEDS THIRTY CHARACTERS AND SHOULD STAY"
        cleaned = clean_ocr_text(raw)
        assert "THIS IS A VERY LONG TITLE" in cleaned

    def test_tabs_converted_to_spaces(self):
        """Tabs within lines are replaced with spaces."""
        from knowledge.ocr import clean_ocr_text

        raw = "Column A\tColumn B\tColumn C"
        cleaned = clean_ocr_text(raw)
        assert "\t" not in cleaned


# ════════════════════════════════════════════════════════════════════════════
# INGESTION _process_pdf TESTS (OCR PATH)
# ════════════════════════════════════════════════════════════════════════════

class TestIngestionPdfOcr:
    """
    Tests for the ingestion._process_pdf method.
    We mock pypdf and ocr_pdf so no real files or binaries are needed.
    """

    def _make_service(self):
        """Build IngestionService with a mocked store."""
        from knowledge.ingestion import IngestionService
        store = MagicMock()
        store.collection_name = "test"
        return IngestionService(store=store)

    def test_text_pdf_does_not_trigger_ocr(self):
        """
        If pypdf returns enough text, ocr_pdf must NOT be called.
        """
        svc = self._make_service()
        rich_text = "This page contains detailed medical information. " * 5

        mock_page = MagicMock()
        mock_page.extract_text.return_value = rich_text
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = Path(f.name)

        try:
            with patch("knowledge.ingestion.pypdf") as mock_pypdf, \
                 patch("knowledge.ingestion.ocr_pdf") as mock_ocr:
                mock_pypdf.PdfReader.return_value = mock_reader
                chunks, ocr_used = svc._process_pdf(tmp, tmp.name, "Test Doc")

            mock_ocr.assert_not_called()
            assert ocr_used is False
        finally:
            tmp.unlink(missing_ok=True)

    def test_image_pdf_triggers_ocr(self):
        """
        If pypdf returns empty pages, ocr_pdf MUST be called.
        """
        svc = self._make_service()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  # Empty — image PDF
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        ocr_text = "WHO recommends regular blood pressure monitoring. " * 10

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = Path(f.name)

        try:
            with patch("knowledge.ingestion.pypdf") as mock_pypdf, \
                 patch("knowledge.ingestion.ocr_pdf", return_value=[ocr_text]) as mock_ocr:
                mock_pypdf.PdfReader.return_value = mock_reader
                chunks, ocr_used = svc._process_pdf(tmp, tmp.name, "WHO Guide")

            mock_ocr.assert_called_once_with(tmp)
            assert ocr_used is True
            assert len(chunks) > 0
        finally:
            tmp.unlink(missing_ok=True)

    def test_ocr_failure_returns_empty_list_not_exception(self):
        """
        If OCR raises an exception, the method must return ([], False)
        and must NOT propagate the exception.
        """
        svc = self._make_service()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = Path(f.name)

        try:
            with patch("knowledge.ingestion.pypdf") as mock_pypdf, \
                 patch("knowledge.ingestion.ocr_pdf", side_effect=RuntimeError("Tesseract not found")):
                mock_pypdf.PdfReader.return_value = mock_reader
                chunks, ocr_used = svc._process_pdf(tmp, tmp.name, "Doc")

            assert chunks == []
            assert ocr_used is False
        finally:
            tmp.unlink(missing_ok=True)

    def test_pdf_with_partial_text_does_not_trigger_ocr(self):
        """
        A PDF where at least one page has enough text skips OCR,
        even if other pages are empty.
        """
        svc = self._make_service()

        empty_page = MagicMock()
        empty_page.extract_text.return_value = ""

        text_page = MagicMock()
        text_page.extract_text.return_value = (
            "Diabetes mellitus is a group of metabolic diseases. " * 3
        )

        mock_reader = MagicMock()
        mock_reader.pages = [empty_page, text_page, empty_page]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = Path(f.name)

        try:
            with patch("knowledge.ingestion.pypdf") as mock_pypdf, \
                 patch("knowledge.ingestion.ocr_pdf") as mock_ocr:
                mock_pypdf.PdfReader.return_value = mock_reader
                chunks, ocr_used = svc._process_pdf(tmp, tmp.name, "Partial")

            mock_ocr.assert_not_called()
            assert ocr_used is False
        finally:
            tmp.unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# FULL INGESTION FLOW TESTS (async)
# ════════════════════════════════════════════════════════════════════════════

class TestIngestionServiceOcr:

    @pytest.mark.asyncio
    async def test_image_pdf_ingested_successfully(self):
        """
        End-to-end: an image PDF goes through OCR and produces indexed chunks.
        ocr_used must be True in the result dict.
        """
        from knowledge.ingestion import IngestionService

        mock_store = AsyncMock()
        mock_store.collection_name = "test"
        mock_store.upsert_chunks.return_value = 5
        svc = IngestionService(store=mock_store)

        ocr_pages = ["WHO diabetes management guidelines state that... " * 10]

        empty_page = MagicMock()
        empty_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [empty_page]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = Path(f.name)

        try:
            with patch("knowledge.ingestion.pypdf") as mock_pypdf, \
                 patch("knowledge.ingestion.ocr_pdf", return_value=ocr_pages):
                mock_pypdf.PdfReader.return_value = mock_reader
                result = await svc.ingest_file(tmp)

            assert result["status"] == "ok"
            assert result["ocr_used"] is True
            assert result["chunks_created"] == 5
        finally:
            tmp.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_text_pdf_ocr_used_is_false(self):
        """
        Text-based PDF must produce ocr_used=False in result.
        """
        from knowledge.ingestion import IngestionService

        mock_store = AsyncMock()
        mock_store.collection_name = "test"
        mock_store.upsert_chunks.return_value = 3
        svc = IngestionService(store=mock_store)

        rich_page = MagicMock()
        rich_page.extract_text.return_value = (
            "Blood pressure measurement is critical. " * 10
        )
        mock_reader = MagicMock()
        mock_reader.pages = [rich_page]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = Path(f.name)

        try:
            with patch("knowledge.ingestion.pypdf") as mock_pypdf, \
                 patch("knowledge.ingestion.ocr_pdf") as mock_ocr:
                mock_pypdf.PdfReader.return_value = mock_reader
                result = await svc.ingest_file(tmp)

            mock_ocr.assert_not_called()
            assert result["ocr_used"] is False
        finally:
            tmp.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_ocr_failure_returns_error_result(self):
        """
        If OCR fails, ingest_file returns status='error' with a message,
        and must NOT raise an exception.
        """
        from knowledge.ingestion import IngestionService

        mock_store = AsyncMock()
        svc = IngestionService(store=mock_store)

        empty_page = MagicMock()
        empty_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [empty_page]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = Path(f.name)

        try:
            with patch("knowledge.ingestion.pypdf") as mock_pypdf, \
                 patch("knowledge.ingestion.ocr_pdf", side_effect=RuntimeError("Tesseract not found")):
                mock_pypdf.PdfReader.return_value = mock_reader
                result = await svc.ingest_file(tmp)

            # Must not crash; must return a warning/error with 0 chunks
            assert result["chunks_created"] == 0
            assert result["status"] in ("warning", "error")
        finally:
            tmp.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_directory_with_mixed_pdfs(self):
        """
        A directory with one text PDF and one image PDF must process both
        correctly, with different ocr_used values.
        """
        from knowledge.ingestion import IngestionService

        mock_store = AsyncMock()
        mock_store.collection_name = "test"
        mock_store.upsert_chunks.return_value = 4
        mock_store.ensure_collection = AsyncMock()
        svc = IngestionService(store=mock_store)

        rich_page = MagicMock()
        rich_page.extract_text.return_value = "Rich text content about health. " * 5
        empty_page = MagicMock()
        empty_page.extract_text.return_value = ""

        call_count = 0

        def make_reader(path, *args, **kwargs):
            reader = MagicMock()
            # Files are processed in alphabetical order: image_doc.pdf first, text_doc.pdf second
            if "image_doc" in str(path):
                reader.pages = [empty_page]
            else:
                reader.pages = [rich_page]
            return reader

        ocr_pages = ["OCR content from scanned WHO document. " * 10]

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "text_doc.pdf").write_bytes(b"%PDF fake")
            Path(tmpdir, "image_doc.pdf").write_bytes(b"%PDF fake")

            with patch("knowledge.ingestion.pypdf") as mock_pypdf, \
                 patch("knowledge.ingestion.ocr_pdf", return_value=ocr_pages):
                mock_pypdf.PdfReader.side_effect = make_reader
                results = await svc.ingest_directory(tmpdir)

        assert len(results) == 2
        ocr_flags = {r["source"]: r.get("ocr_used") for r in results}
        assert ocr_flags.get("image_doc.pdf") is True
        assert ocr_flags.get("text_doc.pdf") is False


# ════════════════════════════════════════════════════════════════════════════
# OCR MODULE UNIT TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestOcrModule:

    def test_ocr_pdf_calls_pytesseract_per_page(self):
        """ocr_pdf must call pytesseract.image_to_string once per PDF page."""
        mock_pytesseract = MagicMock()
        mock_pytesseract.get_tesseract_version.return_value = "5.3.0"
        mock_pytesseract.image_to_string.return_value = "WHO recommends vaccination."

        mock_images = [MagicMock(), MagicMock()]  # 2-page PDF

        with patch("knowledge.ocr._import_pytesseract", return_value=mock_pytesseract), \
             patch("knowledge.ocr._import_pdf2image", return_value=lambda *a, **k: mock_images), \
             patch("knowledge.ocr._verify_tesseract"):
            from knowledge.ocr import ocr_pdf
            pages = ocr_pdf("/tmp/fake.pdf")

        assert len(pages) == 2
        assert mock_pytesseract.image_to_string.call_count == 2

    def test_ocr_pdf_page_failure_returns_empty_string(self):
        """
        If OCR fails on a single page, that page becomes '' and the rest continue.
        """
        mock_pytesseract = MagicMock()
        mock_pytesseract.get_tesseract_version.return_value = "5.3.0"
        # Page 1 succeeds, page 2 raises
        mock_pytesseract.image_to_string.side_effect = [
            "Valid OCR text from page one.",
            Exception("Page 2 OCR error"),
        ]

        mock_images = [MagicMock(), MagicMock()]

        with patch("knowledge.ocr._import_pytesseract", return_value=mock_pytesseract), \
             patch("knowledge.ocr._import_pdf2image", return_value=lambda *a, **k: mock_images), \
             patch("knowledge.ocr._verify_tesseract"):
            from knowledge.ocr import ocr_pdf
            pages = ocr_pdf("/tmp/fake.pdf")

        assert len(pages) == 2
        assert "Valid OCR text" in pages[0]
        assert pages[1] == ""  # Failed page is empty string, not an exception

    def test_ocr_pdf_raises_when_pytesseract_missing(self):
        """If pytesseract is not installed, ocr_pdf raises RuntimeError."""
        with patch("knowledge.ocr._import_pytesseract", return_value=None):
            from knowledge.ocr import ocr_pdf
            with pytest.raises(RuntimeError, match="pytesseract"):
                ocr_pdf("/tmp/fake.pdf")

    def test_ocr_pdf_raises_when_pdf2image_missing(self):
        """If pdf2image is not installed, ocr_pdf raises RuntimeError."""
        mock_pytesseract = MagicMock()
        with patch("knowledge.ocr._import_pytesseract", return_value=mock_pytesseract), \
             patch("knowledge.ocr._import_pdf2image", return_value=None):
            from knowledge.ocr import ocr_pdf
            with pytest.raises(RuntimeError, match="pdf2image"):
                ocr_pdf("/tmp/fake.pdf")
