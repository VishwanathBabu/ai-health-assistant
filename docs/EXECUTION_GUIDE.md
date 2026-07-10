# AI Health Assistant — Phase 2

A multi-agent AI health assistant with **RAG (Retrieval-Augmented Generation)** built on FastAPI, Ollama, and Qdrant.

---

## What's New in Phase 2

Phase 1 gave you a multi-agent pipeline (Router → Symptom → Emergency → Safety). Phase 2 adds:

- **Document upload** — POST `/documents/upload` accepts `.txt`, `.pdf`, `.md`
- **Vector indexing** — POST `/documents/index` chunks files and stores them in Qdrant
- **Semantic search** — GET `/knowledge/search` retrieves relevant medical passages
- **RAG in chat** — POST `/chat` retrieves relevant knowledge before generating responses
- **Qdrant health check** — `/health/ready` now also verifies Qdrant connectivity

The chat endpoint still works with `use_rag=false` if Qdrant is unavailable.

---

## Architecture

```
POST /chat
    │
    ├── KnowledgeStore.search()   ← Phase 2: retrieve relevant chunks from Qdrant
    │
    ├── RouterAgent               ← classify intent
    │
    ├── SymptomAgent (if routed)  ← extract symptom structure
    │
    ├── EmergencyAgent (always)   ← detect emergencies
    │
    └── SafetyAgent (always last) ← compose final response + inject RAG context
```

Document pipeline:

```
POST /documents/upload  →  saves file to documents/uploads/
POST /documents/index   →  chunker → sentence-transformers → Qdrant upsert
GET  /knowledge/search  →  query → embed → Qdrant search → ranked chunks
```

---

## ⚕️ Medical Disclaimer

This system does **not** provide medical diagnosis, prescription advice, or treatment plans. Always consult a qualified healthcare professional.

---

# Complete Execution Guide

## Step 1 — Project Setup

**Python version required: 3.10 or higher**

Verify Python version:
```
python --version
```
Expected output:
```
Python 3.10.x  (or 3.11.x, 3.12.x)
```

If Python is missing, download from https://www.python.org/downloads/ and tick "Add Python to PATH" during installation.

Navigate to the project folder:
```
cd ai_health_assistant
```

Create a virtual environment:
```
python -m venv venv
```

Activate the virtual environment (Windows):
```
venv\Scripts\activate
```

Your terminal prompt should now show `(venv)` on the left.

Upgrade pip:
```
python -m pip install --upgrade pip
```

Install all dependencies:
```
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, Qdrant client, sentence-transformers (~500 MB total), pypdf, and all test dependencies.

---

## Step 2 — Ollama

**Check if Ollama is installed:**
```
ollama --version
```

**If Ollama is not installed:**

Go to https://ollama.com/download, download the Windows installer, and run it. Ollama installs as a background service.

**Pull the required model:**
```
ollama pull llama3
```

This downloads approximately 4.7 GB. Wait for it to finish.

**Verify the model exists:**
```
ollama list
```

Expected output:
```
NAME               ID              SIZE    MODIFIED
llama3:latest      365c0bd3c000    4.7 GB  2 minutes ago
```

**Verify Ollama is running:**
```
curl http://localhost:11434/api/generate -d "{\"model\":\"llama3\",\"prompt\":\"say ok\",\"stream\":false}"
```

Expected output (something like):
```json
{"model":"llama3","response":"ok","done":true,...}
```

If you get "connection refused", start Ollama manually:
```
ollama serve
```

---

## Step 3 — Qdrant

Qdrant does **not** require Docker. Use the native Windows binary.

**Download Qdrant:**

Go to https://github.com/qdrant/qdrant/releases and download the latest `qdrant-x86_64-pc-windows-msvc.zip`.

**Extract and run:**
```
cd C:\qdrant
qdrant.exe
```

Or with explicit config path:
```
qdrant.exe --config-path config.yaml
```

**Verify Qdrant is running:**

Open a new terminal and run:
```
curl http://localhost:6333/healthz
```

Expected output:
```
"healthz check passed"
```

Also verify via the dashboard: open http://localhost:6333/dashboard in your browser. You should see the Qdrant UI with an empty collections list.

**Alternative — Docker (if you already have Docker Desktop):**
```
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v %cd%\qdrant_storage:/qdrant/storage qdrant/qdrant
```

Expected Docker output:
```
...
Qdrant gRPC listening on 6334
Qdrant HTTP listening on 6333
```

---

## Step 4 — Environment Variables

The `.env` file lives in the **project root** alongside `main.py`:

```
ai_health_assistant/
├── .env          ← here
├── main.py
├── requirements.txt
└── ...
```

**Every variable explained:**

| Variable | Default | Modify? | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | `ollama` | Yes | Which LLM: `ollama`, `openai`, `anthropic` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | No (unless custom port) | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Yes | Any model you've pulled |
| `OPENAI_API_KEY` | _(blank)_ | Yes | Required if using OpenAI |
| `OPENAI_MODEL` | `gpt-4o` | Yes | OpenAI model name |
| `ANTHROPIC_API_KEY` | _(blank)_ | Yes | Required if using Anthropic |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Yes | Anthropic model name |
| `AGENT_TEMPERATURE` | `0.1` | Yes | 0.0 = deterministic, 1.0 = creative |
| `AGENT_MAX_TOKENS` | `1024` | Yes | Max response length per agent call |
| `AGENT_TIMEOUT_SECONDS` | `60` | Yes | Per-agent timeout |
| `LOG_LEVEL` | `INFO` | Yes | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `console` | Yes | `console` or `json` |
| `API_HOST` | `0.0.0.0` | No | Listen on all interfaces |
| `API_PORT` | `8000` | Yes | HTTP port |
| `API_RELOAD` | `true` | No (set false for production) | Hot reload |
| `QDRANT_URL` | `http://localhost:6333` | No (unless custom host) | Qdrant URL |
| `QDRANT_COLLECTION` | `medical_docs` | No | Vector collection name |

**Do not change** `QDRANT_URL`, `QDRANT_COLLECTION`, `API_HOST`, or `OLLAMA_BASE_URL` unless you have a specific reason (e.g. remote Qdrant server).

---

## Step 5 — Document Ingestion

Place your medical documents in the `documents/uploads/` folder.

**Supported formats:** `.txt`, `.pdf`, `.md`

**Option A — Copy files manually:**
```
copy C:\my_docs\diabetes_guide.pdf ai_health_assistant\documents\uploads\
copy C:\my_docs\fever_treatment.txt ai_health_assistant\documents\uploads\
```

**Option B — Upload via API (server must be running first):**
```
curl -X POST http://localhost:8000/documents/upload -F "file=@C:\my_docs\diabetes_guide.pdf"
```

**Build embeddings and create the vector database:**
```
curl -X POST "http://localhost:8000/documents/index"
```

To wipe the collection and re-index from scratch:
```
curl -X POST "http://localhost:8000/documents/index?recreate=true"
```

**Expected response:**
```json
{
  "status": "ok",
  "files_processed": 2,
  "files_succeeded": 2,
  "total_chunks_indexed": 47,
  "recreated": false,
  "results": [
    {"source": "diabetes_guide.pdf", "chunks_created": 28, "status": "ok"},
    {"source": "fever_treatment.txt", "chunks_created": 19, "status": "ok"}
  ]
}
```

On first run, the sentence-transformers model (`all-MiniLM-L6-v2`, ~80 MB) downloads automatically. Allow 1–2 minutes.

**Verify indexing:**
```
curl "http://localhost:8000/knowledge/search?q=fever+symptoms&top_k=3"
```

If the response contains `results` with non-empty text, indexing succeeded.

---

## Step 6 — Start the Backend

Make sure your terminal is in the project root (where `main.py` lives) and the venv is active.

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Successful startup log:**
```
INFO  Loading config from: ..\.env
INFO  Using local Ollama backend (no API key required).
INFO  Ollama URL: http://localhost:11434 | model: llama3
INFO  LLM provider: ollama | model: llama3
INFO  Qdrant connected at http://localhost:6333 — collection 'medical_docs' ready.
INFO  Initialising orchestrator and agents…
INFO  Orchestrator ready — all agents initialised.
INFO  Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

If Qdrant is not running, you will see:
```
WARNING  Qdrant NOT reachable at http://localhost:6333. RAG features will be unavailable. Chat still works.
```

The server starts regardless. RAG endpoints return 503 but `/chat` still works.

---

## Step 7 — API Verification

### GET /health

**curl:**
```
curl http://localhost:8000/health
```

**Postman:** GET `http://localhost:8000/health`

**Expected response:**
```json
{
  "status": "ok",
  "service": "AI Health Assistant",
  "version": "2.0.0",
  "provider": "ollama",
  "model": "llama3",
  "rag_enabled": true
}
```

---

### GET /health/ready

**curl:**
```
curl http://localhost:8000/health/ready
```

**Postman:** GET `http://localhost:8000/health/ready`

**Expected response (all systems up):**
```json
{
  "status": "ready",
  "llm_provider": "ollama",
  "model": "llama3",
  "llm_reachable": true,
  "qdrant_reachable": true,
  "qdrant_collection": {
    "name": "medical_docs",
    "vectors_count": 47,
    "points_count": 47,
    "status": "green"
  }
}
```

**If Qdrant is down**, `qdrant_reachable` is `false` but status is still `200` as long as the LLM works.
**If Ollama is down**, response is `503`.

---

### POST /documents/upload

**curl:**
```
curl -X POST http://localhost:8000/documents/upload -F "file=@C:\path\to\diabetes.pdf"
```

**Postman:** POST `http://localhost:8000/documents/upload` → Body → form-data → key `file` (type File) → select file

**Expected response:**
```json
{
  "status": "uploaded",
  "filename": "diabetes.pdf",
  "size_kb": 142.5,
  "message": "File saved. Call POST /documents/index to build the vector store."
}
```

---

### POST /documents/index

**curl:**
```
curl -X POST "http://localhost:8000/documents/index"
```

**With recreate (full re-index):**
```
curl -X POST "http://localhost:8000/documents/index?recreate=true"
```

**Postman:** POST `http://localhost:8000/documents/index` (no body needed)

**Expected response:**
```json
{
  "status": "ok",
  "files_processed": 1,
  "files_succeeded": 1,
  "total_chunks_indexed": 28,
  "recreated": false
}
```

---

### GET /knowledge/search

**curl:**
```
curl "http://localhost:8000/knowledge/search?q=diabetes+treatment&top_k=3"
```

**Postman:** GET `http://localhost:8000/knowledge/search` → Params: `q=diabetes treatment`, `top_k=3`

**Expected response:**
```json
{
  "query": "diabetes treatment",
  "results_count": 3,
  "results": [
    {
      "text": "Type 2 diabetes is managed through...",
      "source": "diabetes_guide.pdf",
      "title": "Diabetes Guide",
      "section": "Treatment",
      "chunk_index": 5,
      "score": 0.8732
    }
  ]
}
```

---

### POST /chat

**curl:**
```
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"I have had a headache and fever for 2 days\", \"use_rag\": true}"
```

**Postman:** POST `http://localhost:8000/chat` → Body → raw JSON:
```json
{
  "message": "I have had a headache and fever for 2 days",
  "use_rag": true
}
```

**Expected response:**
```json
{
  "response": "Thank you for sharing how you're feeling...\n\n---\n⚕️ Medical Disclaimer...",
  "sources_used": ["fever_treatment.txt"],
  "rag_active": true
}
```

---

## Step 8 — Testing

Run the full test suite (Phase 1 + Phase 2):
```
pytest tests/ -v
```

Run only Phase 1 tests:
```
pytest tests/test_mvp.py -v
```

Run only Phase 2 tests:
```
pytest tests/test_phase2.py -v
```

Run with coverage:
```
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Successful output looks like:**
```
tests/test_mvp.py::TestRouterAgent::test_normal_symptom_query PASSED
tests/test_mvp.py::TestRouterAgent::test_emergency_routing PASSED
...
tests/test_phase2.py::TestChunker::test_basic_chunking_returns_chunks PASSED
tests/test_phase2.py::TestKnowledgeStore::test_ping_returns_true_when_qdrant_up PASSED
...
========== 35 passed in 4.21s ==========
```

All tests use mocked LLM and Qdrant calls. No real Ollama or Qdrant server is needed to run tests.

---

## Step 9 — Troubleshooting

---

**Problem:** `Connection refused` when calling `/health/ready` or `/chat`
**Cause:** FastAPI server is not running.
**Solution:** Run `uvicorn main:app --reload --host 0.0.0.0 --port 8000` in the project root with the venv active.

---

**Problem:** `Ollama not running` — readiness check returns `503`
**Cause:** Ollama process stopped.
**Solution:** Open a new terminal and run `ollama serve`. Then retry `/health/ready`.

---

**Problem:** `model not found` error from Ollama
**Cause:** The model was never pulled, or was pulled with a different tag.
**Solution:** Run `ollama pull llama3`. Verify with `ollama list`.

---

**Problem:** `Qdrant NOT reachable` warning in startup log
**Cause:** Qdrant is not running on port 6333.
**Solution:** Start Qdrant with `qdrant.exe` (or `docker run qdrant/qdrant`). Re-run `curl http://localhost:6333/healthz` to confirm it's up, then restart FastAPI.

---

**Problem:** Port 8000 already in use — `ERROR: [Errno 10048] Only one usage of each socket address`
**Cause:** Another process is already on port 8000.
**Solution (find and kill):**
```
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```
Or start on a different port:
```
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

---

**Problem:** `No module named 'fastapi'` or similar import error
**Cause:** Dependencies not installed, or wrong Python environment active.
**Solution:**
```
venv\Scripts\activate
pip install -r requirements.txt
```

---

**Problem:** `.env not found` warning in startup log
**Cause:** You are running `uvicorn` from the wrong directory.
**Solution:** Always `cd ai_health_assistant` first, then run `uvicorn main:app`.

---

**Problem:** Embedding model download is very slow or fails
**Cause:** First run downloads `all-MiniLM-L6-v2` (~80 MB) from Hugging Face.
**Solution:** Ensure you have internet access. The download happens once. After that, the model is cached in `~/.cache/huggingface/`.

---

**Problem:** FastAPI startup fails with `RuntimeError: No API key found`
**Cause:** `LLM_PROVIDER` is set to `openai` or `anthropic` but no key is provided.
**Solution:** Either set the API key in `.env`, or set `LLM_PROVIDER=ollama`.

---

**Problem:** `qdrant-client` or `sentence-transformers` not found
**Cause:** Dependencies were not installed (they were commented out in the old requirements.txt).
**Solution:** `pip install -r requirements.txt` — both packages are now active in Phase 2.

---

**Problem:** Vector database connection error during `/documents/index`
**Cause:** Qdrant went down after server startup.
**Solution:** Restart Qdrant, then call `/documents/index` again. Existing chunks are not lost (Qdrant persists to disk).

---

**Problem:** `pytest` fails with `ModuleNotFoundError`
**Cause:** Tests are not being run from the project root, or the venv is not active.
**Solution:**
```
cd ai_health_assistant
venv\Scripts\activate
pytest tests/ -v
```

---

## Step 10 — Final Verification Checklist

Work through each item in order. Only move to the next item after the current one passes.

```
✓ Ollama is running
  → ollama list  (shows llama3:latest)
  → curl http://localhost:11434/api/generate -d "{\"model\":\"llama3\",\"prompt\":\"say ok\",\"stream\":false}"

✓ Qdrant is running
  → curl http://localhost:6333/healthz  (returns: "healthz check passed")

✓ FastAPI starts successfully
  → uvicorn main:app --reload --host 0.0.0.0 --port 8000
  → Look for: "Orchestrator ready — all agents initialised."
  → Look for: "Qdrant connected at http://localhost:6333"

✓ /health works
  → curl http://localhost:8000/health
  → status == "ok", rag_enabled == true

✓ /health/ready works
  → curl http://localhost:8000/health/ready
  → status == "ready", llm_reachable == true, qdrant_reachable == true

✓ Documents upload successfully
  → curl -X POST http://localhost:8000/documents/upload -F "file=@your_doc.txt"
  → status == "uploaded"

✓ Documents are indexed
  → curl -X POST http://localhost:8000/documents/index
  → status == "ok", total_chunks_indexed > 0

✓ Retrieval returns relevant chunks
  → curl "http://localhost:8000/knowledge/search?q=fever&top_k=3"
  → results_count > 0, results[0].score > 0.3

✓ Chat uses retrieved medical knowledge
  → curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"I have fever for 3 days\",\"use_rag\":true}"
  → rag_active == true, sources_used is non-empty

✓ Existing Phase 1 features still work
  → curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"I have chest pain and can't breathe\"}"
  → response contains "EMERGENCY" or "URGENT"

✓ Existing tests pass
  → pytest tests/test_mvp.py -v
  → All tests PASSED

✓ New tests pass
  → pytest tests/test_phase2.py -v
  → All tests PASSED
```

---

## Project Structure

```
ai_health_assistant/
├── main.py                       # FastAPI app — all endpoints including Phase 2
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies (Phase 2 packages active)
├── pyproject.toml                # Build config
├── conftest.py                   # Pytest root marker
├── core/
│   ├── config.py                 # Settings (includes QDRANT_URL, QDRANT_COLLECTION)
│   ├── logger.py                 # Structured logging
│   └── orchestrator.py          # Pipeline (now includes RAG retrieval step)
├── agents/
│   ├── base_agent.py             # Abstract base — LLM dispatch, timeout, fallback
│   ├── router_agent.py           # Intent classification
│   ├── symptom_agent.py          # Symptom extraction
│   ├── emergency_agent.py        # Emergency detection
│   └── safety_agent.py          # Final response composer (Phase 2: RAG context)
├── knowledge/
│   ├── __init__.py
│   ├── store.py                  # Qdrant wrapper + sentence-transformers embedding
│   ├── chunker.py                # Text/PDF chunking
│   └── ingestion.py              # File → chunks → Qdrant pipeline
├── documents/
│   └── uploads/                  # Uploaded documents land here
└── tests/
    ├── test_mvp.py               # Phase 1 tests (unchanged)
    └── test_phase2.py            # Phase 2 tests (RAG, store, chunker, ingestion)
```

---

# OCR Setup (Phase 3 — Image PDF Support)

## What changed

The ingestion pipeline now automatically detects whether a PDF has a text layer. If it does not (e.g. WHO PDFs saved from a browser as image scans), it falls back to OCR using **Tesseract** and **pdf2image**.

Text-based PDFs still go through pypdf only — OCR never runs on them.

The new response field `ocr_used: true/false` tells you which path was taken.

---

## New files

| File | Purpose |
|---|---|
| `knowledge/ocr.py` | OCR logic — Tesseract dispatch, text cleaning |
| `tests/test_ocr.py` | 28 tests for OCR path, cleaning, and fallback handling |

## Modified files

| File | Change |
|---|---|
| `knowledge/ingestion.py` | Added OCR fallback in `_process_pdf()` |
| `requirements.txt` | Added `pytesseract`, `pdf2image`, `Pillow` |

---

## Step-by-step OCR installation (Windows 11)

### 1. Install Python packages

These are already in `requirements.txt` and installed with `pip install -r requirements.txt`:

- `pytesseract==0.3.13`
- `pdf2image==1.17.0`
- `Pillow==10.3.0`

### 2. Install Tesseract binary

pytesseract is a Python wrapper. It needs the Tesseract binary installed separately.

**Download the Windows installer:**

Go to: https://github.com/UB-Mannheim/tesseract/wiki

Download: `tesseract-ocr-w64-setup-5.x.x.exe` (use the latest 5.x version)

**Run the installer:**

- Accept defaults
- On the "Choose Components" screen, make sure "Add to PATH" is checked
- Recommended install path: `C:\Program Files\Tesseract-OCR\`

**Restart your terminal** after installation.

**Verify Tesseract is installed:**
```
tesseract --version
```

Expected output:
```
tesseract 5.3.4
 ...
```

**If you did not add to PATH during install**, add it manually:
1. Open System Properties → Environment Variables
2. Under System Variables, select `Path` → Edit
3. Add: `C:\Program Files\Tesseract-OCR\`
4. Click OK and restart terminal

### 3. Install Poppler (required by pdf2image)

pdf2image converts PDF pages to images. It needs Poppler.

**Download:**

Go to: https://github.com/oschwartz10612/poppler-windows/releases

Download the latest `Release-xx.xx.x-0.zip`

**Extract and add to PATH:**

1. Extract to `C:\poppler\`
2. Add `C:\poppler\Library\bin` to your System PATH (same steps as above)
3. Restart terminal

**Verify Poppler:**
```
pdftoppm -v
```

Expected output:
```
pdftoppm version 24.x.x
```

### 4. Verify OCR works end-to-end

Start the server, upload a scanned PDF, and index it:

```
curl -X POST http://localhost:8000/documents/upload -F "file=@C:\path\to\who_guide.pdf"
curl -X POST http://localhost:8000/documents/index
```

If OCR ran, the index response will include:
```json
{"ocr_used": true, "chunks_created": 35, "status": "ok"}
```

---

## Troubleshooting OCR

**Problem:** `TesseractNotFoundError: tesseract is not installed`
**Cause:** Tesseract binary not in PATH.
**Solution:** Install from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.

---

**Problem:** `pdf2image failed to convert ... Is Poppler installed`
**Cause:** Poppler not installed or not in PATH.
**Solution:** Download from https://github.com/oschwartz10612/poppler-windows/releases, extract, add `Library\bin` to PATH.

---

**Problem:** OCR runs but produces garbage text
**Cause:** PDF scan quality is very low, or non-English content without language pack.
**Solution:** For non-English PDFs, install the Tesseract language pack (e.g. `tesseract-ocr-w64-setup-5.x.x.exe` → Additional languages).

---

**Problem:** OCR is very slow
**Cause:** High DPI setting (300 DPI default) on a large multi-page PDF.
**Solution:** This is expected. A 50-page WHO PDF takes approximately 30–90 seconds depending on your CPU. The result is cached in Qdrant and subsequent searches are instant.

---

**Problem:** `ocr_used: false` even for a scanned PDF
**Cause:** pypdf extracted some text (e.g. embedded metadata or a few characters) that crossed the minimum threshold.
**Solution:** Check the actual search results with `/knowledge/search?q=your+query`. If results look correct, OCR was not needed. If results are empty or nonsense, re-index with `?recreate=true`.

---

## Running tests

Run all tests (no Tesseract or Poppler required — all mocked):

```
pytest tests/ -v
```

Run only OCR tests:

```
pytest tests/test_ocr.py -v
```

Expected output:
```
tests/test_ocr.py::TestHasTextLayer::test_page_with_sufficient_text_returns_true PASSED
tests/test_ocr.py::TestHasTextLayer::test_all_empty_pages_returns_false PASSED
...
28 passed in 0.14s
```
