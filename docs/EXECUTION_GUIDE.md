# AI Health Assistant – Execution Guide

This guide explains how to set up, configure, and run the AI Health Assistant locally.

The application uses:

- FastAPI
- Ollama (Llama 3)
- Qdrant Vector Database
- Sentence Transformers
- Retrieval-Augmented Generation (RAG)

This guide assumes a Windows environment, but the steps are similar for Linux and macOS.

---

# Prerequisites

Before starting, ensure the following software is installed.

| Software | Version |
|------------|---------|
| Python | 3.10 or later |
| Git | Latest |
| Ollama | Latest |
| Qdrant | v1.18+ |
| VS Code (Recommended) | Latest |

Verify your installations.

## Python

```bash
python --version
```

Expected:

```text
Python 3.10.x
```

---

## Git

```bash
git --version
```

Expected:

```text
git version 2.x.x
```

---

## Clone the Repository

```bash
git clone https://github.com/VishwanathBabu/ai-health-assistant.git

cd ai-health-assistant
```

---

# Project Structure

```
ai-health-assistant/
│
├── agents/
├── assets/
├── core/
├── docs/
├── documents/
│   └── uploads/
├── knowledge/
├── tests/
│
├── Dockerfile
├── main.py
├── requirements.txt
├── README.md
└── .env.example
```

---

# Create a Virtual Environment

Create a Python virtual environment.

```bash
py -3.10 -m venv venv
```

Activate it.

## Windows

```bash
venv\Scripts\activate
```

Your terminal should now display

```text
(venv)
```

Upgrade pip.

```bash
python -m pip install --upgrade pip
```

---

# Install Dependencies

Install all project dependencies.

```bash
pip install -r requirements.txt
```

This installs:

- FastAPI
- Uvicorn
- Sentence Transformers
- Qdrant Client
- PyPDF
- Pytest
- Structlog
- Pydantic
- Other required packages

Installation may take several minutes during the first setup because machine learning libraries are downloaded.

---

# Environment Configuration

Create a `.env` file in the project root.

```
ai-health-assistant/
│
├── .env
├── main.py
├── README.md
└── ...
```

You can copy the example configuration.

```bash
copy .env.example .env
```

or

```bash
cp .env.example .env
```

---

## Example Configuration

```env
LLM_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434

OLLAMA_MODEL=llama3

QDRANT_URL=http://localhost:6333

QDRANT_COLLECTION=medical_docs

LOG_LEVEL=INFO

API_HOST=0.0.0.0

API_PORT=8000

API_RELOAD=true
```

---

## Important Environment Variables

| Variable | Description |
|------------|-------------|
| LLM_PROVIDER | Selects the language model provider |
| OLLAMA_MODEL | Local model to use |
| OLLAMA_BASE_URL | Ollama server address |
| QDRANT_URL | Vector database URL |
| QDRANT_COLLECTION | Collection used for document embeddings |
| API_PORT | FastAPI server port |
| LOG_LEVEL | Logging verbosity |

The default values are suitable for local development and usually do not need to be modified.

---

# Verify Python Environment

Run:

```bash
python --version
```

```bash
python -m pip --version
```

Both commands should point to your virtual environment.

Example:

```text
Python 3.10.11

pip .../ai-health-assistant/venv/Lib/site-packages
```

If they point to a global Python installation instead, recreate the virtual environment before continuing.

---

# Next Step

Proceed to installing the required services:

- Ollama
- Qdrant

# Installing Ollama

The AI Health Assistant uses **Ollama** to run a local Large Language Model (LLM). No cloud API key is required when using Ollama.

---

## Step 1 — Install Ollama

Download the latest installer from:

https://ollama.com/download

Run the installer and complete the installation.

---

## Step 2 — Verify Installation

Open a terminal and run:

```bash
ollama --version
```

Expected output:

```text
ollama version x.x.x
```

---

## Step 3 — Download the Llama 3 Model

Pull the required model:

```bash
ollama pull llama3
```

The model is approximately **4–5 GB**, so the initial download may take several minutes.

---

## Step 4 — Verify the Model

```bash
ollama list
```

Example:

```text
NAME             SIZE
llama3:latest    4.7 GB
```

---

## Step 5 — Start Ollama

If Ollama is not already running:

```bash
ollama serve
```

Expected output:

```text
Listening on 127.0.0.1:11434
```

Leave this terminal running.

---

## Step 6 — Verify the API

Open another terminal and run:

```bash
curl http://localhost:11434/api/tags
```

If you receive JSON describing the installed models, Ollama is running correctly.

---

# Installing Qdrant

Qdrant stores vector embeddings for semantic search.

---

## Option 1 — Native Windows Installation (Recommended)

Download the latest Windows release:

https://github.com/qdrant/qdrant/releases

Extract the archive.

Run:

```bash
qdrant.exe
```

The server starts on:

```text
http://localhost:6333
```

---

## Verify Qdrant

Open your browser:

```
http://localhost:6333
```

You should see something similar to:

```json
{
    "title":"qdrant - vector search engine",
    "version":"1.x.x"
}
```

Health check:

```bash
curl http://localhost:6333/healthz
```

Expected:

```text
healthz check passed
```

---

## View Collections

```text
http://localhost:6333/collections
```

Initially, the list will be empty.

After indexing documents, your collection will appear here.

---

# Running the Application

Once both Ollama and Qdrant are running, start the FastAPI application.

Activate the virtual environment if necessary.

```bash
venv\Scripts\activate
```

Run:

```bash
python -m uvicorn main:app --reload
```

Expected startup log:

```text
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

---

# Open Swagger UI

Once the application starts successfully, open:

```
http://localhost:8000/docs
```

Swagger UI allows you to test every API endpoint directly from your browser.

---

# Health Checks

## Basic Health Check

```
GET /health
```

Example:

```
http://localhost:8000/health
```

Expected response:

```json
{
    "status":"ok"
}
```

---

## Readiness Check

```
GET /health/ready
```

Example:

```
http://localhost:8000/health/ready
```

The readiness endpoint verifies:

- FastAPI
- Ollama
- Qdrant

If all services are available, the API returns a successful response.

---

# First-Time Setup Checklist

Before proceeding, confirm:

- Python virtual environment is activated
- Dependencies are installed
- Ollama is running
- Llama 3 model has been downloaded
- Qdrant is running
- FastAPI starts successfully
- Swagger UI opens correctly

# Uploading Medical Documents

The AI Health Assistant uses Retrieval-Augmented Generation (RAG) by indexing medical documents into a Qdrant vector database.

Supported document formats:

- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)

---

# Option 1 — Upload Using Swagger UI

Open:

```
http://localhost:8000/docs
```

Locate:

```
POST /documents/upload
```

Click **Try it out**.

Choose a medical document from your computer.

Click **Execute**.

A successful upload returns a response similar to:

```json
{
    "status": "uploaded",
    "filename": "influenza.pdf"
}
```

---

# Option 2 — Upload Using cURL

```bash
curl -X POST http://localhost:8000/documents/upload ^
-F "file=@C:\Users\YourName\Documents\influenza.pdf"
```

---

# Build the Vector Database

Uploading a document only stores the file.

To make it searchable, create vector embeddings.

Execute:

```
POST /documents/index
```

from Swagger

or

```bash
curl -X POST http://localhost:8000/documents/index
```

---

## Successful Response

```json
{
    "status":"ok",
    "files_processed":1,
    "files_succeeded":1,
    "total_chunks_indexed":24
}
```

Depending on document size, indexing may take a few seconds.

---

# Semantic Search

The semantic search endpoint retrieves the most relevant document chunks using vector similarity.

```
GET /knowledge/search
```

Example:

```
http://localhost:8000/knowledge/search?q=influenza symptoms&top_k=3
```

---

## Example Response

```json
{
    "query":"influenza symptoms",
    "results_count":3,
    "results":[
        {
            "source":"influenza.pdf",
            "score":0.89,
            "text":"Influenza commonly presents with fever, cough, sore throat, muscle aches..."
        }
    ]
}
```

If no documents have been indexed, the results list will be empty.

---

# Chat Endpoint

The `/chat` endpoint is the primary interface for interacting with the assistant.

```
POST /chat
```

---

## Request Body

```json
{
    "message":"What are the symptoms of influenza?",
    "use_rag":true
}
```

When `use_rag` is set to `true`, the assistant retrieves relevant information from the indexed documents before generating a response.

---

## Example Response

```json
{
    "response":"Influenza commonly causes fever, cough, sore throat, fatigue, headache, muscle aches, and chills...",
    "sources_used":[
        "influenza.pdf"
    ],
    "rag_active":true
}
```

---

# Testing with Postman

## Upload Document

**Method**

```
POST
```

**URL**

```
http://localhost:8000/documents/upload
```

**Body**

Select

```
form-data
```

Create one field:

| Key | Type |
|------|------|
| file | File |

Choose your PDF.

Click **Send**.

---

# Index Documents

**Method**

```
POST
```

**URL**

```
http://localhost:8000/documents/index
```

No request body is required.

Click **Send**.

---

# Search Documents

**Method**

```
GET
```

**URL**

```
http://localhost:8000/knowledge/search
```

Parameters

| Key | Value |
|------|-------|
| q | influenza symptoms |
| top_k | 3 |

---

# Chat

**Method**

```
POST
```

**URL**

```
http://localhost:8000/chat
```

**Body**

Raw JSON

```json
{
    "message":"Summarize influenza including symptoms, transmission, prevention, and treatment.",
    "use_rag":true
}
```

---

# Swagger UI

Swagger is available at

```
http://localhost:8000/docs
```

Using Swagger, you can

- Upload documents
- Build embeddings
- Perform semantic search
- Chat with the assistant
- Verify API responses

No additional tools are required.

---

# Typical Workflow

The recommended workflow is:

```
Start Ollama

↓

Start Qdrant

↓

Start FastAPI

↓

Upload Documents

↓

Build Index

↓

Verify Semantic Search

↓

Use Chat Endpoint
```

---

# Docker Support

The project includes a Dockerfile for containerized execution.

> **Note:** Docker is optional. The project can be run directly using Python, Ollama, and Qdrant.

---

## Build the Docker Image

Navigate to the project root and run:

```bash
docker build -t ai-health-assistant .
```

---

## Run the Container

```bash
docker run -p 8000:8000 ai-health-assistant
```

The application will be available at:

```
http://localhost:8000
```

> Ensure Ollama and Qdrant are running before starting the container, unless you have configured them as Docker services.

---

# Running Unit Tests

The project includes unit tests covering the AI agents, RAG pipeline, and API functionality.

Run all tests:

```bash
pytest tests/ -v
```

Example output:

```text
======================== test session starts ========================

tests/test_mvp.py ..................... PASSED
tests/test_phase2.py ................. PASSED

======================== 35 passed ========================
```

---

## Run Individual Test Files

Run only the AI agent tests:

```bash
pytest tests/test_mvp.py -v
```

Run only the RAG tests:

```bash
pytest tests/test_phase2.py -v
```

---

## Run with Coverage

```bash
pytest --cov=. --cov-report=term-missing
```

Coverage reports help identify untested sections of the codebase.

---

# Repository Structure

```
ai-health-assistant/

├── agents/
│   ├── base_agent.py
│   ├── emergency_agent.py
│   ├── router_agent.py
│   ├── safety_agent.py
│   └── symptom_agent.py
│
├── assets/
│   ├── architecture.png
│   ├── swagger.png
│   ├── upload.png
│   ├── indexing.png
│   ├── search.png
│   ├── chat.png
│   └── qdrant.png
│
├── core/
│   ├── config.py
│   ├── logger.py
│   └── orchestrator.py
│
├── documents/
│   └── uploads/
│
├── knowledge/
│   ├── chunker.py
│   ├── ingestion.py
│   └── store.py
│
├── tests/
│   ├── test_mvp.py
│   └── test_phase2.py
│
├── docs/
│   └── EXECUTION_GUIDE.md
│
├── Dockerfile
├── main.py
├── README.md
├── requirements.txt
└── .env.example
```

---

# Demonstration Assets

The repository includes screenshots demonstrating the application.

| Screenshot | Description |
|------------|-------------|
| architecture.png | High-level system architecture |
| swagger.png | Swagger UI |
| upload.png | Document upload endpoint |
| indexing.png | Successful document indexing |
| search.png | Semantic search results |
| chat.png | AI chat response using RAG |
| qdrant.png | Qdrant collection metadata |

These images are displayed in the main `README.md`.

---

# Recommended Startup Order

Whenever you start the project locally, follow this order:

1. Activate the Python virtual environment.
2. Start Ollama.
3. Start Qdrant.
4. Launch FastAPI.
5. Upload medical documents.
6. Build the vector index.
7. Test semantic search.
8. Start chatting with the assistant.

Following this sequence ensures all services are available before the application begins processing requests.

---

# Useful Commands

Activate the virtual environment:

```bash
venv\Scripts\activate
```

Start the FastAPI server:

```bash
python -m uvicorn main:app --reload
```

Run tests:

```bash
pytest tests/ -v
```

Check Ollama models:

```bash
ollama list
```

Start Ollama manually:

```bash
ollama serve
```

Check Qdrant health:

```bash
curl http://localhost:6333/healthz
```

Open Swagger UI:

```
http://localhost:8000/docs
```

---

# Git Workflow

Clone the repository:

```bash
git clone https://github.com/VishwanathBabu/ai-health-assistant.git
```

Create a new branch:

```bash
git checkout -b feature-name
```

Commit changes:

```bash
git add .

git commit -m "Describe your changes"
```

Push:

```bash
git push origin feature-name
```

Open a Pull Request on GitHub.

---

# Best Practices

- Keep your `.env` file private.
- Never commit API keys.
- Re-index documents after uploading new files.
- Test changes before pushing to GitHub.
- Keep dependencies updated.
- Use Docker for consistent environments when deploying or sharing the project.


# Troubleshooting

This section covers the most common issues encountered during setup and execution.

---

## FastAPI Does Not Start

### Problem

The server fails to start or exits immediately.

### Solution

Ensure the virtual environment is activated.

```bash
venv\Scripts\activate
```

Verify all dependencies are installed.

```bash
pip install -r requirements.txt
```

Then start the application again.

```bash
python -m uvicorn main:app --reload
```

---

## Port 8000 Already in Use

### Problem

```
Address already in use
```

### Solution

Identify the process using the port.

```bash
netstat -ano | findstr :8000
```

Terminate the process.

```bash
taskkill /PID <PID_NUMBER> /F
```

Or run the application on another port.

```bash
python -m uvicorn main:app --reload --port 8001
```

---

## Ollama Not Running

### Problem

The application cannot connect to the language model.

### Solution

Start Ollama.

```bash
ollama serve
```

Verify the installed models.

```bash
ollama list
```

---

## Qdrant Not Running

### Problem

Semantic search does not return results.

### Solution

Start Qdrant.

Verify it is healthy.

```bash
curl http://localhost:6333/healthz
```

Expected response:

```text
healthz check passed
```

---

## Documents Are Not Retrieved

### Problem

The chat endpoint responds correctly, but no document sources are returned.

### Solution

Documents must be indexed before they become searchable.

Execute:

```text
POST /documents/index
```

or

```bash
curl -X POST http://localhost:8000/documents/index
```

---

## No Module Named ...

### Problem

```
ModuleNotFoundError
```

### Solution

Activate the virtual environment.

```bash
venv\Scripts\activate
```

Reinstall dependencies.

```bash
pip install -r requirements.txt
```

---

## Wrong Python Version

Verify Python.

```bash
python --version
```

Expected:

```text
Python 3.10.x
```

If another version is used, recreate the virtual environment with Python 3.10.

---

# Frequently Asked Questions

## Does the project require an OpenAI API key?

No.

By default the application uses **Ollama** with a locally hosted **Llama 3** model.

---

## Can I use OpenAI instead?

Yes.

Update the `.env` file.

```
LLM_PROVIDER=openai
```

Add your API key.

```
OPENAI_API_KEY=YOUR_API_KEY
```

---

## Can I upload multiple documents?

Yes.

Upload as many supported documents as required.

After uploading, rebuild the vector index.

---

## Which document formats are supported?

Currently supported:

- PDF
- TXT
- Markdown

---

## Does the project require Docker?

No.

Docker support is included for convenience, but the project runs perfectly using a Python virtual environment.

---

## Can I use another embedding model?

Yes.

The embedding model can be changed in the application configuration.

---

# Final Verification Checklist

Before using the project, verify the following.

## Environment

- Python 3.10 installed
- Virtual environment activated
- Dependencies installed

---

## Ollama

- Installed
- Running
- Llama 3 downloaded

Verify:

```bash
ollama list
```

---

## Qdrant

Running successfully.

Verify:

```bash
curl http://localhost:6333/healthz
```

---

## FastAPI

Server starts successfully.

```bash
python -m uvicorn main:app --reload
```

Swagger opens successfully.

```
http://localhost:8000/docs
```

---

## Document Pipeline

✔ Upload document

↓

✔ Build vector index

↓

✔ Perform semantic search

↓

✔ Chat with RAG enabled

---

## Tests

Execute:

```bash
pytest tests/ -v
```

All tests should pass successfully.

---

# Contributing

Contributions are welcome.

If you would like to improve the project:

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature-name
```

3. Make your changes.
4. Commit.

```bash
git commit -m "Describe your changes"
```

5. Push.

```bash
git push origin feature-name
```

6. Open a Pull Request.

---

# License

This project is released under the **MIT License**.

You are free to use, modify, and distribute the project under the terms of the license.

---

# Support

If you encounter any issues while using this project, please open an issue in the GitHub repository.

Bug reports, feature requests, and pull requests are always welcome.

---

# Acknowledgements

This project makes use of several excellent open-source technologies:

- FastAPI
- Ollama
- Qdrant
- Sentence Transformers
- PyPDF
- Pytest
- Pydantic

Special thanks to the maintainers and contributors of these projects for their work.

---

# Conclusion

You have now completed the setup of the AI Health Assistant.

The application is ready to:

- Upload and index medical documents
- Perform semantic search using Qdrant
- Generate context-aware healthcare responses using Retrieval-Augmented Generation (RAG)
- Run locally with Ollama and FastAPI
- Execute automated unit tests
- Run inside Docker

For an overview of the project architecture, features, and screenshots, refer to the main **README.md**.