# Offline RAG Chatbot

نظام RAG محلي بالكامل (100% offline) مبني على الرسم المعماري المقترح — مع ضغط TurboVec 8× وتشغيل LLM عبر Ollama.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────┐
│  Streamlit  │────▶│   FastAPI    │────▶│ Ollama  │
│  frontend   │     │   backend    │     │ Qwen2.5 │
└─────────────┘     └──────┬───────┘     └─────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         TurboVec     PostgreSQL    Embeddings
         (8× comp.)  (metadata)   e5-large
```

| Stage | Component | Implementation |
|-------|-----------|----------------|
| **Ingestion** | Loaders | LangChain: PDF, Word, MD, TXT |
| | Chunking | RecursiveCharacterTextSplitter — 512 tokens, overlap 64 |
| | Embeddings | `intfloat/multilingual-e5-large` (local) |
| | Compression | TurboVec `bit_width=4` (~8×) |
| **Storage** | Vectors | TurboVec index on disk (`save`/`load`) |
| | Metadata | PostgreSQL — filename, page, chunk text |
| **Retrieval** | Search | TurboVec Top-K + filter by `document_id` |
| | Reranker | Optional cross-encoder |
| **Generation** | LLM | Ollama — Qwen2.5 / Mistral |
| | UI | Streamlit frontend + FastAPI backend |

## Quick Start (Docker)

```bash
# From project root
docker compose up --build
```

| Service | URL |
|---------|-----|
| **UI (Streamlit)** | http://localhost:8501 |
| **API (FastAPI)** | http://localhost:8000/docs |
| **Ollama** | http://localhost:11434 |

First run downloads the embedding model (~2 GB) and pulls `qwen2.5:14b` via Ollama — allow several minutes.

### Pull a different Ollama model

```bash
docker compose exec ollama ollama pull mistral
```

Then set `OLLAMA_MODEL=mistral` in `docker-compose.yml`.

## Local Development (without Docker)

### 1. Start PostgreSQL & Ollama

```bash
docker compose up postgres ollama -d
docker compose exec ollama ollama pull qwen2.5:14b
```

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
mkdir -p data/uploads data/vector_store
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | System health (Ollama, TurboVec, DB) |
| `POST` | `/api/documents/upload` | Upload & ingest a document |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{id}` | Remove document from index |
| `POST` | `/api/chat` | RAG question answering |

### Chat example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هو موضوع المستند؟", "top_k": 5}'
```

Filter to one document:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "document_id": "<uuid>", "use_reranker": true}'
```

## Configuration

Environment variables (see `backend/.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Local embedding model |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Ollama model name |
| `CHUNK_SIZE` | `512` | Token chunk size |
| `CHUNK_OVERLAP` | `64` | Token overlap |
| `TURBOVEC_BIT_WIDTH` | `4` | Quantization (2 or 4) |
| `RERANKER_ENABLED` | `false` | Enable cross-encoder globally |
| `TOP_K` | `5` | Default retrieval count |

## Project Structure

```
.
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py              # FastAPI entry
│       ├── config.py            # Settings
│       ├── api/routes.py        # REST endpoints
│       ├── models/schemas.py    # Pydantic models
│       └── services/
│           ├── ingestion.py     # Loaders + chunking
│           ├── vector_store.py  # TurboVec + E5 embeddings
│           ├── database.py      # PostgreSQL models
│           ├── reranker.py      # Cross-encoder
│           ├── rag.py           # Retrieval + generation
│           └── document_service.py
└── frontend/
    ├── Dockerfile
    └── app.py                   # Streamlit UI
```

## Notes

- **Offline**: After initial model download, everything runs without internet.
- **TurboVec**: Drop-in LangChain vector store with disk persistence (`index.tvim` + `docstore.json`).
- **Arabic support**: `multilingual-e5-large` handles Arabic/English; swap to an Arabic-specific model via `EMBEDDING_MODEL` if needed.
- **RAM**: TurboVec compresses vectors ~8×; expect ~4 GB RAM for large corpora vs ~31 GB raw float32.
