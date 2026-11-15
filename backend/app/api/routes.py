from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
)
from app.services.database import get_session
from app.services.document_service import (
    ChatService,
    DocumentService,
    check_ollama_health,
)
from app.services.chat_stream import stream_chat_sse
from app.services.vector_store import get_vector_store_manager, vector_store_stats
from app.config import get_settings

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    settings = get_settings()
    ollama = await check_ollama_health(settings)
    vector_stats = vector_store_stats(settings)

    return HealthResponse(
        status="ok" if ollama.get("status") == "ok" else "degraded",
        ollama=ollama,
        vector_store=vector_stats,
        database="connected",
    )


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    service = DocumentService()
    try:
        return await service.ingest_file(session, file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Ingestion failed: {exc}"
        ) from exc


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(session: AsyncSession = Depends(get_session)):
    service = DocumentService()
    docs = await service.list_documents(session)
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                id=d.id,
                filename=d.filename,
                file_type=d.file_type,
                chunk_count=d.chunk_count,
                created_at=d.created_at,
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
):
    service = DocumentService()
    deleted = await service.delete_document(session, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted", "document_id": document_id}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
):
    # Do not Depends(get_session) here: StreamingResponse would keep that
    # connection checked out for the whole RAG+LLM wait and starve the pool
    # under a few concurrent questions. ChatService opens a short-lived
    # session only for document-id resolution.
    service = ChatService()
    portfolio_fast = request.portfolio_fast or "PortfolioChat" in (
        http_request.headers.get("user-agent", "")
    )
    wants_stream = request.stream or "text/event-stream" in (
        http_request.headers.get("accept", "").lower()
    )
    try:
        if wants_stream:
            return StreamingResponse(
                stream_chat_sse(request, portfolio_fast),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return await service.chat(request, portfolio_fast=portfolio_fast)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
