import uuid
from pathlib import Path

import httpx
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models.schemas import ChatRequest, ChatResponse, IngestResponse
from app.services.database import Chunk, Document, async_session_factory
from app.services.ingestion import SUPPORTED_EXTENSIONS, load_document, split_documents
from app.services.language import resolve_language
from app.services.hybrid_search import BM25Retriever
from app.services.rag import (
    GenerationService,
    RetrievalService,
    augment_section_chunks,
)
from app.services.reranker import get_reranker
from app.services.vector_store import get_vector_store_manager

_bm25_retriever = BM25Retriever()


class DocumentService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.upload_dir = Path(self.settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def vector_manager(self):
        return get_vector_store_manager()

    async def _remove_by_filename(
        self, session: AsyncSession, filename: str
    ) -> None:
        result = await session.execute(
            select(Document).where(Document.filename == filename)
        )
        for existing in result.scalars().all():
            await self.delete_document(session, existing.id)

    async def ingest_file(
        self,
        session: AsyncSession,
        filename: str,
        content: bytes,
    ) -> IngestResponse:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        await self._remove_by_filename(session, filename)

        document_id = str(uuid.uuid4())
        safe_name = f"{document_id}{suffix}"
        file_path = self.upload_dir / safe_name
        file_path.write_bytes(content)

        raw_docs = load_document(file_path)
        chunks = split_documents(
            raw_docs, self.settings, document_id, filename
        )

        vector_ids = self.vector_manager.add_documents(chunks)

        doc_record = Document(
            id=document_id,
            filename=filename,
            file_type=suffix.lstrip("."),
            file_path=str(file_path),
            chunk_count=len(chunks),
        )
        session.add(doc_record)

        for chunk, vector_id in zip(chunks, vector_ids, strict=True):
            session.add(
                Chunk(
                    id=chunk.id,
                    document_id=document_id,
                    chunk_index=chunk.metadata["chunk_index"],
                    page_number=chunk.metadata.get("page"),
                    text=chunk.page_content,
                    vector_id=vector_id,
                )
            )

        await session.commit()
        _bm25_retriever.invalidate_cache()

        return IngestResponse(
            document_id=document_id,
            filename=filename,
            chunks_created=len(chunks),
            message="Document ingested and indexed with TurboVec compression",
        )

    async def list_documents(self, session: AsyncSession) -> list[Document]:
        result = await session.execute(
            select(Document).order_by(Document.created_at.desc())
        )
        return list(result.scalars().all())

    async def delete_document(
        self, session: AsyncSession, document_id: str
    ) -> bool:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            return False

        chunk_result = await session.execute(
            select(Chunk.vector_id).where(Chunk.document_id == document_id)
        )
        vector_ids = list(chunk_result.scalars().all())
        self.vector_manager.delete_by_ids(vector_ids)

        file_path = Path(doc.file_path)
        if file_path.exists():
            file_path.unlink()

        await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()
        _bm25_retriever.invalidate_cache()
        return True


class ChatService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_manager = get_vector_store_manager()
        self.generation = GenerationService(self.settings)

    async def _resolve_document_id(
        self,
        session: AsyncSession,
        document_id: str | None,
    ) -> str | None:
        if document_id:
            return document_id
        docs = await DocumentService(self.settings).list_documents(session)
        if not docs:
            return None
        if len(docs) == 1:
            return docs[0].id
        # Prefer the newest upload when multiple documents exist.
        return docs[0].id

    async def chat(
        self,
        request: ChatRequest,
        session: AsyncSession | None = None,
        portfolio_fast: bool = False,
    ) -> ChatResponse:
        if portfolio_fast:
            use_rerank = self.settings.portfolio_reranker_enabled
            top_k = min(
                request.top_k or self.settings.top_k,
                self.settings.portfolio_top_k,
            )
            use_hybrid = True
        else:
            use_rerank = (
                request.use_reranker
                if request.use_reranker is not None
                else self.settings.reranker_enabled
            )
            top_k = request.top_k
            use_hybrid = request.use_hybrid
        reranker = get_reranker() if use_rerank else None
        retrieval = RetrievalService(
            self.vector_manager,
            self.settings,
            reranker,
            bm25=_bm25_retriever,
        )
        # Hold a DB session only for the cheap document-id lookup. Keeping one
        # open across retrieve+generate (via the old StreamingResponse path)
        # exhausted SQLAlchemy's QueuePool under a few concurrent questions.
        scoped_document_id = request.document_id
        if session is not None:
            scoped_document_id = await self._resolve_document_id(
                session, request.document_id
            )
        else:
            async with async_session_factory() as resolve_session:
                scoped_document_id = await self._resolve_document_id(
                    resolve_session, request.document_id
                )

        chunks = await retrieval.retrieve(
            query=request.question,
            top_k=top_k,
            document_id=scoped_document_id,
            use_reranker=use_rerank if portfolio_fast else request.use_reranker,
            use_hybrid=use_hybrid,
            portfolio_fast=portfolio_fast,
        )
        if portfolio_fast:
            chunks = await augment_section_chunks(
                request.question,
                chunks,
                vector_manager=self.vector_manager,
                bm25=_bm25_retriever,
                document_id=scoped_document_id,
            )
        lang = resolve_language(request.question, request.language)

        if not chunks:
            empty_msg = (
                "لا أتوفّر لدي معلومة كافية للإجابة. يرجى رفع مستندات أولاً."
                if lang == "ar"
                else "No relevant documents found. Please upload documents first."
            )
            return ChatResponse(
                answer=empty_msg,
                sources=[],
                model=self.settings.ollama_model,
            )

        answer = await self.generation.generate(
            request.question,
            chunks,
            language=lang,
            portfolio_fast=portfolio_fast,
        )
        sources = self.generation.to_source_chunks(chunks)

        return ChatResponse(
            answer=answer,
            sources=sources,
            model=self.settings.ollama_model,
        )


async def check_ollama_health(settings: Settings) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return {"status": "ok", "models": models}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
