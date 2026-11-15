"""SSE streaming for long-running chat (keeps Netlify/proxy connections alive)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from app.models.schemas import ChatRequest, ChatResponse
from app.services.document_service import ChatService

PING_INTERVAL_SECONDS = 3.0


async def stream_chat_sse(
    request: ChatRequest,
    portfolio_fast: bool,
) -> AsyncIterator[str]:
    """Yield SSE events: ping while processing, then result or error."""
    yield "event: ping\ndata: {}\n\n"

    service = ChatService()
    task = asyncio.create_task(
        service.chat(request, portfolio_fast=portfolio_fast)
    )

    try:
        while not task.done():
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            if not task.done():
                yield "event: ping\ndata: {}\n\n"

        result: ChatResponse = await task
        payload = json.dumps(result.model_dump(), ensure_ascii=False)
        yield f"event: result\ndata: {payload}\n\n"
    except Exception as exc:
        error = json.dumps({"detail": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {error}\n\n"
    finally:
        if not task.done():
            task.cancel()
