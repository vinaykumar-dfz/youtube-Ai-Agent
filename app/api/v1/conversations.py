import asyncio
from datetime import datetime
import json
from typing import Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.core.logging import logger
from app.database.models import Conversation, ConversationMessage
from app.database.session import SessionLocal, get_db
from app.agent.runtime import execution_manager
from app.agent.agent import QAAgentService

router = APIRouter(prefix="/conversations", tags=["Conversations & Ask AI"])


# -------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------
class CreateConversationRequest(BaseModel):
    title: str = Field(default="New Discussion", max_length=255)
    context_channel_id: str | None = None
    context_video_id: str | None = None
    channel_context: dict[str, Any] | None = None


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000, description="User question or follow-up prompt")
    context_channel_id: str | None = None
    channel_context: dict[str, Any] | None = None


class ConversationMessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    evidence_sources: list[dict[str, Any]] | None = None
    tool_calls_summary: list[dict[str, Any]] | None = None
    extra_metadata: dict[str, Any] | None = None
    videos: list[dict[str, Any]] = []
    follow_up_questions: list[str] = []
    data_coverage: dict[str, Any] | None = None
    limitations: list[str] = []
    execution_state: dict[str, Any] | None = None
    working_memory: dict[str, Any] | None = None
    workflow_timeline: dict[str, Any] | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationSummaryResponse(BaseModel):
    id: str
    workspace_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class ConversationDetailResponse(BaseModel):
    id: str
    workspace_id: str
    title: str
    context_channel_id: str | None = None
    context_video_id: str | None = None
    channel_context: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessageResponse] = []

    class Config:
        from_attributes = True


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@router.post("", response_model=ConversationDetailResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: CreateConversationRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Create a new conversational discussion thread for the workspace."""
    from app.database.models import AnalysisReport, ChannelConnection

    resolved_channel_context = payload.channel_context
    resolved_channel_id = payload.context_channel_id

    # If context_channel_id is provided but channel_context is missing, resolve it from verified records
    if resolved_channel_id and not resolved_channel_context:
        ch_conn = (
            db.query(ChannelConnection)
            .filter(
                ChannelConnection.workspace_id == tenant.workspace_id,
                ChannelConnection.channel_id == resolved_channel_id,
            )
            .first()
        )
        if ch_conn:
            resolved_channel_context = {
                "channel_id": ch_conn.channel_id,
                "channel_name": ch_conn.title,
                "custom_url": ch_conn.custom_url,
                "subscriber_count": ch_conn.subscriber_count,
                "video_count": ch_conn.video_count,
                "view_count": ch_conn.view_count,
                "context_source": "workspace_channel_connection",
                "is_oauth_connected": bool(getattr(ch_conn, "access_token", None)),
            }
        else:
            rep = (
                db.query(AnalysisReport)
                .filter(
                    AnalysisReport.workspace_id == tenant.workspace_id,
                    AnalysisReport.channel_id == resolved_channel_id,
                )
                .order_by(AnalysisReport.created_at.desc())
                .first()
            )
            if rep:
                r_ms = rep.metrics_summary or {}
                r_info = r_ms.get("channel_info", {}) if isinstance(r_ms, dict) else {}
                resolved_channel_context = {
                    "channel_id": rep.channel_id,
                    "channel_name": rep.channel_title,
                    "custom_url": r_info.get("custom_url"),
                    "subscriber_count": r_info.get("subscriber_count") or (r_ms.get("subscribers", {}).get("raw") if isinstance(r_ms.get("subscribers"), dict) else 0),
                    "video_count": len(r_info.get("recent_videos", [])),
                    "view_count": r_info.get("view_count") or (r_ms.get("total_views", {}).get("raw") if isinstance(r_ms.get("total_views"), dict) else 0),
                    "context_source": "workspace_analysis_report",
                    "is_oauth_connected": False,
                }

    # If neither was provided, check workspace active channel
    if not resolved_channel_id:
        ch_conn = db.query(ChannelConnection).filter(ChannelConnection.workspace_id == tenant.workspace_id).first()
        if ch_conn:
            resolved_channel_id = ch_conn.channel_id
            resolved_channel_context = {
                "channel_id": ch_conn.channel_id,
                "channel_name": ch_conn.title,
                "custom_url": ch_conn.custom_url,
                "subscriber_count": ch_conn.subscriber_count,
                "video_count": ch_conn.video_count,
                "view_count": ch_conn.view_count,
                "context_source": "workspace_channel_connection",
                "is_oauth_connected": bool(getattr(ch_conn, "access_token", None)),
            }
        else:
            rep = db.query(AnalysisReport).filter(AnalysisReport.workspace_id == tenant.workspace_id).order_by(AnalysisReport.created_at.desc()).first()
            if rep:
                r_ms = rep.metrics_summary or {}
                r_info = r_ms.get("channel_info", {}) if isinstance(r_ms, dict) else {}
                resolved_channel_id = rep.channel_id
                resolved_channel_context = {
                    "channel_id": rep.channel_id,
                    "channel_name": rep.channel_title,
                    "custom_url": r_info.get("custom_url"),
                    "subscriber_count": r_info.get("subscriber_count") or (r_ms.get("subscribers", {}).get("raw") if isinstance(r_ms.get("subscribers"), dict) else 0),
                    "video_count": len(r_info.get("recent_videos", [])),
                    "view_count": r_info.get("view_count") or (r_ms.get("total_views", {}).get("raw") if isinstance(r_ms.get("total_views"), dict) else 0),
                    "context_source": "workspace_analysis_report",
                    "is_oauth_connected": False,
                }

    conv = Conversation(
        workspace_id=tenant.workspace_id,
        user_id=tenant.user.id,
        title=payload.title,
        context_channel_id=resolved_channel_id,
        context_video_id=payload.context_video_id,
        channel_context=resolved_channel_context,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("", response_model=list[ConversationSummaryResponse])
def list_conversations(
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """List previous conversations for the authenticated workspace."""
    convs = (
        db.query(Conversation)
        .filter(Conversation.workspace_id == tenant.workspace_id)
        .order_by(Conversation.updated_at.desc())
        .limit(30)
        .all()
    )
    results = []
    for c in convs:
        msg_count = db.query(ConversationMessage).filter(ConversationMessage.conversation_id == c.id).count()
        results.append(
            ConversationSummaryResponse(
                id=c.id,
                workspace_id=c.workspace_id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
                message_count=msg_count,
            )
        )
    return results


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation_history(
    conversation_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Retrieve full message history for a conversation enforcing workspace isolation."""
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.workspace_id == tenant.workspace_id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    # Unpack extra_metadata into structured response fields
    mapped_messages = []
    for m in conv.messages:
        em = m.extra_metadata or {}
        mapped_messages.append(
            ConversationMessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                evidence_sources=m.evidence_sources,
                tool_calls_summary=m.tool_calls_summary,
                extra_metadata=em,
                videos=em.get("videos", []),
                follow_up_questions=em.get("follow_up_questions", []),
                data_coverage=em.get("data_coverage"),
                limitations=em.get("limitations", []),
                execution_state=em.get("execution_state"),
                working_memory=em.get("working_memory"),
                workflow_timeline=em.get("workflow_timeline"),
                created_at=m.created_at,
            )
        )

    return ConversationDetailResponse(
        id=conv.id,
        workspace_id=conv.workspace_id,
        title=conv.title,
        context_channel_id=conv.context_channel_id,
        context_video_id=conv.context_video_id,
        channel_context=conv.channel_context,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=mapped_messages,
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Delete a conversation and all its associated messages."""
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.workspace_id == tenant.workspace_id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    db.delete(conv)
    db.commit()


@router.get("/{conversation_id}/live-execution")
def get_live_execution(
    conversation_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Retrieve the real-time execution state for an active or recently finished query."""
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.workspace_id == tenant.workspace_id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return execution_manager.get_live_state(conversation_id)


@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    payload: SendMessageRequest,
    request: Request,
    stream: bool = False,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Send a user message, run agentic tool reasoning, and return evidence-grounded response or SSE stream."""
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.workspace_id == tenant.workspace_id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found in workspace.",
        )

    accept_header = request.headers.get("accept", "")
    wants_sse = stream or ("text/event-stream" in accept_header)

    if not wants_sse:
        agent = QAAgentService(db)
        try:
            reply = agent.process_message(
                workspace_id=tenant.workspace_id,
                conversation_id=conversation_id,
                user_text=payload.content,
                user_id=tenant.user.id,
                channel_id_override=payload.context_channel_id,
                channel_context_override=payload.channel_context,
            )
            return reply
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent error processing message: {exc!s}",
            )

    # Real-Time SSE Streaming Mode
    run_id = str(uuid.uuid4())
    execution_run = execution_manager.create_run(
        conversation_id=conversation_id,
        run_id=run_id,
        user_query=payload.content.strip(),
    )
    sub_queue = execution_run.subscribe()

    def run_agent_sync():
        thread_db = SessionLocal()
        try:
            agent = QAAgentService(thread_db)
            agent.process_message(
                workspace_id=tenant.workspace_id,
                conversation_id=conversation_id,
                user_text=payload.content,
                user_id=tenant.user.id,
                channel_id_override=payload.context_channel_id,
                channel_context_override=payload.channel_context,
                execution_run=execution_run,
            )
        except Exception as exc:
            logger.error("Background agent execution error: %s", exc, exc_info=True)
            curr = execution_run.current_step_id or "planning"
            execution_run.fail_step(curr, str(exc))
            execution_run.fail_run(str(exc))
        finally:
            thread_db.close()

    async def sse_event_stream():
        loop = asyncio.get_running_loop()
        agent_future = loop.run_in_executor(None, run_agent_sync)

        # Initial run_started event
        initial_data = json.dumps(execution_run.to_dict())
        yield f"event: run_started\ndata: {initial_data}\n\n"

        try:
            while True:
                try:
                    event_data = await asyncio.wait_for(sub_queue.get(), timeout=5.0)
                    ev_type = event_data.get("type", "progress")
                    yield f"event: {ev_type}\ndata: {json.dumps(event_data)}\n\n"
                    if ev_type in ("workflow_completed", "workflow_failed"):
                        break
                except asyncio.TimeoutError:
                    if agent_future.done() and execution_run.status in ("completed", "failed"):
                        break
                    yield ": ping\n\n"
        except Exception as stream_err:
            logger.warning("SSE stream error or client disconnect: %s", stream_err)
        finally:
            execution_run.unsubscribe(sub_queue)
            await agent_future

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
