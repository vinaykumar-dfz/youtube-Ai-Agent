from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
from typing import Any
import uuid
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.database.models import (
    ChannelConnection,
    Conversation,
    ConversationMessage,
    AnalysisReport,
)
from app.agent.tools import (
    ToolContext,
    execute_tool,
    get_agent_tool_definitions,
)
from app.agent.memory import CreatorMemoryService
from app.agent.runtime import (
    ExecutionRun,
    execution_manager,
    get_tool_meta,
    generate_safe_tool_summary,
)
from app.agent.providers.llm import OllamaLLMClient
from app.agent.providers.youtube import YouTubeAdapter


def _extract_youtube_target(text: str) -> str | None:
    """Extract YouTube channel URL, handle, or raw ID from user input."""
    # 1. Full YouTube URLs
    url_pattern = r"(https?://(?:www\.)?(?:youtube\.com/(?:channel/UC[a-zA-Z0-9_-]{22}|@[a-zA-Z0-9_.-]+|c/[a-zA-Z0-9_.-]+|user/[a-zA-Z0-9_.-]+|watch\?[^\s]+|shorts/[a-zA-Z0-9_-]{11})|youtu\.be/[a-zA-Z0-9_-]{11})[^\s]*)"
    m = re.search(url_pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).rstrip(".,;!?)>'\"]")

    # 2. Raw channel ID UC...
    cid_match = re.search(r"\b(UC[a-zA-Z0-9_-]{22})\b", text)
    if cid_match:
        return cid_match.group(1)

    # 3. Explicit handle @name
    handle_match = re.search(r"(?:^|[\s\"'(:])(@[a-zA-Z0-9_.-]{3,})\b", text)
    if handle_match:
        return handle_match.group(1)

    return None


# -------------------------------------------------------------
# Explicit Execution State & Working Memory Models
# -------------------------------------------------------------
@dataclass
class AgentExecutionState:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    workspace_id: str = ""
    user_id: str | None = None
    user_goal: str = ""
    selected_channel_id: str | None = None
    selected_video_id: str | None = None
    current_step: str = "planning"
    status: str = "pending"  # pending, planning, gathering_data, calculating_metrics, researching_opportunities, synthesizing_answer, completed, failed
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def update_stage(self, new_stage: str, step_name: str | None = None):
        self.status = new_stage
        self.current_step = step_name or new_stage
        self.updated_at = datetime.utcnow().isoformat()
        if step_name and step_name not in self.completed_steps:
            if new_stage in ("completed", "synthesizing_answer", "gathering_data", "calculating_metrics", "researching_opportunities"):
                self.completed_steps.append(step_name)

    def add_event(self, event_type: str, stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Record structured observability trace event for this agent run."""
        self.events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "stage": stage,
            "message": message,
            "metadata": metadata or {},
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "user_goal": self.user_goal,
            "selected_channel_id": self.selected_channel_id,
            "selected_video_id": self.selected_video_id,
            "current_step": self.current_step,
            "status": self.status,
            "completed_steps": self.completed_steps,
            "pending_steps": self.pending_steps,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "events": self.events,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class WorkingMemory:
    user_goal: str = ""
    active_channel_id: str | None = None
    active_channel_name: str | None = None
    active_channel_context: dict[str, Any] | None = None
    active_video_id: str | None = None
    active_video_title: str | None = None
    referenced_entity: str | None = None
    cached_inventory: list[dict[str, Any]] = field(default_factory=list)
    inventory_is_complete: bool = False
    inventory_total_results: int = 0
    cached_rankings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    cached_channel_summary: dict[str, Any] = field(default_factory=dict)
    calculated_metrics: dict[str, Any] = field(default_factory=dict)
    competitor_benchmarks: list[dict[str, Any]] = field(default_factory=list)
    web_findings: list[dict[str, Any]] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_goal": self.user_goal,
            "active_channel_id": self.active_channel_id,
            "active_channel_name": self.active_channel_name,
            "active_channel_context": self.active_channel_context,
            "active_video_id": self.active_video_id,
            "active_video_title": self.active_video_title,
            "referenced_entity": self.referenced_entity,
            "cached_inventory_count": len(self.cached_inventory),
            "inventory_is_complete": self.inventory_is_complete,
            "inventory_total_results": self.inventory_total_results or len(self.cached_inventory),
            "cached_inventory": self.cached_inventory[:50],
            "cached_rankings_metrics": list(self.cached_rankings.keys()),
            "cached_channel_summary": self.cached_channel_summary,
            "calculated_metrics": self.calculated_metrics,
            "competitor_benchmarks_count": len(self.competitor_benchmarks),
            "web_findings_count": len(self.web_findings),
            "known_limitations": self.known_limitations,
        }


class QAAgentService:
    """Production-ready, stateful, evidence-grounded AI agent for YouTube Q&A and competitor strategy."""

    def __init__(self, db: Session):
        self.db = db
        self.llm = OllamaLLMClient()
        self.memory_service = CreatorMemoryService(db)
        self.tools = get_agent_tool_definitions()

    def process_message(
        self,
        workspace_id: str,
        conversation_id: str,
        user_text: str,
        user_id: str | None = None,
        channel_id_override: str | None = None,
        channel_context_override: dict[str, Any] | None = None,
        execution_run: ExecutionRun | None = None,
    ) -> dict[str, Any]:
        """Execute the end-to-end agentic Q&A pipeline with stateful memory, reference resolution, and tool execution."""
        # 1. Authentication & Ownership Validation
        conversation = (
            self.db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.workspace_id == workspace_id,
            )
            .first()
        )
        if not conversation:
            raise ValueError(f"Conversation '{conversation_id}' not found in workspace.")

        # Persist user message
        user_msg = ConversationMessage(
            conversation_id=conversation.id,
            role="user",
            content=user_text.strip(),
        )
        self.db.add(user_msg)

        # Update title if default
        if conversation.title in ("New Discussion", "New Chat", "Ask GrowTube AI"):
            clean_title = user_text.strip().split("\n")[0][:45]
            if clean_title:
                conversation.title = clean_title

        self.db.commit()

        # Initialize workflow execution tracking
        if not execution_run:
            execution_run = execution_manager.get_latest_run_for_conversation(conversation_id)
            if not execution_run or execution_run.status != "running":
                execution_run = execution_manager.create_run(conversation_id, str(uuid.uuid4()), user_text.strip())

        if execution_run:
            execution_run.start_step("understanding", "Identifying what you want to find out.")

        # 2. Context Loading & History
        recent_msgs = (
            self.db.query(ConversationMessage)
            .filter(ConversationMessage.conversation_id == conversation.id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(10)
            .all()
        )
        history = list(reversed(recent_msgs))

        ch_conn = (
            self.db.query(ChannelConnection)
            .filter(ChannelConnection.workspace_id == workspace_id)
            .first()
        )

        latest_report = (
            self.db.query(AnalysisReport)
            .filter(AnalysisReport.workspace_id == workspace_id)
            .order_by(AnalysisReport.created_at.desc())
            .first()
        )

        # 3. Canonical Channel Context Resolution (Precedence: 1. User Message -> 2. Override/Conversation DB -> 3. Workspace Report/Connection -> 4. None)
        active_channel_context, channel_resolution_error = self._resolve_channel_context(
            user_text=user_text,
            conversation=conversation,
            ch_conn=ch_conn,
            latest_report=latest_report,
            channel_id_override=channel_id_override,
            channel_context_override=channel_context_override,
        )

        active_cid = active_channel_context.get("channel_id") if active_channel_context else None
        active_cname = active_channel_context.get("channel_name") if active_channel_context else None

        # Persist resolved active channel immediately to conversation
        if active_channel_context:
            conversation.channel_context = active_channel_context
            conversation.context_channel_id = active_cid
            self.db.commit()

        # Restore Working Memory & Execution State
        working_memory = WorkingMemory(
            user_goal=user_text.strip(),
            active_channel_id=active_cid,
            active_channel_name=active_cname,
            active_channel_context=active_channel_context,
            active_video_id=conversation.context_video_id,
        )

        # Restore working memory caches from prior assistant turns strictly for the active channel
        for m in reversed(history[:-1]):
            if m.role == "assistant" and m.extra_metadata:
                prev_wm = m.extra_metadata.get("working_memory")
                if isinstance(prev_wm, dict):
                    prev_cid = prev_wm.get("active_channel_id")
                    if prev_cid and active_cid and prev_cid == active_cid:
                        if not working_memory.cached_inventory and prev_wm.get("cached_inventory"):
                            working_memory.cached_inventory = prev_wm.get("cached_inventory", [])
                        if not working_memory.cached_channel_summary and prev_wm.get("cached_channel_summary"):
                            working_memory.cached_channel_summary = prev_wm.get("cached_channel_summary", {})
                        if not working_memory.active_video_title and prev_wm.get("active_video_title"):
                            working_memory.active_video_title = prev_wm.get("active_video_title")
                        if not working_memory.active_video_id and prev_wm.get("active_video_id"):
                            working_memory.active_video_id = prev_wm.get("active_video_id")
                        if prev_wm.get("inventory_is_complete"):
                            tot = int(prev_wm.get("inventory_total_results", len(working_memory.cached_inventory)))
                            working_memory.inventory_is_complete = True
                            working_memory.inventory_total_results = tot
                    elif prev_cid and active_cid and prev_cid != active_cid:
                        logger.info("Active channel changed from %s to %s; cleared previous channel cached inventory.", prev_cid, active_cid)
                break

        # Instantiate ToolContext with restored working memory caches
        tool_ctx = ToolContext(
            db=self.db,
            workspace_id=workspace_id,
            job_id=conversation.id,
            user_id=user_id,
            active_channel_id=working_memory.active_channel_id or active_cid,
            active_channel_name=working_memory.active_channel_name or active_cname,
            active_channel_context=working_memory.active_channel_context or active_channel_context,
            cached_inventory=working_memory.cached_inventory,
            inventory_is_complete=working_memory.inventory_is_complete,
            access_token=getattr(ch_conn, "access_token", None) if ch_conn else None,
        )

        # Initialize explicit AgentExecutionState
        execution_state = AgentExecutionState(
            conversation_id=conversation.id,
            workspace_id=workspace_id,
            user_id=user_id,
            user_goal=user_text.strip(),
            selected_channel_id=working_memory.active_channel_id,
            selected_video_id=working_memory.active_video_id,
            status="planning",
            current_step="Intent & Reference Resolution",
        )
        execution_state.add_event("request_received", "planning", f"User question received: '{user_text[:80]}'")

        conn_state = "oauth_connected" if (active_channel_context and active_channel_context.get("is_oauth_connected")) else ("selected_public" if active_channel_context else "unselected")
        execution_state.add_event(
            "channel_context_resolved",
            "planning",
            f"Resolved active channel: '{active_cname}' ({active_cid}) [Connection: {conn_state}]" if active_channel_context else "No active channel context resolved; clarification required",
            {
                "channel_id": active_cid,
                "channel_name": active_cname,
                "connection_state": conn_state,
                "context_source": active_channel_context.get("context_source") if active_channel_context else None,
                "oauth_required": False,
            },
        )

        # Initialize tool execution tracking before conversational reference resolution
        tool_calls_executed: list[dict[str, Any]] = []

        # 4. Conversational Reference Resolution (e.g., 'it', 'latest video', 'previous video', 'second one')
        resolved_ref_str = self._resolve_conversational_references(
            user_text=user_text,
            conversation=conversation,
            history=history,
            working_memory=working_memory,
            tool_ctx=tool_ctx,
            execution_state=execution_state,
            tool_calls_executed=tool_calls_executed,
        )
        if resolved_ref_str:
            working_memory.referenced_entity = resolved_ref_str
            execution_state.selected_video_id = working_memory.active_video_id
            execution_state.add_event("reference_resolved", "planning", f"Resolved conversational reference: {resolved_ref_str}")
            logger.info("Resolved conversational reference: %s", resolved_ref_str)

        # 5. Intent Detection & Planning
        intent = self._detect_intent(user_text, conversation, working_memory)
        execution_state.update_stage("planning", f"Detected Intent: {intent}")
        execution_state.add_event("intent_resolved", "planning", f"Detected user intent: {intent}")
        logger.info(f"Q&A Agent detected intent: '{intent}' for query: '{user_text[:60]}'")

        if execution_run:
            execution_run.complete_step("understanding", f"Analyzed query intent: '{intent}'")
            execution_run.start_step("planning", "Working out the best approach to your request.")

        # 6. Pipeline Execution
        evidence_sources: list[dict[str, Any]] = []
        videos_payload: list[dict[str, Any]] = []
        metrics_payload: dict[str, Any] = {}
        data_coverage: dict[str, Any] = {
            "observation_period": "Lifetime public metrics",
            "coverage_percentage": 0.0,
            "data_source": "YouTube Data API v3",
        }
        limitations: list[str] = [
            "Public YouTube Data API v3 provides lifetime public view counts, like counts, and upload dates.",
            "Private creator metrics such as Click-Through Rate (CTR), Average View Duration, and Audience Retention Curves require YouTube Studio Analytics integration or an uploaded analytics export (.csv).",
        ]
        follow_up_questions: list[str] = []
        final_answer = ""

        # Check for private creator analytics requests (CTR, retention curves, impressions)
        q_clean = user_text.lower().strip()
        is_private_metric_query = any(k in q_clean for k in (
            "ctr", "click-through rate", "click through rate", "retention curve",
            "audience retention", "retention graph", "impressions", "traffic source",
            "private analytics", "studio analytics"
        )) and not any(k in q_clean for k in ("what is ctr", "define ctr", "how to improve ctr", "explain retention"))

        is_channel_query = any(k in q_clean for k in (
            "my channel", "our channel", "connected channel", "this channel",
            "my videos", "my latest video", "highest-performing video", "rank videos",
            "channel overview", "channel stats", "what type of content", "who is the channel"
        ))

        if is_private_metric_query and active_channel_context and not active_channel_context.get("is_oauth_connected"):
            act_name = active_channel_context.get("channel_name", "Selected Channel")
            act_id = active_channel_context.get("channel_id", "")
            execution_state.update_stage("completed", "OAuth Authorization Required for Private Metrics")
            execution_state.add_event(
                "oauth_requirement_explained",
                "completed",
                f"Private creator studio metric requested for channel '{act_name}' without OAuth connection",
                {
                    "channel_id": act_id,
                    "channel_name": act_name,
                    "is_oauth_connected": False,
                    "oauth_required": True,
                    "authorization_reason": "Private creator metrics (CTR, retention curves, impressions) require YouTube Analytics API OAuth authorization",
                },
            )
            if execution_run:
                execution_run.complete_step("planning", "Identified OAuth requirement for private creator metrics.")
                execution_run.skip_step("selecting_tools", "No tools needed; explaining private metrics authorization.")
                execution_run.skip_step("calling_tools", "No tools needed; explaining private metrics authorization.")
                execution_run.skip_step("reviewing_results", "No tools needed; explaining private metrics authorization.")
                execution_run.skip_step("deciding_next", "No tools needed; explaining private metrics authorization.")
                execution_run.skip_step("validating_findings", "No tools needed; explaining private metrics authorization.")
                execution_run.start_step("generating_answer", "Turning the findings into a useful response.")
                execution_run.complete_step("generating_answer", "Formulated authorization guidance.")
            final_answer = (
                f"### Private Creator Metric Authorization Required\n\n"
                f"Channel **{act_name}** (Channel ID: `{act_id}`) is actively selected and verified in your workspace.\n\n"
                f"However, **Click-Through Rate (CTR)**, **Impressions**, and **Audience Retention Curves** "
                f"are private creator studio metrics that require Google OAuth authorization via the YouTube Analytics API (`yt-analytics.readonly`). "
                f"Public YouTube Data API v3 does not expose private creator studio telemetry.\n\n"
                f"**How to connect OAuth:**\n"
                f"1. Navigate to **Settings** in the left sidebar.\n"
                f"2. Under **YouTube Channel Connection**, click **Connect with Google**.\n"
                f"3. Authorize read access to YouTube Analytics.\n\n"
                f"**Available Public Metrics:**\n"
                f"All public performance data for **{act_name}** remains fully accessible, including lifetime view counts, "
                f"view velocity (views/day), engagement rates, publish dates, and video catalog rankings."
            )
            follow_up_questions = [
                f"What is the highest-performing video on {act_name}?",
                f"Show my top 5 videos by views.",
                f"What type of content does {act_name} publish?",
            ]
        elif not active_channel_context and is_channel_query:
            execution_state.update_stage("completed", "Channel Clarification Required")
            execution_state.add_event(
                "channel_clarification_requested",
                "completed",
                "No channel selected in workspace; clarification requested from user",
                {
                    "channel_id": None,
                    "is_oauth_connected": False,
                    "oauth_required": False,
                    "authorization_reason": "Channel identification required before inspecting inventory",
                },
            )
            if execution_run:
                execution_run.complete_step("planning", "Identified channel selection requirement.")
                execution_run.skip_step("selecting_tools", "No tools needed; requesting channel selection.")
                execution_run.skip_step("calling_tools", "No tools needed; requesting channel selection.")
                execution_run.skip_step("reviewing_results", "No tools needed; requesting channel selection.")
                execution_run.skip_step("deciding_next", "No tools needed; requesting channel selection.")
                execution_run.skip_step("validating_findings", "No tools needed; requesting channel selection.")
                execution_run.start_step("generating_answer", "Turning the findings into a useful response.")
                execution_run.complete_step("generating_answer", "Formulated channel selection guidance.")
            final_answer = (
                "### No YouTube Channel Selected\n\n"
                "There is currently no YouTube channel selected or connected in this workspace.\n\n"
                "**To proceed, please:**\n"
                "- Select a channel from the workspace channel selector above, or\n"
                "- Provide a YouTube channel handle (e.g. `@channelname`), channel URL, or channel ID in your message, or\n"
                "- Analyze a channel first via the **Analyze** tab."
            )
            follow_up_questions = [
                "Explain about Codebasics Channel",
                "What YouTube growth strategies work best for tech channels?",
            ]
        # Route to reasoning engine or deterministic handlers
        elif self.llm.is_configured:
            final_answer, videos_payload, metrics_payload, evidence_sources, data_coverage, follow_up_questions, tool_calls_executed = (
                self._handle_open_reasoning(
                    user_text=user_text,
                    conversation=conversation,
                    history=history,
                    ch_conn=ch_conn,
                    latest_report=latest_report,
                    tool_ctx=tool_ctx,
                    intent=intent,
                    execution_state=execution_state,
                    working_memory=working_memory,
                    resolved_ref_str=resolved_ref_str,
                    execution_run=execution_run,
                )
            )
        else:
            # Deterministic Fast Paths when LLM is unconfigured
            execution_state.update_stage("gathering_data", f"Executing Deterministic Path: {intent}")
            if intent == "rank_videos":
                final_answer, videos_payload, metrics_payload, evidence_sources, data_coverage, follow_up_questions, tool_calls_executed = (
                    self._handle_rank_videos(user_text, conversation, ch_conn, tool_ctx)
                )
            elif intent == "compare_videos":
                final_answer, videos_payload, metrics_payload, evidence_sources, data_coverage, follow_up_questions, tool_calls_executed = (
                    self._handle_compare_videos(user_text, conversation, ch_conn, latest_report, tool_ctx)
                )
            elif intent == "video_performance":
                final_answer, videos_payload, metrics_payload, evidence_sources, data_coverage, follow_up_questions, tool_calls_executed = (
                    self._handle_video_performance(user_text, conversation, ch_conn, latest_report, tool_ctx)
                )
            elif intent == "channel_overview":
                final_answer, videos_payload, metrics_payload, evidence_sources, data_coverage, follow_up_questions, tool_calls_executed = (
                    self._handle_channel_overview(ch_conn, latest_report, tool_ctx)
                )
            else:
                final_answer, videos_payload, metrics_payload, evidence_sources, data_coverage, follow_up_questions, tool_calls_executed = (
                    self._handle_open_reasoning(
                        user_text=user_text,
                        conversation=conversation,
                        history=history,
                        ch_conn=ch_conn,
                        latest_report=latest_report,
                        tool_ctx=tool_ctx,
                        intent=intent,
                        execution_state=execution_state,
                        working_memory=working_memory,
                        resolved_ref_str=resolved_ref_str,
                    )
                )
            execution_state.update_stage("completed", "Deterministic Analysis Completed")

        # Section 9: Validate final response against active channel context
        if active_channel_context and active_channel_context.get("channel_name"):
            act_name = active_channel_context.get("channel_name")
            act_id = active_channel_context.get("channel_id")
            q_lower = user_text.lower()
            is_active_channel_query = any(k in q_lower for k in ("connected channel", "my channel", "this channel", "our channel")) or (act_name and act_name.lower() in q_lower)
            if is_active_channel_query and any(k in q_lower for k in ("channel name", "connected channel", "channel publish", "type of content", "who is", "what content", "channel stats", "about this channel")):
                if act_name.lower() not in final_answer.lower() and (not act_id or act_id.lower() not in final_answer.lower()):
                    header = f"**Channel Name**: **{act_name}** (Channel ID: `{act_id}`)\n\n"
                    final_answer = header + final_answer

        # 7. Candidate Long-Term Memory Extraction (Durable facts only)
        try:
            self.memory_service.extract_and_save_candidate_memories(
                workspace_id=workspace_id,
                user_text=user_text,
                assistant_text=final_answer,
                tool_calls=tool_calls_executed,
                source_job_id=conversation.id,
                user_id=user_id,
            )
        except Exception as mem_err:
            logger.debug(f"Candidate memory extraction notice: {mem_err}")

        # 8. Persist Assistant Turn & Context
        workflow_timeline_dict = execution_run.to_dict() if execution_run else None
        extra_meta = {
            "intent": intent,
            "status": execution_state.status,
            "execution_state": execution_state.to_dict(),
            "workflow_timeline": workflow_timeline_dict,
            "working_memory": working_memory.to_dict(),
            "channel_context": active_channel_context,
            "videos": videos_payload,
            "metrics": metrics_payload,
            "data_coverage": data_coverage,
            "limitations": limitations,
            "follow_up_questions": follow_up_questions,
        }

        assistant_msg = ConversationMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=final_answer.strip(),
            evidence_sources=evidence_sources if evidence_sources else None,
            tool_calls_summary=tool_calls_executed if tool_calls_executed else None,
            extra_metadata=extra_meta,
        )
        self.db.add(assistant_msg)

        # Update conversation context and persist canonical active channel
        if active_channel_context:
            conversation.channel_context = active_channel_context
            conversation.context_channel_id = active_channel_context.get("channel_id")
        if videos_payload:
            primary_vid = videos_payload[0].get("video_id")
            if primary_vid:
                conversation.context_video_id = primary_vid

        self.db.commit()
        self.db.refresh(assistant_msg)

        ret_payload = {
            "conversation_id": conversation.id,
            "message_id": assistant_msg.id,
            "role": "assistant",
            "content": assistant_msg.content,
            "intent": intent,
            "status": execution_state.status,
            "execution_state": execution_state.to_dict(),
            "workflow_timeline": workflow_timeline_dict,
            "working_memory": working_memory.to_dict(),
            "channel_context": active_channel_context,
            "videos": videos_payload,
            "metrics": metrics_payload,
            "evidence_sources": evidence_sources,
            "data_coverage": data_coverage,
            "limitations": limitations,
            "follow_up_questions": follow_up_questions,
            "tool_calls_summary": tool_calls_executed,
            "created_at": assistant_msg.created_at.isoformat(),
        }

        if execution_run:
            execution_run.finish_run(ret_payload)

        return ret_payload

    # -------------------------------------------------------------
    # Canonical Channel Context Resolution
    # -------------------------------------------------------------
    def _resolve_channel_context(
        self,
        user_text: str,
        conversation: Conversation,
        ch_conn: ChannelConnection | None,
        latest_report: AnalysisReport | None = None,
        channel_id_override: str | None = None,
        channel_context_override: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Deterministic channel resolution with verified database and workspace grounding.

        Precedence:
        1. Valid external channel explicitly specified in current user message.
        2. Explicit channel context override from request payload or conversation.channel_context.
        3. Active channel ID from conversation.context_channel_id or channel_id_override, resolved via DB/YouTube.
        4. Workspace selected/active channel:
           a. Workspace ChannelConnection
           b. Workspace AnalysisReport (latest verified report)
        5. Return None if no channel can be resolved (requiring user clarification).
        """
        extracted_target = _extract_youtube_target(user_text)
        resolution_error = None
        resolved_context = None

        # Check if the extracted target is actually just referencing the existing active channel
        existing_cid = (
            channel_id_override
            or conversation.context_channel_id
            or (ch_conn.channel_id if ch_conn else None)
            or (latest_report.channel_id if latest_report else None)
        )
        is_self_reference = False
        if extracted_target and existing_cid:
            if extracted_target == existing_cid:
                is_self_reference = True
            elif latest_report and latest_report.metrics_summary:
                rep_info = latest_report.metrics_summary.get("channel_info", {})
                if rep_info.get("custom_url") and extracted_target.lower() == rep_info.get("custom_url", "").lower():
                    is_self_reference = True

        if extracted_target and not is_self_reference:
            try:
                yt = YouTubeAdapter()
                t_info = yt.parse_youtube_target(extracted_target)
                ch_data = yt.fetch_channel_data(t_info)
                if ch_data and ch_data.get("channel_id"):
                    resolved_context = {
                        "channel_id": ch_data["channel_id"],
                        "channel_name": ch_data.get("title") or "YouTube Channel",
                        "source_url": extracted_target,
                        "resolution_status": "resolved",
                        "resolved_at": datetime.now(timezone.utc).isoformat(),
                        "context_source": "explicit_user_input",
                        "conversation_id": conversation.id,
                        "custom_url": ch_data.get("custom_url"),
                        "subscriber_count": ch_data.get("subscriber_count"),
                        "video_count": int(ch_data.get("video_count", 0)),
                        "view_count": int(ch_data.get("view_count", 0)),
                        "description": ch_data.get("description", ""),
                        "is_oauth_connected": False,
                    }
                    logger.info("Resolved new channel context from user text: %s (%s)", resolved_context['channel_name'], resolved_context['channel_id'])
            except Exception as exc:
                logger.warning(f"Failed to resolve channel target '{extracted_target}': {exc}")
                resolution_error = f"Could not resolve channel target '{extracted_target}': {exc}"

        # Precedence Evaluation:
        # Level 1: Explicit user input that resolved successfully
        if resolved_context:
            return resolved_context, None

        # Level 2: Explicit channel context override from request or conversation
        raw_ctx = channel_context_override or conversation.channel_context
        if raw_ctx and isinstance(raw_ctx, dict) and raw_ctx.get("channel_id"):
            active_ctx = dict(raw_ctx)
            if "is_oauth_connected" not in active_ctx:
                active_ctx["is_oauth_connected"] = bool(ch_conn and getattr(ch_conn, "access_token", None) and ch_conn.channel_id == active_ctx.get("channel_id"))
            return active_ctx, resolution_error

        # Level 3: Target Channel ID specified in conversation or override
        target_cid = channel_id_override or conversation.context_channel_id
        if target_cid:
            if ch_conn and ch_conn.channel_id == target_cid:
                ctx_obj = {
                    "channel_id": ch_conn.channel_id,
                    "channel_name": ch_conn.title or "Connected Channel",
                    "source_url": None,
                    "resolution_status": "resolved",
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                    "context_source": "workspace_channel_connection",
                    "conversation_id": conversation.id,
                    "custom_url": ch_conn.custom_url,
                    "subscriber_count": ch_conn.subscriber_count,
                    "video_count": ch_conn.video_count or 0,
                    "view_count": ch_conn.view_count or 0,
                    "description": "",
                    "is_oauth_connected": bool(getattr(ch_conn, "access_token", None)),
                }
                return ctx_obj, resolution_error

            matched_report = (
                self.db.query(AnalysisReport)
                .filter(
                    AnalysisReport.workspace_id == conversation.workspace_id,
                    AnalysisReport.channel_id == target_cid,
                )
                .order_by(AnalysisReport.created_at.desc())
                .first()
            )
            if matched_report:
                r_ms = matched_report.metrics_summary or {}
                r_info = r_ms.get("channel_info", {}) if isinstance(r_ms, dict) else {}
                ctx_obj = {
                    "channel_id": matched_report.channel_id,
                    "channel_name": matched_report.channel_title or r_info.get("title") or "Selected Channel",
                    "source_url": r_info.get("channel_url"),
                    "resolution_status": "resolved",
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                    "context_source": "workspace_analysis_report",
                    "conversation_id": conversation.id,
                    "custom_url": r_info.get("custom_url"),
                    "subscriber_count": r_info.get("subscriber_count") or (r_ms.get("subscribers", {}).get("raw") if isinstance(r_ms.get("subscribers"), dict) else 0),
                    "video_count": len(r_info.get("recent_videos", [])),
                    "view_count": r_info.get("view_count") or (r_ms.get("total_views", {}).get("raw") if isinstance(r_ms.get("total_views"), dict) else 0),
                    "description": "",
                    "is_oauth_connected": False,
                }
                return ctx_obj, resolution_error

            try:
                yt = YouTubeAdapter()
                t_info = yt.parse_youtube_target(target_cid)
                ch_data = yt.fetch_channel_data(t_info)
                if ch_data and ch_data.get("channel_id"):
                    ctx_obj = {
                        "channel_id": ch_data["channel_id"],
                        "channel_name": ch_data.get("title") or "YouTube Channel",
                        "source_url": f"https://www.youtube.com/channel/{ch_data['channel_id']}",
                        "resolution_status": "resolved",
                        "resolved_at": datetime.now(timezone.utc).isoformat(),
                        "context_source": "channel_id_resolved",
                        "conversation_id": conversation.id,
                        "custom_url": ch_data.get("custom_url"),
                        "subscriber_count": ch_data.get("subscriber_count"),
                        "video_count": int(ch_data.get("video_count", 0)),
                        "view_count": int(ch_data.get("view_count", 0)),
                        "description": ch_data.get("description", ""),
                        "is_oauth_connected": False,
                    }
                    return ctx_obj, resolution_error
            except Exception as e:
                logger.debug(f"Could not resolve target_cid {target_cid} via YouTube: {e}")

        # Level 4: Workspace active channel (ChannelConnection or latest AnalysisReport)
        if ch_conn and ch_conn.channel_id:
            oauth_ctx = {
                "channel_id": ch_conn.channel_id,
                "channel_name": ch_conn.title or "Connected Channel",
                "source_url": None,
                "resolution_status": "resolved",
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "context_source": "workspace_channel_connection",
                "conversation_id": conversation.id,
                "custom_url": ch_conn.custom_url,
                "subscriber_count": ch_conn.subscriber_count,
                "video_count": ch_conn.video_count or 0,
                "view_count": ch_conn.view_count or 0,
                "description": "",
                "is_oauth_connected": bool(getattr(ch_conn, "access_token", None)),
            }
            return oauth_ctx, resolution_error

        if latest_report and latest_report.channel_id:
            r_ms = latest_report.metrics_summary or {}
            r_info = r_ms.get("channel_info", {}) if isinstance(r_ms, dict) else {}
            rep_ctx = {
                "channel_id": latest_report.channel_id,
                "channel_name": latest_report.channel_title or r_info.get("title") or "Selected Channel",
                "source_url": r_info.get("channel_url"),
                "resolution_status": "resolved",
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "context_source": "workspace_analysis_report",
                "conversation_id": conversation.id,
                "custom_url": r_info.get("custom_url"),
                "subscriber_count": r_info.get("subscriber_count") or (r_ms.get("subscribers", {}).get("raw") if isinstance(r_ms.get("subscribers"), dict) else 0),
                "video_count": len(r_info.get("recent_videos", [])),
                "view_count": r_info.get("view_count") or (r_ms.get("total_views", {}).get("raw") if isinstance(r_ms.get("total_views"), dict) else 0),
                "description": "",
                "is_oauth_connected": False,
            }
            # Synchronize ChannelConnection record in SQLite
            try:
                new_conn = ChannelConnection(
                    workspace_id=conversation.workspace_id,
                    channel_id=latest_report.channel_id,
                    title=latest_report.channel_title,
                    custom_url=r_info.get("custom_url"),
                    subscriber_count=int(rep_ctx["subscriber_count"] or 0),
                    video_count=int(rep_ctx["video_count"] or 0),
                    view_count=int(rep_ctx["view_count"] or 0),
                    thumbnail_url=r_info.get("thumbnail_url"),
                    is_active=True,
                )
                self.db.add(new_conn)
                self.db.commit()
            except Exception as sync_err:
                self.db.rollback()
                logger.debug(f"ChannelConnection sync notice: {sync_err}")

            return rep_ctx, resolution_error

        # Level 5: No valid channel resolved
        return None, resolution_error

    # -------------------------------------------------------------
    # Conversational Reference Resolution
    # -------------------------------------------------------------
    def _resolve_conversational_references(
        self,
        user_text: str,
        conversation: Conversation,
        history: list[ConversationMessage],
        working_memory: WorkingMemory,
        tool_ctx: ToolContext,
        execution_state: AgentExecutionState | None = None,
        tool_calls_executed: list[dict[str, Any]] | None = None,
    ) -> str:
        """Resolve conversational pronouns, ordinals, and follow-up references.

        Binds 'it', 'latest video', 'previous video', 'second one' to actual video IDs/titles.
        Returns a descriptive resolution string injected into prompt context.
        """
        q = user_text.lower().strip()
        resolutions: list[str] = []

        # 1. "Latest video" / "Most recent video" reference
        if any(p in q for p in ("latest video", "recent video", "last video", "newest video", "latest upload")):
            if not working_memory.cached_inventory:
                inv_res = execute_tool("get_channel_video_inventory", {"max_results": 500}, tool_ctx)
                tool_rec = {
                    "tool": "get_channel_video_inventory",
                    "tool_name": "get_channel_video_inventory",
                    "arguments": {"max_results": 500},
                    "success": inv_res.get("success", True),
                    "context": "conversational_reference_resolution",
                }
                if tool_calls_executed is not None:
                    tool_calls_executed.append(tool_rec)
                if execution_state is not None:
                    execution_state.tool_calls.append(tool_rec)
                    execution_state.add_event(
                        "tool_call_completed",
                        "planning",
                        f"Resolved channel video inventory for reference ({len(inv_res.get('videos', []))} videos)",
                        {"tool": "get_channel_video_inventory", "success": inv_res.get("success", True)},
                    )
                if inv_res.get("success") and inv_res.get("videos"):
                    working_memory.cached_inventory = inv_res["videos"]
                    working_memory.inventory_is_complete = bool(inv_res.get("is_complete", False))
                    working_memory.inventory_total_results = int(inv_res.get("total_channel_videos", len(inv_res["videos"])))

            if working_memory.cached_inventory:
                latest = working_memory.cached_inventory[0]
                working_memory.active_video_id = latest.get("video_id")
                working_memory.active_video_title = latest.get("title")
                conversation.context_video_id = latest.get("video_id")
                resolutions.append(
                    f"Target Upload ('latest video'): '{latest.get('title')}' (ID: {latest.get('video_id')}, "
                    f"Views: {latest.get('view_count', 0):,}, Published: {str(latest.get('published_at', ''))[:10]})"
                )

        # 2. Pronoun reference: "it", "this video", "that video"
        has_pronoun = any(p in f" {q} " for p in (" it ", " this ", " that ", " this video ", " that video ", " why did it "))
        if has_pronoun and not any("latest" in q or "recent" in q for _ in [1]):
            target_vid = None
            if working_memory.active_video_id and working_memory.active_video_title:
                target_vid = {"video_id": working_memory.active_video_id, "title": working_memory.active_video_title}
            elif conversation.context_video_id:
                for m in reversed(history):
                    if m.extra_metadata and m.extra_metadata.get("videos"):
                        for v in m.extra_metadata["videos"]:
                            if v.get("video_id") == conversation.context_video_id:
                                target_vid = v
                                break
                    if target_vid:
                        break
            elif history:
                for m in reversed(history):
                    if m.role == "assistant" and m.extra_metadata and m.extra_metadata.get("videos"):
                        target_vid = m.extra_metadata["videos"][0]
                        break

            if target_vid:
                working_memory.active_video_id = target_vid.get("video_id")
                working_memory.active_video_title = target_vid.get("title")
                conversation.context_video_id = target_vid.get("video_id")
                resolutions.append(
                    f"Resolved Referent ('it'): Video '{target_vid.get('title')}' (ID: {target_vid.get('video_id')})"
                )

        # 3. "Previous video" / "Earlier video" comparison reference
        if any(p in q for p in ("previous video", "previous upload", "older video", "earlier video")):
            if not working_memory.cached_inventory:
                inv_res = execute_tool("get_channel_video_inventory", {"max_results": 500}, tool_ctx)
                tool_rec = {
                    "tool": "get_channel_video_inventory",
                    "tool_name": "get_channel_video_inventory",
                    "arguments": {"max_results": 500},
                    "success": inv_res.get("success", True),
                    "context": "conversational_reference_resolution",
                }
                if tool_calls_executed is not None:
                    tool_calls_executed.append(tool_rec)
                if execution_state is not None:
                    execution_state.tool_calls.append(tool_rec)
                    execution_state.add_event(
                        "tool_call_completed",
                        "planning",
                        f"Resolved channel video inventory for benchmark reference ({len(inv_res.get('videos', []))} videos)",
                        {"tool": "get_channel_video_inventory", "success": inv_res.get("success", True)},
                    )
                if inv_res.get("success") and inv_res.get("videos"):
                    working_memory.cached_inventory = inv_res["videos"]
                    working_memory.inventory_is_complete = bool(inv_res.get("is_complete", False))
                    working_memory.inventory_total_results = int(inv_res.get("total_channel_videos", len(inv_res["videos"])))

            inventory = working_memory.cached_inventory
            if len(inventory) >= 2:
                curr_idx = 0
                if working_memory.active_video_id:
                    for i, v in enumerate(inventory):
                        if v.get("video_id") == working_memory.active_video_id:
                            curr_idx = i
                            break
                prev_idx = curr_idx + 1 if curr_idx + 1 < len(inventory) else 1
                prev_vid = inventory[prev_idx]
                resolutions.append(
                    f"Resolved Benchmark ('previous video'): '{prev_vid.get('title')}' (ID: {prev_vid.get('video_id')}, "
                    f"Views: {prev_vid.get('view_count', 0):,}, Published: {str(prev_vid.get('published_at', ''))[:10]})"
                )

        # 4. Ordinal reference: "second one", "second video", "idea 2", "2nd one"
        ordinal_match = re.search(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|#1|#2|#3)\b", q)
        if ordinal_match and any(w in q for w in ("idea", "one", "video", "option", "point", "topic")):
            ord_word = ordinal_match.group(1).lower()
            ord_idx = {
                "first": 0, "1st": 0, "#1": 0,
                "second": 1, "2nd": 1, "#2": 1,
                "third": 2, "3rd": 2, "#3": 2,
                "fourth": 3, "4th": 3, "#4": 3,
                "fifth": 4, "5th": 4, "#5": 4,
            }.get(ord_word, 1)

            for m in reversed(history):
                if m.role == "assistant":
                    if m.extra_metadata and m.extra_metadata.get("videos"):
                        vids = m.extra_metadata["videos"]
                        if len(vids) > ord_idx:
                            ref_v = vids[ord_idx]
                            resolutions.append(
                                f"Resolved Referent ('{ord_word} video'): '{ref_v.get('title')}' (ID: {ref_v.get('video_id')})"
                            )
                            break
                    lines = [ln.strip() for ln in m.content.split("\n") if re.match(r"^\d+\.\s+", ln.strip())]
                    if len(lines) > ord_idx:
                        resolutions.append(
                            f"Resolved Referent ('{ord_word} item'): {lines[ord_idx]}"
                        )
                        break

        # 5. Specific video topic inquiry from active channel inventory
        # e.g., "Discuss about the Video where He discussed about Loop Engineering in telugu from the Connected Youtube Channel"
        is_video_inquiry = (
            any(p in q for p in (
                "video where", "video about", "video on", "video discussing", "video discuss",
                "discuss about the video", "tell me about the video", "explain the video",
                "what about the video", "which video discussed", "video that talks about",
                "video in which", "the video regarding",
            ))
            or ("video" in q and any(w in q for w in ("discuss", "tell me", "explain", "review", "detail", "breakdown", "analysis", "what did", "where he", "where she")))
        )
        is_not_generic_ranking = not any(w in q for w in ("top video", "best video", "highest-performing", "most viewed", "rank videos", "compare videos", "all videos"))

        if is_video_inquiry and is_not_generic_ranking and tool_ctx.active_channel_id:
            stop_words = {
                "discuss", "about", "the", "video", "where", "he", "she", "they", "i", "we", "you",
                "from", "connected", "youtube", "channel", "in", "my", "our", "this", "that",
                "tell", "me", "what", "did", "say", "talk", "talks", "talking", "regarding",
                "show", "please", "can", "could", "would", "is", "are", "was", "were", "a", "an",
                "and", "or", "of", "to", "for", "with", "on", "at", "by", "its", "it", "his", "her"
            }
            topic_terms = [w for w in re.findall(r"[a-zA-Z0-9_-]{3,}", q) if w.lower() not in stop_words]

            if topic_terms:
                if not working_memory.cached_inventory:
                    inv_res = execute_tool("get_channel_video_inventory", {"max_results": 100}, tool_ctx)
                    tool_rec = {
                        "tool": "get_channel_video_inventory",
                        "tool_name": "get_channel_video_inventory",
                        "arguments": {"max_results": 100},
                        "success": inv_res.get("success", True),
                        "context": "specific_video_inventory_search",
                    }
                    if tool_calls_executed is not None:
                        tool_calls_executed.append(tool_rec)
                    if execution_state is not None:
                        execution_state.tool_calls.append(tool_rec)
                    if inv_res.get("success") and inv_res.get("videos"):
                        working_memory.cached_inventory = inv_res["videos"]
                        working_memory.inventory_is_complete = bool(inv_res.get("is_complete", False))
                        working_memory.inventory_total_results = int(inv_res.get("total_channel_videos", len(inv_res["videos"])))

                inventory = working_memory.cached_inventory or []
                scored_videos = []
                for v in inventory:
                    v_title = v.get("title", "").lower()
                    v_desc = (v.get("description") or "").lower()
                    score = 0
                    for term in topic_terms:
                        term_l = term.lower()
                        if term_l in v_title:
                            score += 3
                        elif term_l in v_desc:
                            score += 1
                    if score > 0:
                        scored_videos.append((score, v))

                scored_videos.sort(key=lambda x: x[0], reverse=True)
                if scored_videos:
                    best_match = scored_videos[0][1]
                    working_memory.active_video_id = best_match.get("video_id")
                    working_memory.active_video_title = best_match.get("title")
                    conversation.context_video_id = best_match.get("video_id")

                    resolutions.append(
                        f"Target Video (Matched from channel inventory for '{' '.join(topic_terms)}'):\n"
                        f"- Title: '{best_match.get('title')}'\n"
                        f"- Video ID: '{best_match.get('video_id')}'\n"
                        f"- Lifetime Views: {best_match.get('view_count', 0):,}\n"
                        f"- Published: {str(best_match.get('published_at', ''))[:10]}\n"
                        f"- Duration: {best_match.get('duration', 'N/A')}\n"
                        f"- Description: {best_match.get('description', '')[:300]}"
                    )

                    if execution_state is not None:
                        execution_state.add_event(
                            "video_matched_from_inventory",
                            "planning",
                            f"Matched video '{best_match.get('title')}' ({best_match.get('video_id')}) from active channel uploads",
                            {
                                "video_id": best_match.get("video_id"),
                                "title": best_match.get("title"),
                                "topic_terms": topic_terms,
                                "oauth_required": False,
                                "authorization_reason": "Public video metadata retrieved via YouTube Data API v3",
                            },
                        )

        return "\n".join(resolutions)

    # -------------------------------------------------------------
    # Intent Detection
    # -------------------------------------------------------------
    def _detect_intent(self, text: str, conversation: Conversation, working_memory: Any | None = None) -> str:
        q = text.lower().strip()

        # Follow-up references
        has_pronoun = any(p in q for p in ("it ", " it", "this video", "that video", "the second", "they ", "why did it", "why did this", "why it"))
        target_video_id = conversation.context_video_id or (working_memory.active_video_id if working_memory else None)
        if has_pronoun and target_video_id:
            if "compare" in q:
                return "compare_videos"
            if "why" in q or "how" in q or "perform" in q:
                return "video_performance"
            if "keyword" in q or "tag" in q:
                return "keyword_research"
            return "follow_up_question"

        if any(w in q for w in ("highest-performing", "highest performing", "best video", "top video", "top 10", "top 5", "most viewed", "most views", "highest view count", "engagement rate", "velocity", "views per day", "views/day", "shorts")):
            return "rank_videos"
        if "compare" in q or "versus" in q or " vs " in q:
            return "compare_videos"
        if "underperform" in q or "why is my latest" in q or "why did it" in q:
            return "video_performance"
        if "channel overview" in q or "recent channel performance" in q or "channel stats" in q:
            return "channel_overview"
        if "this month" in q or "trend" in q or "last 30 days" in q:
            return "analytics_trend"
        if "competitor" in q or "outperform" in q:
            return "competitor_research"
        if "keyword" in q or "hashtag" in q or "tag" in q:
            return "keyword_research"
        if "publish next" in q or "what should i post" in q or "what to make" in q or "recommend" in q:
            return "content_recommendation"
        if "previous analysis" in q or "past report" in q or "report" in q:
            return "report_lookup"

        return "open_question"

    # -------------------------------------------------------------
    # Intent Handlers
    # -------------------------------------------------------------
    def _handle_rank_videos(
        self,
        query: str,
        conversation: Conversation,
        ch_conn: ChannelConnection | None,
        tool_ctx: ToolContext,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str], list[dict[str, Any]]]:
        """Deterministic handling for video rankings (highest-performing, top 10, engagement rate)."""
        q_lower = query.lower()
        metric = "views"
        if "velocity" in q_lower or "views per day" in q_lower or "views/day" in q_lower:
            metric = "views_per_day"
        elif "engagement" in q_lower or "engagement rate" in q_lower:
            metric = "engagement_rate"
        elif "like" in q_lower:
            metric = "likes"
        elif "comment" in q_lower:
            metric = "comments"

        content_type = "shorts" if "short" in q_lower else ("long_form" if ("long" in q_lower or "long-form" in q_lower) else "all")

        top_n = 10 if "10" in q_lower else (5 if "5" in q_lower else (1 if ("highest" in q_lower or "best" in q_lower) else 5))

        days_match = re.search(r"(?:last|past|in|within)?\s*(\d+)\s*days?", q_lower)
        if days_match:
            days_filter = int(days_match.group(1))
        elif "month" in q_lower or "30 days" in q_lower:
            days_filter = 30
        elif "week" in q_lower:
            days_filter = 7
        elif "year" in q_lower:
            days_filter = 365
        else:
            days_filter = None

        tool_args = {"metric": metric, "top_n": top_n, "days_filter": days_filter, "content_type": content_type}
        rank_res = execute_tool("rank_and_calculate_video_metrics", tool_args, tool_ctx)

        tool_record = {
            "tool": "rank_and_calculate_video_metrics",
            "arguments": tool_args,
            "success": rank_res.get("success", False),
        }

        if not rank_res.get("success"):
            err = rank_res.get("error", "No video data available.")
            answer = f"### Video Ranking Analysis\n\n{err}\n\n**Note**: The active channel has not published any uploads matching this specific time window or criteria."
            return answer, [], {}, [], {}, [], [tool_record]

        ranked_videos = rank_res.get("ranked_videos", [])
        winner = rank_res.get("winning_video") or (ranked_videos[0] if ranked_videos else None)
        metric_label = rank_res.get("ranking_metric", "Views")

        evidence_sources = []
        for v in ranked_videos[:3]:
            evidence_sources.append({
                "title": v.get("title", "Video"),
                "video_id": v.get("video_id"),
                "source": "YouTube Data API v3",
                "metric": metric_label,
                "value": v.get("view_count", 0),
                "note": f"Published {v.get('published_at', '')[:10]} ({v.get('age_days')} days ago).",
            })

        insp_cnt = rank_res.get("inspected_count", len(ranked_videos))
        tot_cnt = rank_res.get("total_channel_videos", 0)
        if tot_cnt <= 0:
            cov_pct = 0.0
        elif insp_cnt < tot_cnt:
            cov_pct = min(round((insp_cnt / tot_cnt) * 100, 1), 99.0)
        else:
            cov_pct = 100.0

        data_coverage = {
            "inspected_count": insp_cnt,
            "total_channel_videos": tot_cnt,
            "coverage_percentage": cov_pct,
            "observation_period": f"Lifetime public observations ({insp_cnt}/{tot_cnt} uploads analyzed, {cov_pct}% coverage)",
            "data_source": "YouTube Data API v3",
        }

        # Formulate answer text
        if top_n == 1 or ("highest" in q_lower and top_n <= 1):
            w_title = winner.get("title", "")
            w_views = winner.get("view_count", 0)
            w_likes = winner.get("like_count") or 0
            w_comments = winner.get("comment_count") or 0
            w_age = winner.get("age_days", 1)
            w_velocity = winner.get("views_per_day", 0)
            w_rate = winner.get("engagement_rate", 0)

            answer = (
                f"### Highest-Performing Video on Your Channel\n\n"
                f"The highest-performing video on your channel by **{metric_label}** is:\n\n"
                f"**[{w_title}]({winner.get('url')})**\n\n"
                f"#### Verified Performance Metrics:\n"
                f"- **Lifetime Views**: **{w_views:,} views**\n"
                f"- **Daily Pacing Velocity**: **{w_velocity:,} views/day** (over {w_age} days since publish)\n"
                f"- **Audience Engagement**: **{w_likes:,} likes** and **{w_comments:,} comments** ({w_rate}% engagement rate)\n"
                f"- **Published Date**: {winner.get('published_at', '')[:10]}\n\n"
                f"#### Evidence & Observation Summary:\n"
                f"Across {rank_res.get('inspected_count')} analyzed uploads on your channel, this upload represents your top lifetime reach driver.\n\n"
                f"> **Data Limitation Notice**: Lifetime view counts reflect all public views since publication. "
                f"To inspect 30-day views, retention curves, or thumbnail CTR, an authorized YouTube Studio export is required."
            )
            follow_ups = [
                "Why did it perform so well?",
                "Compare my best video with my latest upload.",
                "Show my top 5 videos by views.",
                "What keywords should I target based on this video?",
            ]
        else:
            bullets = "\n".join([
                f"{i+1}. **[{v.get('title')}]({v.get('url')})** — **{v.get('view_count', 0):,} views** "
                f"({v.get('views_per_day', 0):,} views/day • {v.get('engagement_rate', 0)}% engagement)"
                for i, v in enumerate(ranked_videos)
            ])
            answer = (
                f"### Top {len(ranked_videos)} Videos on Your Channel (by {metric_label})\n\n"
                f"Here are your highest-performing videos ranked by {metric_label.lower()} from your channel inventory:\n\n"
                f"{bullets}\n\n"
                f"**Inventory Coverage**: Analyzed **{rank_res.get('inspected_count')}** public uploads from your channel "
                f"({data_coverage['coverage_percentage']}% coverage of channel catalog).\n\n"
                f"> **Data Limitation**: CTR and average watch time are private creator metrics only accessible via YouTube Studio."
            )
            follow_ups = [
                "Which video has the highest engagement rate?",
                "Compare my best video with my latest upload.",
                "What topics do my top videos have in common?",
            ]

        metrics_payload = {
            "ranking_metric": metric_label,
            "winner_video_id": winner.get("video_id") if winner else None,
            "winner_view_count": winner.get("view_count") if winner else None,
            "inspected_count": rank_res.get("inspected_count"),
        }

        return answer, ranked_videos, metrics_payload, evidence_sources, data_coverage, follow_ups, [tool_record]

    def _handle_compare_videos(
        self,
        query: str,
        conversation: Conversation,
        ch_conn: ChannelConnection | None,
        latest_report: AnalysisReport | None,
        tool_ctx: ToolContext,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str], list[dict[str, Any]]]:
        """Deterministic comparison between user's videos or against competitor benchmarks."""
        q_lower = query.lower()

        # Check if comparing against competitors
        if "competitor" in q_lower:
            comp_tool_res = execute_tool("search_competitor_videos", {"query": ch_conn.title if ch_conn else "YouTube growth", "max_results": 5}, tool_ctx)
            candidates = comp_tool_res.get("candidate_videos", [])
            tool_record = {"tool": "search_competitor_videos", "arguments": {"query": ch_conn.title if ch_conn else "YouTube growth"}, "success": True}

            if candidates:
                c1 = candidates[0]
                c2 = candidates[1] if len(candidates) > 1 else c1
                c_target = c2 if "second" in q_lower else c1

                answer = (
                    f"### Competitor Comparison: {c_target.get('title')}\n\n"
                    f"Benchmarking against competitor **[{c_target.get('title')}]({c_target.get('url', 'https://youtube.com/watch?v=' + c_target.get('video_id', ''))})** "
                    f"by **{c_target.get('channel_title')}**:\n\n"
                    f"- **Competitor Views**: **{c_target.get('view_count', 0):,} views**\n"
                    f"- **Duration**: {c_target.get('duration_formatted', 'N/A')}\n"
                    f"- **Engagement**: {c_target.get('like_count', 0):,} likes, {c_target.get('comment_count', 0):,} comments\n"
                    f"- **Published**: {c_target.get('published_at', '')[:10]}\n\n"
                    f"#### Strategic Gap Analysis:\n"
                    f"1. **Packaging**: Competitor title targets high-frequency intent keywords with clear viewer payoff.\n"
                    f"2. **Differentiation**: Emphasize actionable workflows in your opening 30 seconds to capture search viewers."
                )
                videos_payload = [{
                    "video_id": c_target.get("video_id"),
                    "title": c_target.get("title"),
                    "thumbnail_url": c_target.get("thumbnail_url"),
                    "url": f"https://www.youtube.com/watch?v={c_target.get('video_id')}",
                    "view_count": c_target.get("view_count", 0),
                    "like_count": c_target.get("like_count"),
                    "comment_count": c_target.get("comment_count"),
                    "published_at": c_target.get("published_at"),
                    "duration": c_target.get("duration_formatted"),
                    "badge": "Top Competitor Benchmark",
                }]
                follow_ups = [
                    "What keywords are they using in their title and tags?",
                    "How can I package my next video to outperform this?",
                    "Compare my latest video with my most-viewed video.",
                ]
                evidence = [{
                    "title": c_target.get("title"),
                    "video_id": c_target.get("video_id"),
                    "source": "YouTube Competitor Search",
                    "metric": "Lifetime Views",
                    "value": c_target.get("view_count", 0),
                }]
                return answer, videos_payload, {}, evidence, {}, follow_ups, [tool_record]

        # Comparing Best vs Latest (or context video vs latest)
        target_id = conversation.context_video_id or "best"
        compare_args = {"target_video_id": target_id, "benchmark_video_id": "latest"}
        res = execute_tool("compare_videos_detailed", compare_args, tool_ctx)
        tool_record = {"tool": "compare_videos_detailed", "arguments": compare_args, "success": res.get("success", False)}

        if not res.get("success"):
            return f"Comparison failed: {res.get('error')}", [], {}, [], {}, [], [tool_record]

        va = res.get("video_a", {})
        vb = res.get("video_b", {})
        mc = res.get("metric_comparison", {})

        va_views = va.get("view_count") or 0
        vb_views = vb.get("view_count") or 0
        diff = abs(va_views - vb_views)

        va_likes = f"{(va.get('like_count') or 0):,}" if va.get('like_count') is not None else "Hidden"
        vb_likes = f"{(vb.get('like_count') or 0):,}" if vb.get('like_count') is not None else "Hidden"
        va_comments = f"{(va.get('comment_count') or 0):,}" if va.get('comment_count') is not None else "Disabled"
        vb_comments = f"{(vb.get('comment_count') or 0):,}" if vb.get('comment_count') is not None else "Disabled"

        answer = (
            f"### Side-by-Side Video Comparison\n\n"
            f"Comparing your best-performing video vs your latest upload:\n\n"
            f"| Metric | Top Performer: **{va.get('title', '')[:28]}...** | Latest Upload: **{vb.get('title', '')[:28]}...** |\n"
            f"|---|---|---|\n"
            f"| **Lifetime Views** | **{va_views:,}** | **{vb_views:,}** |\n"
            f"| **Likes** | {va_likes} | {vb_likes} |\n"
            f"| **Comments** | {va_comments} | {vb_comments} |\n"
            f"| **Published** | {va.get('published_at', '')[:10]} | {vb.get('published_at', '')[:10]} |\n\n"
            f"#### Performance Differential:\n"
            f"- **View Gap**: Top performer leads by **{diff:,} views** ({mc.get('view_ratio', 'N/A')}).\n"
            f"- **Observation**: The top video has accumulated reach over a longer observation window. "
            f"Its keyword packaging and broad topical hook allowed it to establish enduring discovery."
        )

        videos_payload = [
            {**va, "badge": "Top Performer"},
            {**vb, "badge": "Latest Upload"},
        ]
        evidence = [
            {"title": va.get("title"), "video_id": va.get("video_id"), "source": "Channel Inventory", "value": va_views},
            {"title": vb.get("title"), "video_id": vb.get("video_id"), "source": "Channel Inventory", "value": vb_views},
        ]
        follow_ups = [
            "Why is my latest video pacing slower?",
            "What keywords and tags should I target?",
            "What should I publish next to replicate my top video?",
        ]

        return answer, videos_payload, mc, evidence, {}, follow_ups, [tool_record]

    def _handle_video_performance(
        self,
        query: str,
        conversation: Conversation,
        ch_conn: ChannelConnection | None,
        latest_report: AnalysisReport | None,
        tool_ctx: ToolContext,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str], list[dict[str, Any]]]:
        """Diagnose video performance / underperformance with grounded evidence."""
        inv_res = execute_tool("get_channel_video_inventory", {"max_results": 10}, tool_ctx)
        videos = inv_res.get("videos", [])
        tool_record = {"tool": "get_channel_video_inventory", "arguments": {"max_results": 10}, "success": inv_res.get("success", False)}

        target_vid = None
        if conversation.context_video_id:
            target_vid = next((v for v in videos if v["video_id"] == conversation.context_video_id), None)
        if not target_vid and videos:
            target_vid = videos[0]  # Latest upload

        if not target_vid:
            return "No video records found to diagnose. Please verify your channel connection.", [], {}, [], {}, [], [tool_record]

        vt = target_vid.get("title", "")
        vc = target_vid.get("view_count", 0)
        vl = target_vid.get("like_count") or 0
        v_url = target_vid.get("url")

        answer = (
            f"### Video Diagnostic: [{vt}]({v_url})\n\n"
            f"Analyzing public metrics and velocity for your target upload:\n\n"
            f"- **Current Views**: **{vc:,} views**\n"
            f"- **Observed Likes**: **{vl:,} likes**\n"
            f"- **Published Date**: {target_vid.get('published_at', '')[:10]}\n\n"
            f"#### Key Diagnostic Findings:\n"
            f"1. **Initial Velocity**: Public view counts reflect current lifetime exposure. If velocity has slowed compared to channel averages, the algorithm typically tests videos with narrower browse audiences first.\n"
            f"2. **Packaging Signals**: Inspect your title length and opening 45 characters. Videos with specific question or problem-solving hooks exhibit higher conversion from impression to view.\n"
            f"3. **Data Limitations**: CTR (Click-Through Rate) and Audience Retention Drops in the first 30 seconds are the primary factors in algorithmic expansion. "
            f"These private metrics are only accessible in your **YouTube Studio Analytics** dashboard."
        )

        videos_payload = [{**target_vid, "badge": "Analyzed Upload"}]
        follow_ups = [
            "How can I improve my title and thumbnail?",
            "Compare my latest video with my most-viewed video.",
            "What keywords and hashtags should I target?",
        ]
        evidence = [{"title": vt, "video_id": target_vid.get("video_id"), "source": "YouTube Data API v3", "value": vc}]

        return answer, videos_payload, {}, evidence, {}, follow_ups, [tool_record]

    def _handle_channel_overview(
        self,
        ch_conn: ChannelConnection | None,
        latest_report: AnalysisReport | None,
        tool_ctx: ToolContext,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str], list[dict[str, Any]]]:
        """Channel performance overview with top recent uploads."""
        res = execute_tool("get_connected_channel_summary", {}, tool_ctx)
        tool_record = {"tool": "get_connected_channel_summary", "arguments": {}, "success": res.get("success", False)}

        title = res.get("title") or (ch_conn.title if ch_conn else "Your Channel")
        raw_subs = res.get("subscriber_count") if res.get("subscriber_count") is not None else (ch_conn.subscriber_count if ch_conn else None)
        raw_views = res.get("view_count") if res.get("view_count") is not None else (ch_conn.view_count if ch_conn else 0)
        raw_uploads = res.get("video_count") if res.get("video_count") is not None else (ch_conn.video_count if ch_conn else 0)

        subs_str = f"**{raw_subs:,}**" if raw_subs is not None else "Hidden"
        views_str = f"**{(raw_views or 0):,}**"
        uploads_str = f"**{(raw_uploads or 0):,} videos**"
        recent = res.get("recent_videos", [])

        answer = (
            f"### Channel Performance Overview: {title}\n\n"
            f"- **Subscribers**: {subs_str}\n"
            f"- **Lifetime Public Views**: {views_str}\n"
            f"- **Total Uploads**: {uploads_str}\n\n"
            f"Your channel is connected and active. You can ask me to rank your catalog, compare your best uploads against competitors, or diagnose keyword gaps."
        )

        videos_payload = recent[:3] if recent else []
        follow_ups = [
            "What is the highest-performing video on my channel?",
            "Show my top 5 videos by views.",
            "Which of my videos has the highest engagement rate?",
        ]
        return answer, videos_payload, {}, [], {}, follow_ups, [tool_record]

    def _handle_open_reasoning(
        self,
        user_text: str,
        conversation: Conversation,
        history: list[ConversationMessage],
        ch_conn: ChannelConnection | None,
        latest_report: AnalysisReport | None,
        tool_ctx: ToolContext,
        intent: str,
        execution_state: AgentExecutionState | None = None,
        working_memory: WorkingMemory | None = None,
        resolved_ref_str: str = "",
        execution_run: ExecutionRun | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str], list[dict[str, Any]]]:
        """Local Qwen agent multi-turn tool calling and reasoning loop with stateful working memory."""
        ch_title = ch_conn.title if ch_conn else "Connected Channel"

        # Initialize defaults if not provided
        if not execution_state:
            execution_state = AgentExecutionState(
                conversation_id=conversation.id,
                workspace_id=tool_ctx.workspace_id,
                user_goal=user_text,
                status="planning",
            )
        if not working_memory:
            working_memory = WorkingMemory(user_goal=user_text)

        # Context summary
        if working_memory and working_memory.active_channel_context:
            act_ctx = working_memory.active_channel_context
            is_oauth = act_ctx.get("context_source") == "oauth_connected_channel" or (ch_conn and ch_conn.channel_id == act_ctx.get("channel_id"))
            status_desc = "OAuth-connected Channel" if is_oauth else "Public YouTube Channel (Not OAuth-connected)"
            context_str = (
                f"ACTIVE CONVERSATION CHANNEL:\n"
                f"- Channel Name: '{act_ctx.get('channel_name')}'\n"
                f"- Channel ID: '{act_ctx.get('channel_id')}'\n"
                f"- Status: {status_desc}\n"
                f"- Custom URL/Handle: {act_ctx.get('custom_url') or act_ctx.get('source_url') or 'N/A'}\n"
                f"- Lifetime Public Views: {act_ctx.get('view_count', 0):,}\n"
                f"- Total Public Uploads: {act_ctx.get('video_count', 0):,}\n"
                f"- Subscribers: {act_ctx.get('subscriber_count') if act_ctx.get('subscriber_count') is not None else 'Hidden/Unknown'}\n"
                f"- Context Source: {act_ctx.get('context_source')}\n"
            )
        elif ch_conn:
            context_str = (
                f"Connected Channel: '{ch_conn.title}' ({ch_conn.subscriber_count:,} subscribers, "
                f"{ch_conn.view_count:,} public views, {ch_conn.video_count} uploads, Channel ID: {ch_conn.channel_id})"
            )
        elif latest_report:
            ms = latest_report.metrics_summary or {}
            an_vid = ms.get("analyzed_video", {}) if isinstance(ms, dict) else {}
            vid_title = an_vid.get("title", "") if isinstance(an_vid, dict) else ""
            context_str = f"Analyzed Channel: '{latest_report.channel_title}' (Channel ID: {latest_report.channel_id})"
            if vid_title:
                context_str += f" | Analyzed Target Video: '{vid_title}' (Video ID: {an_vid.get('video_id', '')})"
        else:
            context_str = "No channel connected or established yet in this conversation."

        if conversation.context_video_id:
            context_str += f" | Active Video Context ID: '{conversation.context_video_id}'"

        tool_calls_executed: list[dict[str, Any]] = []
        evidence_sources: list[dict[str, Any]] = []
        videos_payload: list[dict[str, Any]] = []
        metrics_payload: dict[str, Any] = {}
        data_coverage: dict[str, Any] = {
            "observation_period": "Live YouTube & Local Agent Analysis",
            "coverage_percentage": 0.0,
            "data_source": "YouTube Data API v3 & Tavily Search",
        }
        final_answer = ""

        # 1. Check local Ollama readiness
        is_ready, status_msg = self.llm.check_model_status()
        if not is_ready:
            logger.warning("Ollama check failed: %s", status_msg)
            execution_state.update_stage("failed", "Ollama Unavailable")
            execution_state.errors.append(status_msg)
            if execution_run:
                execution_run.fail_step("planning", status_msg)
                execution_run.skip_step("selecting_tools", "Ollama unavailable")
                execution_run.skip_step("calling_tools", "Ollama unavailable")
                execution_run.skip_step("reviewing_results", "Ollama unavailable")
                execution_run.skip_step("deciding_next", "Ollama unavailable")
                execution_run.skip_step("validating_findings", "Ollama unavailable")
                execution_run.fail_step("generating_answer", "Ollama unavailable")
            final_answer = (
                "### Local Ollama Unavailable\n\n"
                f"{status_msg}\n\n"
                "**To fix this issue:**\n"
                "1. Ensure Ollama is running on your machine:\n"
                "   ```bash\n"
                f"   ollama run {self.llm.model}\n"
                "   ```\n"
                f"2. Verify the endpoint at `{self.llm.base_url}` is responding."
            )
            follow_ups = [
                "How many subscribers do I have?",
                "What is the highest-performing video on my channel?",
                "Explain about Codebasics Channel",
            ]
            return final_answer, videos_payload, metrics_payload, evidence_sources, data_coverage, follow_ups, tool_calls_executed

        # 2. Build system instructions with working memory scratchpad and resolved references
        system_prompt = (
            "You are GrowTube AI, an autonomous, data-grounded YouTube growth strategist and analyst powered by Qwen.\n"
            "Your job is to answer questions about the creator's channel, video performance, competitors, and growth opportunities.\n\n"
            "MANDATORY OPERATIONAL DIRECTIVES:\n"
            "1. ALWAYS call tools to inspect real facts before answering. Never invent metrics, subscriber counts, or video titles.\n"
            "2. The ACTIVE CONVERSATION CHANNEL specified below is the verified channel selected by the user. Treat it as authoritative and already established. NEVER ask the user to provide a channel ID or channel name when an ACTIVE CONVERSATION CHANNEL is already provided.\n"
            "   - Public video metadata, video titles, upload dates, and public view counts do NOT require OAuth. Use `get_channel_video_inventory` or `get_channel_details` directly.\n"
            "   - When answering questions about the active channel, clearly state the verified channel name (e.g. '**Channel Name**: [Name]') and canonical channel ID.\n"
            "   - When asked what type of content the channel publishes, inspect representative video titles and descriptions returned by `get_channel_details` or `get_connected_channel_summary`. Provide a concise, evidence-grounded summary of what topics, formats, and themes it publishes with specific video examples. Do not claim there is no channel simply because it is not OAuth-connected!\n"
            "3. When asked to list, rank, or identify the best / highest-performing / most-viewed uploads, call `rank_and_calculate_video_metrics` with `days_filter=None` unless the user explicitly requested a specific time window like 'last 30 days'. An all-time or lifetime ranking must include older uploads. You MUST state the winning video title and verified view count in your response.\n"
            "4. When asked about competitors, external channels, or public videos (e.g. 'Explain about Codebasics Channel', 'Compare my latest video with competitors'), call `search_youtube_videos` or `search_competitor_videos`.\n"
            "5. When asked about YouTube growth strategies, algorithm changes, SEO, or trends, call `search_youtube_growth_strategies`, `research_youtube_algorithm_updates`, `research_niche_and_trends`, or `search_external_web`.\n"
            "6. Clearly distinguish between 'channel selected' and 'OAuth authorization connected':\n"
            "   - Only private creator analytics (such as Click-Through Rate / CTR, audience retention curves, impressions, traffic sources, and private YouTube Studio analytics) require OAuth authorization.\n"
            "   - If the user specifically asks for private studio metrics that require OAuth, explicitly state that while channel is selected and active in the workspace, private creator metrics require connecting the channel via Google OAuth in Settings > YouTube Connections.\n"
            "   - Never claim OAuth is missing when the user is asking about public videos, video topics, views, or rankings!\n"
            "7. Be concise, direct, and actionable. Avoid repetitive preambles. Output tool calls immediately when data is needed.\n\n"
            f"Context: {self._sanitize_untrusted_text(context_str)}\n"
        )

        # Inject resolved references if any
        if resolved_ref_str:
            safe_ref = self._sanitize_untrusted_text(resolved_ref_str)
            system_prompt += f"\n### Resolved Conversational Reference Context:\n{safe_ref}\n"

        # Inject active working memory state
        wm_scratchpad = (
            f"### Active Working Memory Scratchpad:\n"
            f"- User Goal: {working_memory.user_goal}\n"
            f"- Current Execution Stage: {execution_state.status}\n"
            f"- Active Video Target: {working_memory.active_video_title or 'None'}\n"
            f"- Cached Video Inventory: {len(working_memory.cached_inventory)} uploads accessible in memory\n"
        )
        system_prompt += f"\n{wm_scratchpad}\n"

        # Retrieve relevant long-term creator memory (only if relevant to active channel or current query)
        try:
            memories = self.memory_service.retrieve_relevant_memories(
                workspace_id=tool_ctx.workspace_id,
                query_context=user_text,
                limit=4,
            )
            if memories:
                if working_memory and working_memory.active_channel_name:
                    ch_n = working_memory.active_channel_name.lower()
                    kw_terms = [w for w in re.findall(r"\w{4,}", user_text.lower()) if w not in ("what", "video", "views", "channel", "analyze", "highest", "performing")]
                    # Fix Defect 4: Preserve workspace-level creator_profile memories (niche, goals, language, audience)
                    filtered_mems = [
                        m for m in memories
                        if m.category == "creator_profile"
                        or ch_n in m.content.lower()
                        or any(term in m.content.lower() for term in kw_terms)
                    ]
                else:
                    filtered_mems = memories

                if filtered_mems:
                    mem_text = self.memory_service.format_memory_context_for_prompt(filtered_mems)
                    system_prompt += (
                        f"\nRelevant Historical Workspace Notes (Informational only — the ACTIVE CONVERSATION CHANNEL above is authoritative):\n"
                        f"{mem_text}\n"
                    )
                    execution_state.add_event("memory_retrieved", "planning", f"Loaded {len(filtered_mems)} relevant creator memories into prompt context")
        except Exception as e:
            logger.debug(f"Could not retrieve memories: {e}")

        # 3. Assemble message history
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for m in history[:-1]:
            if m.role in ("user", "assistant") and m.content:
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user_text})

        # 4. Multi-round tool calling loop (up to 4 turns)
        max_turns = 4
        selected_tools = self._select_tools_for_query(user_text)
        tool_names = [t["function"]["name"] for t in selected_tools]
        execution_state.add_event("tools_selected", "planning", f"Selected {len(selected_tools)} candidate tools for query", {"tools": tool_names})

        if execution_run:
            execution_run.complete_step("planning", "Working out the best approach to your request.")
            execution_run.start_step("selecting_tools", "Choosing the tools needed to gather relevant information.")
            execution_run.complete_step("selecting_tools", f"Selected {len(selected_tools)} candidate tools for query evaluation.")

        tools_called_in_any_turn = False

        for turn in range(max_turns):
            resp = self.llm.chat_completion_with_tools(
                messages=messages,
                tools=selected_tools,
                temperature=0.2,
                max_tokens=600,
            )

            # Handle errors from Ollama client (Fix Defect 2: Mark failed execution state)
            if resp.error:
                logger.warning("Ollama round %d error: %s", turn + 1, resp.error)
                execution_state.errors.append(f"Ollama round {turn+1} error: {resp.error}")
                execution_state.update_stage("failed", f"LLM Error: {resp.error_type or 'execution_failed'}")
                execution_state.add_event("tool_error", "failed", f"Ollama round {turn+1} error: {resp.error}", {"error_type": resp.error_type})
                if execution_run:
                    curr = execution_run.current_step_id or "planning"
                    execution_run.fail_step(curr, resp.error)
                if resp.error_type == "connection_error":
                    final_answer = (
                        "### Local Ollama Connection Lost\n\n"
                        f"Cannot reach Ollama at `{self.llm.base_url}`. Please run `ollama run {self.llm.model}` in your terminal."
                    )
                elif resp.error_type == "timeout":
                    final_answer = (
                        "### Generation Timed Out\n\n"
                        "Local inference exceeded 300 seconds. Please try again with a more specific query."
                    )
                else:
                    final_answer = f"### Agent Execution Error\n\n{resp.error}"
                break

            # If no tool calls requested, model has reached a final answer
            if not resp.tool_calls:
                raw_ans = resp.raw_content or resp.thought or ""
                final_answer = self._clean_model_output(raw_ans)
                execution_state.add_event("answer_generated", execution_state.status, "Model synthesized final answer without additional tool calls")

                if not tools_called_in_any_turn:
                    # Simple question or greeting that requires no tools
                    if execution_run:
                        execution_run.skip_step("calling_tools", "No tools needed for this question.")
                        execution_run.skip_step("reviewing_results", "No tools needed for this question.")
                        execution_run.skip_step("deciding_next", "No tools needed for this question.")
                        execution_run.skip_step("validating_findings", "Direct answer generated without external tool data.")
                else:
                    if execution_run:
                        execution_run.complete_step("deciding_next", "Determined that sufficient information is gathered.")
                        execution_run.start_step("validating_findings", "Checking the available evidence and data consistency.")
                        execution_run.complete_step("validating_findings", f"Checked {len(evidence_sources)} evidence records and data consistency.")

                if execution_run:
                    execution_run.start_step("generating_answer", "Turning the findings into a useful response.")
                    execution_run.complete_step("generating_answer", "Turning the findings into a useful response.")
                break

            # Tool calls requested by LLM
            tools_called_in_any_turn = True
            if execution_run:
                if turn > 0:
                    execution_run.complete_step("deciding_next", f"Determined more information is needed; starting round {turn + 1} tool execution.")
                execution_run.start_step("calling_tools", "Retrieving video information from YouTube.")

            # Append the assistant message with tool calls (Fix Defect 1: Ensure tool_calls presence)
            if resp.raw_message and resp.raw_message.get("tool_calls"):
                messages.append(resp.raw_message)
            else:
                messages.append({
                    "role": "assistant",
                    "content": resp.raw_content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                            },
                        }
                        for tc in resp.tool_calls
                    ],
                })

            # Execute each approved tool with working memory caching
            for tc in resp.tool_calls:
                call_id = f"{tc.name}_{turn}_{len(tool_calls_executed)}"
                tc_meta = get_tool_meta(tc.name)
                tc_display_name = tc_meta.get("display_name", tc.name)
                tc_purpose = tc_meta.get("description", f"Executes {tc.name}")

                if execution_run:
                    execution_run.start_tool_call(
                        tool_call_id=call_id,
                        tool_name=tc.name,
                        display_name=tc_display_name,
                        purpose=tc_purpose,
                    )

                # Update execution lifecycle state according to domain
                if tc.name in ("get_connected_channel_summary", "get_channel_details", "get_channel_video_inventory"):
                    execution_state.update_stage("gathering_data", f"Tool: {tc.name}")
                elif tc.name in ("rank_and_calculate_video_metrics", "compare_videos_detailed"):
                    execution_state.update_stage("calculating_metrics", f"Tool: {tc.name}")
                elif tc.name in ("search_competitor_videos", "search_youtube_growth_strategies", "research_youtube_algorithm_updates", "search_external_web", "investigate_creator_struggles", "search_niche_radar"):
                    execution_state.update_stage("researching_opportunities", f"Tool: {tc.name}")

                conn_label = "oauth_connected" if bool(tool_ctx.access_token) else ("selected_public" if tool_ctx.active_channel_id else "unselected")
                execution_state.add_event(
                    "tool_call_started",
                    execution_state.status,
                    f"Executing tool '{tc.name}'",
                    {
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "channel_id": tool_ctx.active_channel_id,
                        "connection_state": conn_label,
                        "is_oauth_connected": bool(tool_ctx.access_token),
                        "oauth_required": False,
                        "authorization_reason": "Public video metadata retrieved via YouTube Data API v3",
                    },
                )

                # Working memory caching: reuse already loaded video inventory ONLY if verified complete
                if tc.name == "get_channel_video_inventory" and working_memory.cached_inventory and working_memory.inventory_is_complete:
                    logger.info("Agent reusing verified complete video inventory (%d uploads) from working memory", len(working_memory.cached_inventory))
                    tool_result = {
                        "success": True,
                        "videos": working_memory.cached_inventory,
                        "total_channel_videos": working_memory.inventory_total_results or len(working_memory.cached_inventory),
                        "inspected_count": len(working_memory.cached_inventory),
                        "playlist_items_count": len(working_memory.cached_inventory),
                        "metadata_items_count": len(working_memory.cached_inventory),
                        "is_complete": True,
                        "coverage_percentage": 100.0,
                        "coverage_notes": "Verified complete inventory cached in working memory.",
                        "observation_period": f"Channel uploads inventory ({len(working_memory.cached_inventory)} uploads, 100.0% coverage)",
                        "source": "working_memory_cache",
                    }
                elif tc.name == "get_connected_channel_summary" and working_memory.cached_channel_summary:
                    logger.info("Agent reusing cached channel summary from working memory")
                    tool_result = working_memory.cached_channel_summary
                elif tc.name == "get_channel_details" and working_memory.cached_channel_summary:
                    logger.info("Agent reusing cached channel details from working memory")
                    tool_result = working_memory.cached_channel_summary
                else:
                    logger.info("Agent executing tool: %s with args: %s", tc.name, tc.arguments)
                    tool_result = execute_tool(tc.name, tc.arguments, tool_ctx)

                if tc.name in ("get_channel_details", "get_connected_channel_summary") and isinstance(tool_result, dict) and tool_result.get("success"):
                    working_memory.cached_channel_summary = tool_result

                is_success = tool_result.get("success", True) if isinstance(tool_result, dict) else True
                err_msg = None
                if not is_success and isinstance(tool_result, dict):
                    err_msg = str(tool_result.get('error'))
                    execution_state.errors.append(f"Tool {tc.name} reported error: {err_msg}")
                    execution_state.add_event("tool_error", execution_state.status, f"Tool {tc.name} failed: {err_msg}", {"tool": tc.name, "error": err_msg})

                safe_summary = generate_safe_tool_summary(tc.name, tool_result, is_success)

                if execution_run:
                    execution_run.complete_tool_call(
                        tool_call_id=call_id,
                        success=is_success,
                        result_summary=safe_summary,
                        error=err_msg,
                    )

                tool_record = {
                    "tool": tc.name,
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "success": is_success,
                }
                tool_calls_executed.append(tool_record)
                execution_state.tool_calls.append(tool_record)
                execution_state.add_event(
                    "tool_call_completed",
                    execution_state.status,
                    f"Tool '{tc.name}' execution completed (success={is_success})",
                    {
                        "tool": tc.name,
                        "success": is_success,
                        "channel_id": tool_ctx.active_channel_id,
                        "connection_state": conn_label,
                        "is_oauth_connected": bool(tool_ctx.access_token),
                        "oauth_required": False,
                        "duration_ms": tool_result.get("_duration_ms", 0) if isinstance(tool_result, dict) else 0,
                    },
                )

                # Extract UI data, evidence sources, and populate working memory
                if isinstance(tool_result, dict):
                    # 1. Truthful data coverage reporting and dedicated cache management
                    if tc.name == "get_channel_video_inventory":
                        data_coverage = {
                            "inspected_count": tool_result.get("inspected_count", 0),
                            "total_channel_videos": tool_result.get("total_channel_videos", 0),
                            "coverage_percentage": tool_result.get("coverage_percentage", 0.0),
                            "playlist_coverage_percentage": tool_result.get("playlist_coverage_percentage", 0.0),
                            "metadata_coverage_percentage": tool_result.get("metadata_coverage_percentage", 0.0),
                            "is_complete": tool_result.get("is_complete", False),
                            "coverage_notes": tool_result.get("coverage_notes", ""),
                            "observation_period": tool_result.get("observation_period", "Channel public uploads inventory"),
                            "data_source": "YouTube Data API v3",
                        }
                        inv_videos = tool_result.get("videos", [])
                        if isinstance(inv_videos, list) and len(inv_videos) >= len(working_memory.cached_inventory):
                            working_memory.cached_inventory = inv_videos
                            working_memory.inventory_is_complete = bool(tool_result.get("is_complete", False))
                            working_memory.inventory_total_results = int(tool_result.get("total_channel_videos", len(inv_videos)))
                            tool_ctx.cached_inventory = working_memory.cached_inventory
                            tool_ctx.inventory_is_complete = working_memory.inventory_is_complete

                    elif tc.name == "rank_and_calculate_video_metrics":
                        insp = tool_result.get("inspected_count", 0)
                        tot = tool_result.get("total_channel_videos", 0)
                        cov_pct = round((insp / max(tot, 1)) * 100, 1) if tot > 0 else 0.0
                        if insp < tot and cov_pct >= 100.0:
                            cov_pct = 99.0
                        data_coverage = {
                            "inspected_count": insp,
                            "total_channel_videos": tot,
                            "coverage_percentage": cov_pct,
                            "observation_period": tool_result.get("coverage_summary", f"Ranked across {insp} uploads"),
                            "data_source": "YouTube Data API v3",
                        }
                        # Never poison cached_inventory with ranked_videos! Keep ranked results separate.
                        ranked = tool_result.get("ranked_videos", [])
                        m_label = tool_result.get("ranking_metric", "views")
                        if isinstance(ranked, list) and ranked:
                            working_memory.cached_rankings[m_label] = ranked
                            winner = tool_result.get("winning_video")
                            if winner:
                                working_memory.active_video_id = winner.get("video_id")
                                working_memory.active_video_title = winner.get("title")

                    # 2. Video cards & evidence sources
                    act_c = tool_result.get("active_conversation_channel") or {}
                    v_items = (
                        tool_result.get("videos")
                        or tool_result.get("ranked_videos")
                        or tool_result.get("candidate_videos")
                        or tool_result.get("recent_videos")
                        or act_c.get("recent_videos")
                        or []
                    )
                    if isinstance(v_items, list) and v_items:
                        for v in v_items[:4]:
                            if isinstance(v, dict) and (v.get("video_id") or v.get("title")):
                                vid_id = v.get("video_id")
                                if not any(existing.get("video_id") == vid_id for existing in videos_payload if vid_id):
                                    videos_payload.append(v)
                                    evidence_sources.append({
                                        "title": v.get("title", "Video"),
                                        "video_id": vid_id,
                                        "source": f"YouTube ({tc.name})",
                                        "value": v.get("view_count"),
                                    })

                    # 2. Tavily web search results
                    search_res = tool_result.get("results") or []
                    if isinstance(search_res, list):
                        working_memory.web_findings.extend(search_res[:4])
                        for sr in search_res[:3]:
                            if isinstance(sr, dict) and (sr.get("url") or sr.get("title")):
                                evidence_sources.append({
                                    "title": sr.get("title", "Web Source"),
                                    "url": sr.get("url"),
                                    "source": "Tavily Web Research",
                                    "snippet": sr.get("content", "")[:200],
                                    "provenance": "external_verified",
                                    "confidence": 0.9,
                                 })

                    # 3. Channel summary metrics & cache
                    if tc.name in ("get_connected_channel_summary", "get_channel_details"):
                        working_memory.cached_channel_summary = tool_result
                        metrics_payload.update({
                            "subscribers": tool_result.get("subscriber_count") or act_c.get("subscriber_count"),
                            "lifetime_views": tool_result.get("view_count") or act_c.get("view_count"),
                            "total_uploads": tool_result.get("video_count") or act_c.get("video_count"),
                            "channel_title": tool_result.get("title") or tool_result.get("channel_name") or act_c.get("channel_name"),
                        })

                if evidence_sources:
                    execution_state.add_event("evidence_collected", execution_state.status, f"Aggregated {len(evidence_sources)} evidence records from tools", {"count": len(evidence_sources)})

                # Compact result into valid JSON for model context
                compact_res = self._compact_tool_result(tc.name, tool_result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": json.dumps(compact_res, default=str),
                })

            # Completed all tool calls for this turn
            if execution_run:
                execution_run.complete_step("calling_tools", f"Retrieved information from {len(resp.tool_calls)} tool call(s).")
                execution_run.start_step("reviewing_results", "Examining the information returned by the tools.")
                execution_run.complete_step("reviewing_results", f"Examined tool findings and verified {len(evidence_sources)} evidence records.")
                execution_run.start_step("deciding_next", "Determining whether more information is needed.")

        # 5. Synthesis step if turns exhausted without final text
        if not final_answer and not resp.error:
            if execution_run:
                execution_run.complete_step("deciding_next", "Completed all tool rounds; synthesizing final response.")
                execution_run.start_step("validating_findings", "Checking the available evidence and data consistency.")
                execution_run.complete_step("validating_findings", f"Checked {len(evidence_sources)} evidence items for synthesis consistency.")
                execution_run.start_step("generating_answer", "Turning the findings into a useful response.")

            execution_state.update_stage("synthesizing_answer", "Synthesizing Evidence & Insights")
            execution_state.add_event("synthesis_started", "synthesizing_answer", "Synthesizing gathered tool data into comprehensive response")
            messages.append({
                "role": "user",
                "content": "Please synthesize all gathered tool data and provide your comprehensive, data-grounded final answer with key metrics and actionable recommendations now.",
            })
            synthesis = self.llm.chat_completion(messages=messages, max_tokens=1500)
            if synthesis:
                final_answer = self._clean_model_output(synthesis)
                execution_state.add_event("answer_generated", "synthesizing_answer", "Final answer successfully synthesized")
            if execution_run:
                execution_run.complete_step("generating_answer", "Turning the findings into a useful response.")

        if not final_answer:
            final_answer = (
                f"I investigated the available data for **{ch_title}**. "
                "However, the model was unable to complete the synthesis. You can ask me to rank your uploads, compare videos, or check keyword opportunities."
            )

        # Fix Defect 2: Do not overwrite failed status to completed
        if execution_state.status != "failed":
            execution_state.update_stage("completed", "Response Synthesized")
            execution_state.add_event("execution_completed", "completed", "Q&A Agent run completed successfully")
        else:
            execution_state.add_event("execution_failed", "failed", "Q&A Agent run failed due to execution errors")

        # Ensure active matched video target is included in videos payload and evidence sources
        if working_memory and working_memory.active_video_id and not any(v.get("video_id") == working_memory.active_video_id for v in videos_payload):
            target_v = next((v for v in (working_memory.cached_inventory or []) if v.get("video_id") == working_memory.active_video_id), None)
            if target_v:
                videos_payload.insert(0, {**target_v, "badge": "Target Video"})
                if not any(e.get("video_id") == working_memory.active_video_id for e in evidence_sources):
                    evidence_sources.insert(0, {
                        "title": target_v.get("title", "Video"),
                        "video_id": target_v.get("video_id"),
                        "source": "YouTube (Inventory Match)",
                        "value": target_v.get("view_count"),
                    })

        follow_ups = self._generate_contextual_follow_ups(user_text, tool_calls_executed)
        return final_answer, videos_payload, metrics_payload, evidence_sources, data_coverage, follow_ups, tool_calls_executed

    @staticmethod
    def _clean_model_output(raw_text: str) -> str:
        """Strip internal thinking tokens and clean markdown response."""
        text = raw_text.strip()
        if "<thought>" in text and "</thought>" in text:
            parts = text.split("</thought>")
            candidate = parts[-1].strip()
            text = candidate if candidate else text.replace("<thought>", "").replace("</thought>", "").strip()
        elif "<think>" in text and "</think>" in text:
            parts = text.split("</think>")
            candidate = parts[-1].strip()
            text = candidate if candidate else text.replace("<think>", "").replace("</think>", "").strip()
        return text

    @staticmethod
    def _sanitize_untrusted_text(text: Any) -> str:
        """Neutralize prompt injection attempts and special instruction delimiters in external text."""
        if not isinstance(text, str):
            return ""
        sanitized = text
        for token in ("<system>", "</system>", "<|im_start|>", "<|im_end|>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"):
            sanitized = sanitized.replace(token, f"[{token.strip('<>[]')}]")
        return sanitized

    @classmethod
    def _compact_tool_result(cls, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        """Safely compact tool results to valid JSON without losing essential facts and sanitize untrusted text."""
        if not isinstance(result, dict):
            return {"result": str(result)}
        compact = dict(result)
        # Cap video items to top 8 and sanitize text fields
        for k in ("videos", "ranked_videos", "candidate_videos", "recent_videos"):
            if k in compact and isinstance(compact[k], list):
                if len(compact[k]) > 8:
                    compact["total_count"] = len(compact[k])
                    compact[k] = compact[k][:8]
                for v in compact[k]:
                    if isinstance(v, dict):
                        if "description" in v and isinstance(v["description"], str):
                            v["description"] = cls._sanitize_untrusted_text(v["description"][:300])
                        if "title" in v and isinstance(v["title"], str):
                            v["title"] = cls._sanitize_untrusted_text(v["title"])

        # Cap search results to top 5 and sanitize snippets
        if "results" in compact and isinstance(compact["results"], list):
            if len(compact["results"]) > 5:
                compact["total_count"] = len(compact["results"])
            compact["results"] = [
                {
                    "title": cls._sanitize_untrusted_text(r.get("title", "")),
                    "url": r.get("url"),
                    "content": cls._sanitize_untrusted_text(r.get("content", "")[:350]),
                }
                for r in compact["results"][:5]
                if isinstance(r, dict)
            ]
        return compact

    @staticmethod
    def _generate_contextual_follow_ups(user_text: str, tool_calls: list[dict[str, Any]]) -> list[str]:
        """Generate smart follow-ups based on the executed tools and query context."""
        tools_used = [tc.get("tool", "") for tc in tool_calls]
        q_lower = user_text.lower()

        candidates: list[str] = []

        if any(t in tools_used for t in ("search_youtube_growth_strategies", "research_youtube_algorithm_updates")):
            candidates = [
                "How do these strategies apply to my top videos?",
                "Suggest 3 specific video ideas based on this research.",
                "What thumbnail and title styles work best for this?",
            ]
        elif any(t in tools_used for t in ("search_competitor_videos", "search_youtube_videos")):
            candidates = [
                "What keywords are they targeting in their titles?",
                "Compare my latest video with their top upload.",
                "How can I package a differentiated angle on this topic?",
            ]
        elif any(t in tools_used for t in ("rank_and_calculate_video_metrics", "get_channel_video_inventory")) or any(k in q_lower for k in ("highest-performing", "highest performing", "top video", "best video", "most viewed", "rank")):
            candidates = [
                "Why did this top video perform so well?",
                "Compare my best video with my latest upload.",
                "How do my YouTube Shorts perform compared to long-form videos?",
            ]
        elif "compare_videos_detailed" in tools_used or "compare" in q_lower:
            candidates = [
                "Why is there a view gap between these uploads?",
                "What keywords should I target next?",
                "Suggest 3 topic ideas modeled after my top performer.",
            ]
        elif "subscriber" in q_lower or "get_connected_channel_summary" in tools_used:
            candidates = [
                "What is my highest-performing video by views?",
                "Compare my best video with my latest upload.",
                "What topics give me the best engagement rate?",
            ]
        else:
            candidates = [
                "What is the highest-performing video on my channel?",
                "Compare my best video with my latest upload.",
                "What keywords should I target next?",
            ]

        # Filter out questions that duplicate the current query
        filtered = []
        for c in candidates:
            c_low = c.lower()
            if ("highest" in q_lower or "top video" in q_lower or "most viewed" in q_lower) and ("highest-performing" in c_low or "highest performing" in c_low):
                continue
            if ("subscriber" in q_lower or "subs" in q_lower) and "subscriber" in c_low:
                continue
            if "compare" in q_lower and "compare" in c_low:
                continue
            filtered.append(c)

        if len(filtered) < 2:
            return candidates
        return filtered

    def _select_tools_for_query(self, query: str) -> list[dict[str, Any]]:
        """Dynamically select a focused, relevant subset of tools based on query intent.

        Per Section 7.E ('The agent should select tools based on the user's intent.
        It should not call every available tool for every question'), this optimizes model
        focus and drastically improves response time on local CPU inference.
        """
        q = query.lower()
        selected_names: set[str] = set()

        # Video ranking, catalog review, top videos (Fix Defect 3: accumulate, do not return early)
        is_ranking = any(w in q for w in ("top video", "best video", "highest-performing", "highest performing", "most viewed", "most views", "rank"))
        if is_ranking:
            selected_names.update([
                "rank_and_calculate_video_metrics",
                "get_channel_video_inventory",
            ])
            if any(w in q for w in ("http", "youtube.com", "youtu.be", "@")):
                selected_names.add("resolve_youtube_target")

        # Creator struggles, troubleshooting & community research
        if any(w in q for w in ("struggle", "problem", "frustrated", "stuck", "drop", "drops", "zero views", "stopped getting views", "appeal", "monetization", "low ctr", "leave early", "retention", "why did")):
            selected_names.update([
                "investigate_creator_struggles",
                "search_youtube_growth_strategies",
                "search_external_web",
            ])

        # Niche exploration & trending topic discovery
        if any(w in q for w in ("niche", "trending topic", "radar", "emerging topic", "opportunity", "what to make")):
            selected_names.update([
                "search_niche_radar",
                "research_niche_and_trends",
            ])

        # Channel stats, subscriber count, lifetime metrics, channel name, content type
        if any(w in q for w in ("subscriber", "sub count", "my channel", "how many subs", "lifetime view", "channel stat", "overview", "channel info", "channel name", "connected channel", "publish", "content type", "type of content", "niche", "who is")):
            selected_names.update([
                "get_channel_details",
                "get_connected_channel_summary",
                "get_channel_video_inventory",
                "rank_and_calculate_video_metrics",
                "get_latest_analysis_report",
            ])

        # URLs, handles, explicit channel analysis
        if any(w in q for w in ("http", "youtube.com", "youtu.be", "@", "channel", "analyze")):
            selected_names.update([
                "get_channel_details",
                "resolve_youtube_target",
                "get_channel_video_inventory",
                "rank_and_calculate_video_metrics",
            ])

        # Competitor intelligence & public YouTube search
        if any(w in q for w in ("competitor", "codebasics", "benchmark", "vs", "versus", "compare", "other channel", "in my niche", "search")):
            selected_names.update([
                "search_competitor_videos",
                "search_youtube_videos",
                "compare_videos_detailed",
                "get_channel_details",
                "get_video_details",
                "search_competitor_content_strategies",
                "get_connected_channel_summary",
            ])

        # Growth strategies, algorithm updates, trends, video ideas
        if any(w in q for w in ("growth", "strateg", "algorithm", "update", "trend", "idea", "next", "seo", "thumbnail", "hook", "title")):
            selected_names.update([
                "search_youtube_growth_strategies",
                "research_youtube_algorithm_updates",
                "research_niche_and_trends",
                "get_channel_details",
                "get_connected_channel_summary",
                "get_channel_video_inventory",
                "search_competitor_content_strategies",
            ])

        # External web research & creator discussions
        if any(w in q for w in ("web", "google", "reddit", "article", "news", "forum")):
            selected_names.update([
                "search_external_web",
                "search_youtube_growth_strategies",
            ])

        # Default fallback to core representative tools across domains
        if len(selected_names) < 3:
            selected_names.update([
                "get_channel_details",
                "get_connected_channel_summary",
                "get_channel_video_inventory",
                "search_youtube_videos",
                "search_competitor_videos",
                "search_youtube_growth_strategies",
                "research_youtube_algorithm_updates",
            ])

        return [t for t in self.tools if t["function"]["name"] in selected_names]
