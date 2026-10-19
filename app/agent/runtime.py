from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from typing import Any, Callable

logger = logging.getLogger("growtube.execution_manager")

# Standard 8 workflow step definitions with exact user-facing labels and natural descriptions
WORKFLOW_STEPS = [
    {
        "id": "understanding",
        "label": "Understanding your question",
        "description": "Identifying what you want to find out.",
    },
    {
        "id": "planning",
        "label": "Planning with the LLM",
        "description": "Working out the best approach to your request.",
    },
    {
        "id": "selecting_tools",
        "label": "Selecting the right tools",
        "description": "Choosing the tools needed to gather relevant information.",
    },
    {
        "id": "calling_tools",
        "label": "Calling tools",
        "description": "Retrieving video information from YouTube.",
    },
    {
        "id": "reviewing_results",
        "label": "Reviewing tool results",
        "description": "Examining the information returned by the tools.",
    },
    {
        "id": "deciding_next",
        "label": "Deciding what to do next",
        "description": "Determining whether more information is needed.",
    },
    {
        "id": "validating_findings",
        "label": "Validating the findings",
        "description": "Checking the available evidence and data consistency.",
    },
    {
        "id": "generating_answer",
        "label": "Generating your answer",
        "description": "Turning the findings into a useful response.",
    },
]

# Registered tool display names and concise purpose descriptions
TOOL_METADATA: dict[str, dict[str, str]] = {
    "get_channel_video_inventory": {
        "display_name": "Channel Video Inventory",
        "description": "Retrieves public upload catalog and metadata from YouTube",
    },
    "get_channel_details": {
        "display_name": "Channel Analytics",
        "description": "Inspects channel statistics, subscriber counts, and recent videos",
    },
    "get_connected_channel_summary": {
        "display_name": "Channel Overview",
        "description": "Gathers high-level channel performance and upload metrics",
    },
    "rank_and_calculate_video_metrics": {
        "display_name": "Video Performance Ranking",
        "description": "Calculates view counts, velocity, and engagement across uploads",
    },
    "compare_videos_detailed": {
        "display_name": "Side-by-Side Video Comparison",
        "description": "Compares performance differentials between top and latest uploads",
    },
    "search_youtube_videos": {
        "display_name": "YouTube Video Search",
        "description": "Searches public YouTube videos for topic and format benchmarking",
    },
    "search_competitor_videos": {
        "display_name": "Competitor Search",
        "description": "Finds competing channel uploads in the same content niche",
    },
    "search_external_web": {
        "display_name": "Web Research",
        "description": "Researches creator forums, strategy articles, and public web sources",
    },
    "search_youtube_growth_strategies": {
        "display_name": "Growth Strategy Research",
        "description": "Investigates proven growth patterns and packaging techniques",
    },
    "research_youtube_algorithm_updates": {
        "display_name": "Algorithm Updates Research",
        "description": "Inspects recent YouTube recommendation and algorithmic guidelines",
    },
    "research_niche_and_trends": {
        "display_name": "Niche Trends Research",
        "description": "Explores trending topics, formats, and viewer demand in the niche",
    },
    "investigate_creator_struggles": {
        "display_name": "Creator Struggles Investigation",
        "description": "Identifies universal creator retention and audience growth bottlenecks",
    },
    "search_niche_radar": {
        "display_name": "Niche Radar Search",
        "description": "Discovers high-demand topical whitespace and underserved niches",
    },
    "resolve_youtube_target": {
        "display_name": "Resolve Target URL",
        "description": "Resolves YouTube handles, URLs, and channel identifiers",
    },
    "retrieve_creator_memories": {
        "display_name": "Creator Memory Lookup",
        "description": "Retrieves workspace creator preferences, goals, and feedback",
    },
}


def get_tool_meta(tool_name: str) -> dict[str, str]:
    """Retrieve human-friendly metadata for a tool."""
    return TOOL_METADATA.get(
        tool_name,
        {
            "display_name": tool_name.replace("_", " ").title(),
            "description": f"Executes {tool_name.replace('_', ' ')}",
        },
    )


def generate_safe_tool_summary(tool_name: str, result: dict[str, Any] | Any, is_success: bool) -> str:
    """Generate safe, readable summary of a tool execution without exposing private tokens or CoT."""
    if not is_success:
        err = result.get("error") if isinstance(result, dict) else "Execution failed"
        return f"Failed: {err}"

    if not isinstance(result, dict):
        return "Completed successfully"

    if tool_name == "get_channel_video_inventory":
        vids = result.get("videos") or []
        tot = result.get("total_channel_videos") or len(vids)
        return f"Retrieved {len(vids)} uploads (catalog total: {tot})"

    if tool_name == "rank_and_calculate_video_metrics":
        insp = result.get("inspected_count", 0)
        winner = result.get("winning_video") or {}
        winner_title = winner.get("title", "")
        metric = result.get("ranking_metric", "views")
        if winner_title:
            return f"Ranked {insp} uploads by {metric} (Top: '{winner_title[:28]}...')"
        return f"Ranked {insp} uploads by {metric}"

    if tool_name in ("get_channel_details", "get_connected_channel_summary"):
        title = result.get("title") or result.get("channel_name") or "Channel"
        subs = result.get("subscriber_count")
        views = result.get("view_count") or 0
        subs_str = f"{subs:,} subs" if subs is not None else "subs hidden"
        return f"Stats for '{title}': {subs_str}, {views:,} views"

    if tool_name == "compare_videos_detailed":
        va = result.get("video_a", {}).get("title", "Video A")
        vb = result.get("video_b", {}).get("title", "Video B")
        diff = result.get("metric_comparison", {}).get("view_difference", 0)
        return f"Compared '{va[:20]}...' vs '{vb[:20]}...' (Gap: {diff:,} views)"

    if tool_name in ("search_youtube_videos", "search_competitor_videos"):
        candidates = result.get("candidate_videos") or result.get("videos") or []
        return f"Found {len(candidates)} relevant videos"

    if tool_name == "search_external_web":
        results = result.get("results") or []
        return f"Gathered {len(results)} verified web sources"

    if tool_name in ("search_youtube_growth_strategies", "research_youtube_algorithm_updates", "research_niche_and_trends"):
        findings = result.get("findings") or result.get("results") or []
        return f"Extracted {len(findings)} strategy insights"

    return "Completed successfully"


@dataclass
class WorkflowToolCallState:
    call_id: str
    tool_name: str
    display_name: str
    description: str
    status: str  # running, completed, failed
    summary: str = ""
    duration_ms: float = 0.0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "display_name": self.display_name,
            "description": self.description,
            "status": self.status,
            "summary": self.summary,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class WorkflowStepState:
    id: str
    label: str
    description: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    summary: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    tool_calls: list[WorkflowToolCallState] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "status": self.status,
            "summary": self.summary,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
        }


class ExecutionRun:
    """Represents a live workflow execution for a single user question."""

    def __init__(self, conversation_id: str, run_id: str, user_goal: str):
        self.conversation_id = conversation_id
        self.run_id = run_id
        self.user_goal = user_goal
        self.status = "running"  # running, completed, failed
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.current_step_id: str | None = None
        self.error: str | None = None
        self.result: dict[str, Any] | None = None

        # Initialize the 8 canonical steps
        self.steps: dict[str, WorkflowStepState] = {}
        for s in WORKFLOW_STEPS:
            self.steps[s["id"]] = WorkflowStepState(
                id=s["id"],
                label=s["label"],
                description=s["description"],
                status="pending",
            )

        # Event listeners for SSE streaming (list of asyncio.Queue)
        self.queues: list[asyncio.Queue] = []
        self.event_log: list[dict[str, Any]] = []

    def subscribe(self, loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Queue:
        """Subscribe an asyncio queue to receive live workflow events."""
        q = asyncio.Queue()
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        self.queues.append((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Unsubscribe an asyncio queue."""
        self.queues = [pair for pair in self.queues if pair[0] != q]

    def to_dict(self) -> dict[str, Any]:
        calling_step = self.steps.get("calling_tools")
        all_tool_calls = [tc.to_dict() for tc in calling_step.tool_calls] if calling_step else []
        return {
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "user_goal": self.user_goal,
            "status": self.status,
            "current_step_id": self.current_step_id,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": {step.id: step.to_dict() for step in self.steps.values()},
            "tool_calls": all_tool_calls,
            "events_count": len(self.event_log),
        }

    def dispatch_event(self, event_data: dict[str, Any]):
        """Append event to log and push to all active SSE subscriber queues."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        full_event = {
            "conversation_id": self.conversation_id,
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timeline": self.to_dict(),
            **event_data,
        }
        self.event_log.append(full_event)

        for q, loop in list(self.queues):
            try:
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(q.put_nowait, full_event)
                else:
                    q.put_nowait(full_event)
            except Exception as e:
                logger.debug(f"Failed to push to SSE queue: {e}")

    def start_step(self, step_id: str, summary: str = ""):
        """Mark step as running."""
        if step_id not in self.steps:
            return
        step = self.steps[step_id]
        step.status = "running"
        step.started_at = datetime.now(timezone.utc).isoformat()
        if summary:
            step.summary = summary
        self.current_step_id = step_id

        self.dispatch_event({
            "type": "step_started",
            "step_id": step.id,
            "step_label": step.label,
            "step_description": step.description,
            "status": "running",
            "summary": step.summary,
        })

    def complete_step(self, step_id: str, summary: str = ""):
        """Mark step as completed."""
        if step_id not in self.steps:
            return
        step = self.steps[step_id]
        step.status = "completed"
        step.completed_at = datetime.now(timezone.utc).isoformat()
        if summary:
            step.summary = summary

        self.dispatch_event({
            "type": "step_completed",
            "step_id": step.id,
            "step_label": step.label,
            "status": "completed",
            "summary": step.summary,
        })

    def skip_step(self, step_id: str, summary: str = "Skipped (not needed for this query)"):
        """Mark step as skipped (never mark as completed)."""
        if step_id not in self.steps:
            return
        step = self.steps[step_id]
        step.status = "skipped"
        step.summary = summary
        step.completed_at = datetime.now(timezone.utc).isoformat()

        self.dispatch_event({
            "type": "step_skipped",
            "step_id": step.id,
            "step_label": step.label,
            "status": "skipped",
            "summary": step.summary,
        })

    def fail_step(self, step_id: str, error_msg: str):
        """Mark step as failed."""
        if step_id not in self.steps:
            return
        step = self.steps[step_id]
        step.status = "failed"
        step.summary = error_msg
        step.completed_at = datetime.now(timezone.utc).isoformat()

        self.dispatch_event({
            "type": "step_failed",
            "step_id": step.id,
            "step_label": step.label,
            "status": "failed",
            "error": error_msg,
        })

    def start_tool_call(
        self,
        call_id: str | None = None,
        tool_name: str = "",
        display_name: str | None = None,
        purpose: str | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> WorkflowToolCallState:
        """Record start of individual nested tool call."""
        actual_id = tool_call_id or call_id or f"tool_{len(self.steps['calling_tools'].tool_calls)}"
        meta = get_tool_meta(tool_name)
        tool_state = WorkflowToolCallState(
            call_id=actual_id,
            tool_name=tool_name,
            display_name=display_name or meta.get("display_name", tool_name),
            description=purpose or meta.get("description", f"Executes {tool_name}"),
            status="running",
        )
        # Ensure Step 4 is active
        calling_step = self.steps["calling_tools"]
        if calling_step.status != "running":
            self.start_step("calling_tools", "Retrieving video information from YouTube.")

        calling_step.tool_calls.append(tool_state)

        self.dispatch_event({
            "type": "tool_started",
            "step_id": "calling_tools",
            "tool_call_id": tool_state.call_id,
            "tool_display_name": tool_state.display_name,
            "tool_description": tool_state.description,
            "status": "running",
            "tool_call": tool_state.to_dict(),
        })
        return tool_state

    def complete_tool_call(
        self,
        call_id: str | None = None,
        tool_name: str = "",
        result: Any = None,
        is_success: bool = True,
        duration_ms: float = 0.0,
        tool_call_id: str | None = None,
        success: bool | None = None,
        result_summary: str | None = None,
        error: str | None = None,
        **kwargs: Any,
    ):
        """Record completion of individual nested tool call."""
        actual_id = tool_call_id or call_id
        actual_success = is_success if success is None else success
        calling_step = self.steps["calling_tools"]
        target_call = next((tc for tc in calling_step.tool_calls if tc.call_id == actual_id), None)
        meta = get_tool_meta(tool_name)
        safe_summary = result_summary or generate_safe_tool_summary(tool_name, result, actual_success)

        if not target_call:
            target_call = WorkflowToolCallState(
                call_id=actual_id or f"tool_{len(calling_step.tool_calls)}",
                tool_name=tool_name,
                display_name=meta.get("display_name", tool_name),
                description=meta.get("description", f"Executes {tool_name}"),
                status="completed" if actual_success else "failed",
                summary=safe_summary,
                duration_ms=duration_ms,
            )
            calling_step.tool_calls.append(target_call)
        else:
            target_call.status = "completed" if actual_success else "failed"
            target_call.summary = safe_summary
            target_call.duration_ms = duration_ms
            target_call.completed_at = datetime.now(timezone.utc).isoformat()

        self.dispatch_event({
            "type": "tool_completed" if actual_success else "tool_failed",
            "step_id": "calling_tools",
            "tool_call_id": target_call.call_id,
            "tool_display_name": target_call.display_name,
            "status": target_call.status,
            "summary": target_call.summary,
            "tool_call": target_call.to_dict(),
        })

    def finish_run(self, result: dict[str, Any]):
        """Complete workflow run."""
        self.status = "completed"
        self.result = result
        self.updated_at = datetime.now(timezone.utc).isoformat()

        self.dispatch_event({
            "type": "workflow_completed",
            "status": "completed",
            "summary": "Agent workflow finished successfully",
            "workflow_state": self.to_dict(),
            "result": result,
        })

    def fail_run(self, error_msg: str):
        """Fail workflow run."""
        self.status = "failed"
        self.error = error_msg
        self.updated_at = datetime.now(timezone.utc).isoformat()

        self.dispatch_event({
            "type": "workflow_failed",
            "status": "failed",
            "error": error_msg,
            "workflow_state": self.to_dict(),
        })


class ExecutionManager:
    """Singleton manager tracking active conversation agent runs and event subscribers."""

    def __init__(self):
        self._runs: dict[str, ExecutionRun] = {}  # keyed by run_id
        self._conv_to_run: dict[str, str] = {}    # conversation_id -> latest run_id

    def create_run(self, conversation_id: str, run_id: str, user_goal: str = "", **kwargs) -> ExecutionRun:
        goal = user_goal or kwargs.get("user_query") or kwargs.get("user_text") or ""
        run = ExecutionRun(conversation_id, run_id, goal)
        self._runs[run_id] = run
        self._conv_to_run[conversation_id] = run_id
        return run

    def get_run(self, run_id: str) -> ExecutionRun | None:
        return self._runs.get(run_id)

    def get_latest_run_for_conversation(self, conversation_id: str) -> ExecutionRun | None:
        run_id = self._conv_to_run.get(conversation_id)
        if run_id:
            return self._runs.get(run_id)
        return None

    def get_live_state(self, conversation_id: str) -> dict[str, Any] | None:
        run = self.get_latest_run_for_conversation(conversation_id)
        if run:
            return run.to_dict()
        return None


# Global singleton instance
execution_manager = ExecutionManager()
