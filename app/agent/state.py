from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class WorkflowStage(str, Enum):
    RECEIVED = "RECEIVED"
    RESOLVING_INPUT = "RESOLVING_INPUT"
    PLANNING = "PLANNING"
    COLLECTING_EVIDENCE = "COLLECTING_EVIDENCE"
    ANALYZING = "ANALYZING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    NEEDS_MORE_DATA = "NEEDS_MORE_DATA"
    PARTIAL = "PARTIAL"
    RETRY_PENDING = "RETRY_PENDING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Map internal workflow states to user-facing frontend stages
FRONTEND_STAGE_MAP = {
    WorkflowStage.RECEIVED: "Collecting Data",
    WorkflowStage.RESOLVING_INPUT: "Collecting Data",
    WorkflowStage.PLANNING: "Analyzing Content",
    WorkflowStage.COLLECTING_EVIDENCE: "Analyzing Content",
    WorkflowStage.ANALYZING: "Researching Opportunities",
    WorkflowStage.VERIFYING: "Generating Insights",
    WorkflowStage.COMPLETED: "Finalizing Report",
    WorkflowStage.NEEDS_MORE_DATA: "Analyzing Content",
    WorkflowStage.PARTIAL: "Generating Insights",
    WorkflowStage.RETRY_PENDING: "Analyzing Content",
    WorkflowStage.FAILED: "Analyzing Content",
    WorkflowStage.CANCELLED: "Collecting Data",
}

# Permitted state transitions in the state machine graph
VALID_TRANSITIONS: dict[WorkflowStage, set[WorkflowStage]] = {
    WorkflowStage.RECEIVED: {
        WorkflowStage.RESOLVING_INPUT,
        WorkflowStage.FAILED,
        WorkflowStage.CANCELLED,
    },
    WorkflowStage.RESOLVING_INPUT: {
        WorkflowStage.PLANNING,
        WorkflowStage.NEEDS_MORE_DATA,
        WorkflowStage.FAILED,
        WorkflowStage.CANCELLED,
    },
    WorkflowStage.PLANNING: {
        WorkflowStage.COLLECTING_EVIDENCE,
        WorkflowStage.ANALYZING,
        WorkflowStage.FAILED,
        WorkflowStage.CANCELLED,
    },
    WorkflowStage.COLLECTING_EVIDENCE: {
        WorkflowStage.ANALYZING,
        WorkflowStage.COLLECTING_EVIDENCE,  # Subsequent research step
        WorkflowStage.NEEDS_MORE_DATA,
        WorkflowStage.FAILED,
        WorkflowStage.CANCELLED,
    },
    WorkflowStage.ANALYZING: {
        WorkflowStage.VERIFYING,
        WorkflowStage.COLLECTING_EVIDENCE,  # Agent loops back for more evidence
        WorkflowStage.PARTIAL,
        WorkflowStage.FAILED,
        WorkflowStage.CANCELLED,
    },
    WorkflowStage.VERIFYING: {
        WorkflowStage.COMPLETED,
        WorkflowStage.ANALYZING,
        WorkflowStage.PARTIAL,
        WorkflowStage.FAILED,
        WorkflowStage.CANCELLED,
    },
    WorkflowStage.NEEDS_MORE_DATA: {
        WorkflowStage.COLLECTING_EVIDENCE,
        WorkflowStage.PARTIAL,
        WorkflowStage.FAILED,
    },
    WorkflowStage.RETRY_PENDING: {
        WorkflowStage.PLANNING,
        WorkflowStage.COLLECTING_EVIDENCE,
        WorkflowStage.FAILED,
    },
    WorkflowStage.PARTIAL: {
        WorkflowStage.COMPLETED,
        WorkflowStage.FAILED,
    },
    WorkflowStage.COMPLETED: set(),
    WorkflowStage.FAILED: {WorkflowStage.RETRY_PENDING},
    WorkflowStage.CANCELLED: set(),
}


def can_transition(current: WorkflowStage, target: WorkflowStage) -> bool:
    """Validate whether the transition is permitted by the state machine graph."""
    if current == target:
        return True
    return target in VALID_TRANSITIONS.get(current, set())


@dataclass
class ShortTermMemory:
    """Scoped working memory for the active analysis job."""
    scratchpad: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    dedup_cache_keys: list[str] = field(default_factory=list)

    def add_observation(self, note: str) -> None:
        if note not in self.observations:
            self.observations.append(note)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ShortTermMemory":
        if not data:
            return cls()
        return cls(
            scratchpad=data.get("scratchpad", []),
            observations=data.get("observations", []),
            unresolved_questions=data.get("unresolved_questions", []),
            dedup_cache_keys=data.get("dedup_cache_keys", []),
        )


@dataclass
class AnalysisWorkflowState:
    """Full typed state schema for an analysis job."""
    job_id: str
    workspace_id: str
    input_type: str
    input_target: str
    user_id: str | None = None
    stage: WorkflowStage = WorkflowStage.RECEIVED
    objective: str = "Analyze target video and extract evidence-grounded competitor insights"
    research_plan: list[dict[str, Any]] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)
    tool_call_history: list[dict[str, Any]] = field(default_factory=list)
    resolved_target: dict[str, Any] | None = None
    channel_info: dict[str, Any] | None = None
    competitor_candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_competitors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    memory_references: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2
    tool_call_count: int = 0
    max_tool_calls: int = 8
    errors: list[str] = field(default_factory=list)
    final_report: dict[str, Any] | None = None
    short_term_memory: ShortTermMemory = field(default_factory=ShortTermMemory)

    def transition_to(self, new_stage: WorkflowStage) -> bool:
        """Attempt valid transition in state graph."""
        if can_transition(self.stage, new_stage):
            self.stage = new_stage
            return True
        raise ValueError(f"Invalid state transition from {self.stage} to {new_stage}")

    def get_frontend_stage(self) -> str:
        return FRONTEND_STAGE_MAP.get(self.stage, "Analyzing Content")

    def get_progress_percent(self) -> int:
        progress_weights = {
            WorkflowStage.RECEIVED: 5,
            WorkflowStage.RESOLVING_INPUT: 15,
            WorkflowStage.PLANNING: 30,
            WorkflowStage.COLLECTING_EVIDENCE: 55,
            WorkflowStage.ANALYZING: 75,
            WorkflowStage.VERIFYING: 90,
            WorkflowStage.COMPLETED: 100,
            WorkflowStage.PARTIAL: 85,
            WorkflowStage.FAILED: 100,
            WorkflowStage.CANCELLED: 100,
        }
        return progress_weights.get(self.stage, 50)

    def record_tool_call(self, tool_name: str, args: dict[str, Any], result: dict[str, Any], duration_ms: float) -> None:
        self.tool_call_count += 1
        record = {
            "step": self.tool_call_count,
            "tool": tool_name,
            "stage": self.stage.value,
            "duration_ms": round(duration_ms, 2),
            "args_summary": {k: (str(v)[:100] if not isinstance(v, (int, float, bool)) else v) for k, v in args.items()},
            "success": result.get("success", True),
        }
        self.tool_call_history.append(record)

    def compact_summary_for_llm(self) -> str:
        """Assemble compact context of current workflow state for LLM reasoning without large blobs."""
        target_name = self.resolved_target.get("title") if self.resolved_target else self.input_target
        ch_name = self.channel_info.get("title") if self.channel_info else "Target Channel"
        comp_count = len(self.selected_competitors)
        gaps_count = len(self.gaps)

        lines = [
            f"Current Stage: {self.stage.value} (Tool Calls: {self.tool_call_count}/{self.max_tool_calls})",
            f"Target: '{target_name}' on '{ch_name}'",
            f"Selected Competitors: {comp_count} videos analyzed",
            f"Identified Gaps: {gaps_count} gaps recorded",
            f"Completed Steps: {', '.join(self.completed_steps) if self.completed_steps else 'None'}",
        ]
        if self.short_term_memory.observations:
            lines.append("Key Observations: " + " | ".join(self.short_term_memory.observations[-3:]))
        if self.missing_data:
            lines.append("Missing Data Indicators: " + ", ".join(self.missing_data))
        if self.errors:
            lines.append("Recent Warnings: " + " | ".join(self.errors[-2:]))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        data["short_term_memory"] = self.short_term_memory.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisWorkflowState":
        stage_val = data.get("stage", WorkflowStage.RECEIVED.value)
        try:
            stage = WorkflowStage(stage_val)
        except ValueError:
            stage = WorkflowStage.RECEIVED

        stm_data = data.get("short_term_memory")
        stm = ShortTermMemory.from_dict(stm_data)

        # Reconstruct clean state
        state = cls(
            job_id=data["job_id"],
            workspace_id=data["workspace_id"],
            input_type=data.get("input_type", "channel_url"),
            input_target=data.get("input_target", ""),
            user_id=data.get("user_id"),
            stage=stage,
            objective=data.get("objective", "Analyze target video and extract competitor insights"),
            research_plan=data.get("research_plan", []),
            completed_steps=data.get("completed_steps", []),
            pending_actions=data.get("pending_actions", []),
            tool_call_history=data.get("tool_call_history", []),
            resolved_target=data.get("resolved_target"),
            channel_info=data.get("channel_info"),
            competitor_candidates=data.get("competitor_candidates", []),
            selected_competitors=data.get("selected_competitors", []),
            metrics=data.get("metrics", {}),
            evidence_records=data.get("evidence_records", []),
            memory_references=data.get("memory_references", []),
            gaps=data.get("gaps", []),
            recommendations=data.get("recommendations", []),
            missing_data=data.get("missing_data", []),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 2),
            tool_call_count=data.get("tool_call_count", 0),
            max_tool_calls=data.get("max_tool_calls", 8),
            errors=data.get("errors", []),
            final_report=data.get("final_report"),
            short_term_memory=stm,
        )
        return state
