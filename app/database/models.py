import uuid
from datetime import datetime
from typing import Any
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    google_id: Mapped[str | None] = mapped_column(String(100), unique=True, index=True, nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    auth_provider: Mapped[str] = mapped_column(String(50), default="local", nullable=False)  # local, google
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    memberships: Mapped[list["Membership"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserSession(Base, TimestampMixin):
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    session_token: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    csrf_token: Mapped[str] = mapped_column(String(128), nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="sessions")


class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    workspaces: Mapped[list["Workspace"]] = relationship(back_populates="organization", cascade="all, delete-orphan")


class Workspace(Base, TimestampMixin):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), index=True, nullable=False)

    organization: Mapped["Organization"] = relationship(back_populates="workspaces")
    memberships: Mapped[list["Membership"]] = relationship(back_populates="workspace", cascade="all, delete-orphan")


class Membership(Base, TimestampMixin):
    __tablename__ = "memberships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(36), ForeignKey("workspaces.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="member", nullable=False)  # owner, admin, editor, viewer

    user: Mapped["User"] = relationship(back_populates="memberships")
    workspace: Mapped["Workspace"] = relationship(back_populates="memberships")


class ChannelConnection(Base, TimestampMixin):
    __tablename__ = "channel_connections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), ForeignKey("workspaces.id"), index=True, nullable=False)
    channel_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    custom_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    subscriber_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    view_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    thumbnail_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class AnalysisJob(Base, TimestampMixin):
    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), ForeignKey("workspaces.id"), index=True, nullable=False)
    input_type: Mapped[str] = mapped_column(String(50), nullable=False)  # channel_url, video_url, file_upload
    input_target: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="queued", index=True, nullable=False)  # queued, running, succeeded, failed, partial
    current_stage: Mapped[str] = mapped_column(String(100), default="initialized", nullable=False)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    checkpoint_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    events: Mapped[list["WorkflowEvent"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class WorkflowEvent(Base, TimestampMixin):
    __tablename__ = "workflow_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("analysis_jobs.id"), index=True, nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    stage: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[str] = mapped_column(String(20), default="info", nullable=False)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    job: Mapped["AnalysisJob"] = relationship(back_populates="events")


class EvidenceRecord(Base, TimestampMixin):
    __tablename__ = "evidence_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    job_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)


class FindingRecord(Base, TimestampMixin):
    __tablename__ = "finding_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    job_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    epistemic_tier: Mapped[str] = mapped_column(String(50), default="measured_observation", nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)


class AnalysisReport(Base, TimestampMixin):
    __tablename__ = "analysis_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    job_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    channel_title: Mapped[str] = mapped_column(String(255), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(100), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    key_findings: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    recommendations: Mapped[list[str]] = mapped_column(JSON, nullable=False)


class ContentOpportunity(Base, TimestampMixin):
    __tablename__ = "content_opportunities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    topic_category: Mapped[str] = mapped_column(String(100), nullable=False)
    demand_score: Mapped[float] = mapped_column(Float, default=8.0, nullable=False)
    saturation_score: Mapped[float] = mapped_column(Float, default=4.0, nullable=False)
    authority_score: Mapped[float] = mapped_column(Float, default=8.5, nullable=False)
    opportunity_score: Mapped[int] = mapped_column(Integer, default=85, nullable=False)
    differentiation_angle: Mapped[str] = mapped_column(Text, nullable=False)
    target_audience: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    is_bookmarked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class ContentScript(Base, TimestampMixin):
    __tablename__ = "content_scripts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    opportunity_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    video_type: Mapped[str] = mapped_column(String(50), default="YouTube Video", nullable=False)
    script_draft: Mapped[str] = mapped_column(Text, nullable=False)
    checklist: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="draft", nullable=False)


class ExperimentRecord(Base, TimestampMixin):
    __tablename__ = "experiment_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    channel_id: Mapped[str] = mapped_column(String(100), nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    target_metric: Mapped[str] = mapped_column(String(100), nullable=False)
    baseline_value: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    observed_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="in_progress", nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class CreatorMemory(Base, TimestampMixin):
    __tablename__ = "creator_memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), ForeignKey("workspaces.id"), index=True, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=True)
    category: Mapped[str] = mapped_column(String(50), index=True, nullable=False)  # creator_profile, episodic, semantic
    memory_type: Mapped[str] = mapped_column(String(100), index=True, nullable=False)  # e.g., niche_definition, audience_profile, past_finding, verified_insight
    content: Mapped[str] = mapped_column(Text, nullable=False)
    structured_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    source_job_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    source_evidence_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    verification_status: Mapped[str] = mapped_column(String(50), default="verified_observation", nullable=False)  # verified_observation, creator_stated, ai_hypothesis


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), ForeignKey("workspaces.id"), index=True, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=True)
    title: Mapped[str] = mapped_column(String(255), default="New Discussion", nullable=False)
    context_channel_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    context_video_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    channel_context: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="ConversationMessage.created_at.asc()"
    )


class ConversationMessage(Base, TimestampMixin):
    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_sources: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    tool_calls_summary: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class VideoObservation(Base, TimestampMixin):
    __tablename__ = "video_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    video_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    channel_id: Mapped[str | None] = mapped_column(String(100), index=True, nullable=True)
    channel_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    niche_keyword: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    view_count: Mapped[int] = mapped_column(Integer, nullable=False)
    like_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True, nullable=False)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class NicheSearchCache(Base, TimestampMixin):
    __tablename__ = "niche_search_caches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    niche_keyword: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    time_period: Mapped[str] = mapped_column(String(50), default="all", nullable=False)
    category: Mapped[str] = mapped_column(String(50), default="most_viewed", nullable=False)
    cached_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    last_refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class CreatorStruggleCache(Base, TimestampMixin):
    __tablename__ = "creator_struggle_caches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    cache_key: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    category: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    query: Mapped[str] = mapped_column(String(255), nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cached_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    last_refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


