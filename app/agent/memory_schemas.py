from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


MemoryCategory = Literal["creator_profile", "episodic", "semantic"]
VerificationStatus = Literal["verified_observation", "creator_stated", "ai_hypothesis"]


class CreatorMemoryBase(BaseModel):
    category: MemoryCategory
    memory_type: str = Field(description="Subtype: niche_definition, audience_profile, past_finding, verified_insight, experiment_outcome")
    content: str = Field(description="Compact, readable summary suitable for LLM working context")
    structured_payload: dict[str, Any] = Field(default_factory=dict, description="Structured metadata and evidence")
    source_job_id: str | None = None
    source_evidence_id: str | None = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    verification_status: VerificationStatus = Field(default="verified_observation")


class CreatorMemoryCreate(CreatorMemoryBase):
    pass


class CreatorMemoryResponse(CreatorMemoryBase):
    id: str
    workspace_id: str
    user_id: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CreatorProfileUpdate(BaseModel):
    channel_niche: str | None = None
    target_audience: str | None = None
    content_format: str | None = None
    language_preference: str | None = None
    stated_goals: list[str] | None = None
