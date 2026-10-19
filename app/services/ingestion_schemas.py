import re
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class InputType(str, Enum):
    CHANNEL_URL = "channel_url"
    VIDEO_URL = "video_url"
    FILE_UPLOAD = "file_upload"


class AnalysisSubmission(BaseModel):
    input_target: str = Field(min_length=3, description="YouTube channel URL, video URL, or uploaded resource handle")
    input_type: InputType | None = None

    @field_validator("input_target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Target URL or identifier cannot be blank")
        return v


class IngestionResult(BaseModel):
    source_type: str
    source_name: str
    channel_id: str | None = None
    channel_title: str | None = None
    thumbnail_url: str | None = None
    custom_url: str | None = None
    channel_url: str | None = None
    description: str | None = None
    video_id: str | None = None
    video_title: str | None = None
    video_count: int = 0
    metrics_data: dict = {}
    analyzed_video: dict | None = None
    recent_videos: list[dict] = []
    transcripts: list[dict] = []
    coverage_status: str = "complete"
    provenance_signature: str = ""
