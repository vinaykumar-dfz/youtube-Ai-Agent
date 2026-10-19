from datetime import datetime
from typing import Any
from pydantic import BaseModel


class EventResponse(BaseModel):
    id: str
    stage: str
    message: str
    level: str
    created_at: datetime
    metadata_json: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class JobResponse(BaseModel):
    id: str
    workspace_id: str
    input_type: str
    input_target: str
    status: str
    current_stage: str
    progress_percent: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    events: list[EventResponse] = []
    result_summary: dict[str, Any] | None = None

    model_config = {"from_attributes": True}
