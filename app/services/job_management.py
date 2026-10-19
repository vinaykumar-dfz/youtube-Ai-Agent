from typing import Any
from sqlalchemy.orm import Session

from app.database.models import AnalysisJob, WorkflowEvent


class JobManagementService:
    def __init__(self, db: Session):
        self.db = db

    def create_job(self, workspace_id: str, input_type: str, input_target: str) -> AnalysisJob:
        job = AnalysisJob(
            workspace_id=workspace_id,
            input_type=input_type,
            input_target=input_target,
            status="queued",
            current_stage="Collecting Data",
            progress_percent=0,
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        self.add_event(
            job.id,
            workspace_id,
            stage="Collecting Data",
            message=f"Job queued for target: {input_target}",
        )
        return job

    def add_event(
        self,
        job_id: str,
        workspace_id: str,
        stage: str,
        message: str,
        level: str = "info",
        metadata_json: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        event = WorkflowEvent(
            job_id=job_id,
            workspace_id=workspace_id,
            stage=stage,
            message=message,
            level=level,
            metadata_json=metadata_json,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def update_job_progress(
        self,
        job_id: str,
        stage: str,
        progress_percent: int,
        status: str = "running",
        result_summary: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> AnalysisJob | None:
        job = self.db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.current_stage = stage
            job.progress_percent = progress_percent
            job.status = status
            if result_summary:
                job.result_summary = result_summary
            if error_message:
                job.error_message = error_message
            self.db.commit()
            self.db.refresh(job)
        return job

    def get_job(self, job_id: str, workspace_id: str) -> AnalysisJob | None:
        return (
            self.db.query(AnalysisJob)
            .filter(AnalysisJob.id == job_id, AnalysisJob.workspace_id == workspace_id)
            .first()
        )

    def list_jobs(self, workspace_id: str, limit: int = 10) -> list[AnalysisJob]:
        return (
            self.db.query(AnalysisJob)
            .filter(AnalysisJob.workspace_id == workspace_id)
            .order_by(AnalysisJob.created_at.desc())
            .limit(limit)
            .all()
        )
