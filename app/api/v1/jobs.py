import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.core.config import settings
from app.database.session import get_db
from app.services.job_schemas import JobResponse
from app.services.job_management import JobManagementService
from app.workers.worker import process_next_job

router = APIRouter(prefix="/jobs", tags=["Jobs & Workflow"])


class SubmitJobRequest(BaseModel):
    input_target: str = Field(min_length=3, description="YouTube Channel URL, video URL, or channel handle")
    input_type: str = Field(default="channel_url")


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def submit_analysis_job(
    payload: SubmitJobRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Submit a channel URL or video URL for real agent analysis."""
    service = JobManagementService(db)
    job = service.create_job(
        workspace_id=tenant.workspace_id,
        input_type=payload.input_type,
        input_target=payload.input_target,
    )
    # Proactively invoke worker processing so analysis runs smoothly
    process_next_job()
    return job


@router.post("/upload", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def upload_analysis_file(
    file: UploadFile = File(...),
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Upload a CSV, JSON analytics export, or transcript file for analysis."""
    suffix = Path(file.filename or "").suffix.lower()
    allowed_suffixes = [".csv", ".json", ".txt", ".srt", ".vtt"]
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed_suffixes)}",
        )

    file_id = f"{uuid4().hex}_{file.filename}"
    save_path = settings.upload_path / file_id

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    service = JobManagementService(db)
    job = service.create_job(
        workspace_id=tenant.workspace_id,
        input_type="file_upload",
        input_target=f"upload://{save_path}",
    )
    process_next_job()
    return job


@router.get("", response_model=list[JobResponse])
def list_jobs(
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    service = JobManagementService(db)
    return service.list_jobs(tenant.workspace_id)


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(
    job_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    service = JobManagementService(db)
    # Check if job is still queued and advance it
    process_next_job()
    job = service.get_job(job_id, tenant.workspace_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found")
    return job
