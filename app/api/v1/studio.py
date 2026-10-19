from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.database.session import get_db
from app.services.content_studio import ContentStudioService

router = APIRouter(prefix="/studio", tags=["Content Studio & Script Generator"])


class GenerateScriptRequest(BaseModel):
    title: str = Field(min_length=3)
    video_type: str = Field(default="YouTube Video")
    opportunity_id: str | None = None


class UpdateScriptRequest(BaseModel):
    script_draft: str


@router.get("/scripts")
def list_scripts(
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    service = ContentStudioService(db)
    return service.list_scripts(tenant.workspace_id)


@router.post("/scripts/generate", status_code=status.HTTP_201_CREATED)
def generate_script(
    payload: GenerateScriptRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Generate structured script and checklist matching UI reference Screen 6."""
    service = ContentStudioService(db)
    return service.generate_script_from_topic(
        workspace_id=tenant.workspace_id,
        title=payload.title,
        video_type=payload.video_type,
        opportunity_id=payload.opportunity_id,
    )


@router.put("/scripts/{script_id}")
def update_script(
    script_id: str,
    payload: UpdateScriptRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    service = ContentStudioService(db)
    updated = service.update_script(script_id, tenant.workspace_id, payload.script_draft)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Script not found")
    return updated
