from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.database.session import get_db
from app.agent.memory_schemas import (
    CreatorMemoryCreate,
    CreatorMemoryResponse,
    CreatorProfileUpdate,
)
from app.agent.memory import CreatorMemoryService

router = APIRouter(prefix="/memories", tags=["Creator Memory"])


@router.get("", response_model=list[CreatorMemoryResponse])
def list_creator_memories(
    category: str | None = Query(None, description="Filter by category: creator_profile, episodic, semantic"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Retrieve long-term creator memories for the authenticated workspace."""
    service = CreatorMemoryService(db)
    return service.list_workspace_memories(tenant.workspace_id, category=category)


@router.post("", response_model=CreatorMemoryResponse, status_code=status.HTTP_201_CREATED)
def create_creator_memory(
    payload: CreatorMemoryCreate,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Manually add or update a creator memory or stated preference."""
    service = CreatorMemoryService(db)
    mem = service.save_memory(
        workspace_id=tenant.workspace_id,
        user_id=tenant.user.id,
        category=payload.category,
        memory_type=payload.memory_type,
        content=payload.content,
        structured_payload=payload.structured_payload,
        confidence_score=payload.confidence_score,
        verification_status=payload.verification_status,
    )
    return mem


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_creator_memory(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Delete a memory record enforcing workspace tenant boundaries."""
    service = CreatorMemoryService(db)
    deleted = service.delete_memory(tenant.workspace_id, memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory record not found or unauthorized.",
        )
    return None


@router.get("/profile")
def get_creator_profile(
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Get unified creator profile synthesized from creator_profile memories."""
    service = CreatorMemoryService(db)
    return service.get_creator_profile(tenant.workspace_id)


@router.put("/profile")
def update_creator_profile(
    payload: CreatorProfileUpdate,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Update stated creator profile preferences."""
    service = CreatorMemoryService(db)
    if payload.channel_niche:
        service.save_memory(
            workspace_id=tenant.workspace_id,
            user_id=tenant.user.id,
            category="creator_profile",
            memory_type="channel_niche",
            content=payload.channel_niche,
            verification_status="creator_stated",
        )
    if payload.target_audience:
        service.save_memory(
            workspace_id=tenant.workspace_id,
            user_id=tenant.user.id,
            category="creator_profile",
            memory_type="target_audience",
            content=payload.target_audience,
            verification_status="creator_stated",
        )
    if payload.content_format:
        service.save_memory(
            workspace_id=tenant.workspace_id,
            user_id=tenant.user.id,
            category="creator_profile",
            memory_type="content_format",
            content=payload.content_format,
            verification_status="creator_stated",
        )
    if payload.stated_goals is not None:
        service.save_memory(
            workspace_id=tenant.workspace_id,
            user_id=tenant.user.id,
            category="creator_profile",
            memory_type="content_goals",
            content=", ".join(payload.stated_goals),
            structured_payload={"goals": payload.stated_goals},
            verification_status="creator_stated",
        )

    return service.get_creator_profile(tenant.workspace_id)
