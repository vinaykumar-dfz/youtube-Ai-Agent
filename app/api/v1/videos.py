from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.database.session import get_db
from app.services.video_analysis import VideoAnalysisService

router = APIRouter(prefix="/videos", tags=["Video Analysis"])


class VideoAnalyzeRequest(BaseModel):
    target: str = Field(..., min_length=5, description="YouTube Video URL or 11-character video ID")


@router.post("/analyze")
def analyze_video_post(
    payload: VideoAnalyzeRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Analyze a single YouTube video URL or ID and return complete evidence-backed dashboard payload."""
    try:
        service = VideoAnalysisService(db=db)
        return service.analyze_video(video_target=payload.target, workspace_id=tenant.workspace_id)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze video: {str(exc)}",
        )


@router.get("/analyze")
def analyze_video_get(
    target: str | None = Query(default=None, description="YouTube Video URL or ID"),
    url: str | None = Query(default=None, description="Alias for target URL"),
    video_id: str | None = Query(default=None, description="Alias for video ID"),
    v: str | None = Query(default=None, description="Alias for YouTube video parameter v"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """GET endpoint supporting deep-linking with query parameters."""
    resolved_target = target or url or video_id or v
    if not resolved_target:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'target', 'url', or 'video_id' query parameter.",
        )

    try:
        service = VideoAnalysisService(db=db)
        return service.analyze_video(video_target=resolved_target, workspace_id=tenant.workspace_id)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze video: {str(exc)}",
        )
