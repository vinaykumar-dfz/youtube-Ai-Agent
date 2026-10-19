from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.core.errors import YouTubeQuotaExceededException, YouTubeAPIException
from app.core.logging import logger
from app.database.session import get_db
from app.services.niche_radar import NicheRadarService

router = APIRouter(prefix="/niche-radar", tags=["Niche Radar"])


class RefreshRequest(BaseModel):
    niche: str
    time_period: str = "all"
    category: str = "most_viewed"
    sort_by: str = "default"


@router.get("/suggest")
def suggest_niches(
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Suggest top niches based on connected channel or system topics."""
    try:
        return NicheRadarService.suggest_niches(tenant.workspace_id, db)
    except Exception as exc:
        logger.error(f"Error suggesting niches: {exc}")
        return {
            "suggested_niches": [
                "AI Agents",
                "Python Automation",
                "Autonomous Coding",
                "Machine Learning",
                "YouTube Growth Strategy",
                "SaaS Development",
            ],
            "detected_niche": "AI Agents",
            "connected_channel": None,
        }


@router.get("/search")
def search_niche_radar(
    niche: str = Query(..., description="Niche keyword or topic"),
    time_period: str = Query("all", description="Time period filter: 7d, 30d, 90d, all"),
    category: str = Query("most_viewed", description="Category: most_viewed, high_average, recent, momentum, small_channel"),
    sort_by: str = Query("default", description="Optional secondary sort: default, views, velocity, recent, likes"),
    force_refresh: bool = Query(False, description="Bypass cache and pull fresh YouTube data"),
    page_token: str | None = Query(None, description="YouTube pagination token"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Search and retrieve real YouTube videos and performance metrics for a specific niche."""
    clean_niche = niche.strip()
    if not clean_niche:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Niche keyword cannot be empty.",
        )

    try:
        results = NicheRadarService.get_or_search_niche_radar(
            workspace_id=tenant.workspace_id,
            niche=clean_niche,
            time_period=time_period,
            category=category,
            sort_by=sort_by,
            force_refresh=force_refresh,
            page_token=page_token,
            db=db,
        )
        return results
    except YouTubeQuotaExceededException:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="YouTube Data API quota has been exhausted. Please wait for quota replenishment or use cached results.",
        )
    except YouTubeAPIException as yt_err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"YouTube API error: {yt_err.message}",
        )
    except Exception as exc:
        logger.error(f"Error in Niche Radar search: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error processing Niche Radar: {str(exc)}",
        )


@router.post("/refresh")
def force_refresh_niche_radar(
    req: RefreshRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Force fresh query to YouTube API, record observation snapshot, and update cache."""
    clean_niche = req.niche.strip()
    if not clean_niche:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Niche keyword cannot be empty.",
        )

    try:
        results = NicheRadarService.get_or_search_niche_radar(
            workspace_id=tenant.workspace_id,
            niche=clean_niche,
            time_period=req.time_period,
            category=req.category,
            sort_by=req.sort_by,
            force_refresh=True,
            db=db,
        )
        return results
    except YouTubeQuotaExceededException:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="YouTube Data API quota has been exhausted. Please wait for quota replenishment.",
        )
    except Exception as exc:
        logger.error(f"Error refreshing Niche Radar: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh niche data: {str(exc)}",
        )
