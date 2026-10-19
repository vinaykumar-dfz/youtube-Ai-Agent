from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.core.logging import logger
from app.database.session import get_db
from app.services.creator_struggles import UniversalStrugglesEngine

router = APIRouter(prefix="/creator-struggles", tags=["Creator Struggles"])


@router.get("/suggestions")
def get_suggested_queries(
    tenant: TenantContext = Depends(get_tenant_context),
):
    """Retrieve curated prompt ideas covering common and advanced YouTube creator struggles."""
    return {
        "suggestions": UniversalStrugglesEngine.get_suggested_queries(),
    }


@router.get("/investigate")
@router.get("/search")
def investigate_creator_problem(
    q: str = Query(
        "My coding tutorials get impressions, but viewers do not click, and those who click leave in the first minute.",
        description="Arbitrary YouTube creator problem in natural language",
    ),
    force_refresh: bool = Query(False, description="Bypass cache and pull fresh multi-source evidence"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Investigate virtually any YouTube creator problem across YouTube, Reddit, X, and Articles concurrently."""
    clean_q = q.strip()
    if not clean_q:
        clean_q = "My coding tutorials get impressions, but viewers do not click, and those who click leave in the first minute."

    try:
        engine = UniversalStrugglesEngine()
        payload = engine.investigate(
            user_query=clean_q,
            force_refresh=force_refresh,
            db=db,
        )
        return payload
    except Exception as exc:
        logger.error(f"Error investigating creator problem: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation failed: {exc}",
        )
