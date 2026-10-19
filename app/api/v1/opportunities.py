from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.database.session import get_db
from app.services.opportunity_engine import OpportunityEngineService

router = APIRouter(prefix="/opportunities", tags=["Content Opportunities"])


@router.get("")
def list_opportunities(
    category: str | None = Query(None, description="Filter by category: Tutorials, Reviews, News, Tools, High Potential"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Retrieve ranked content opportunities matching UI reference Screen 5."""
    service = OpportunityEngineService(db)
    opps = service.list_opportunities(tenant.workspace_id, category=category)
    return [
        {
            "id": o.id,
            "title": o.title,
            "category": o.topic_category,
            "opportunity_score": o.opportunity_score,
            "demand_score": o.demand_score,
            "saturation_score": o.saturation_score,
            "authority_score": o.authority_score,
            "differentiation_angle": o.differentiation_angle,
            "target_audience": o.target_audience,
            "tags": o.tags,
            "is_bookmarked": o.is_bookmarked,
        }
        for o in opps
    ]


@router.post("/{opp_id}/bookmark")
def toggle_bookmark(
    opp_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    service = OpportunityEngineService(db)
    opp = service.toggle_bookmark(opp_id, tenant.workspace_id)
    if not opp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Opportunity not found")
    return {"id": opp.id, "is_bookmarked": opp.is_bookmarked}
