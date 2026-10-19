from sqlalchemy.orm import Session
from app.database.models import ContentOpportunity


class OpportunityEngineService:
    def __init__(self, db: Session):
        self.db = db

    def calculate_score(self, demand: float, saturation: float, authority: float) -> int:
        """Explicit multi-factor scoring formula from PRD and architecture contracts."""
        raw = (demand * 3.5) + ((10.0 - saturation) * 2.5) + (authority * 3.0)
        return min(99, max(10, int(raw)))

    def create_opportunity(
        self,
        workspace_id: str,
        title: str,
        topic_category: str,
        demand_score: float,
        saturation_score: float,
        authority_score: float,
        differentiation_angle: str,
        target_audience: str,
        tags: list[str],
        job_id: str | None = None,
        is_bookmarked: bool = False,
    ) -> ContentOpportunity:
        opp_score = self.calculate_score(demand_score, saturation_score, authority_score)
        opp = ContentOpportunity(
            workspace_id=workspace_id,
            job_id=job_id,
            title=title,
            topic_category=topic_category,
            demand_score=demand_score,
            saturation_score=saturation_score,
            authority_score=authority_score,
            opportunity_score=opp_score,
            differentiation_angle=differentiation_angle,
            target_audience=target_audience,
            tags=tags,
            is_bookmarked=is_bookmarked,
        )
        self.db.add(opp)
        self.db.flush()
        return opp

    def list_opportunities(self, workspace_id: str, category: str | None = None) -> list[ContentOpportunity]:
        query = self.db.query(ContentOpportunity).filter(ContentOpportunity.workspace_id == workspace_id)
        if category and category.lower() != "all":
            query = query.filter(ContentOpportunity.topic_category.ilike(f"%{category}%"))
        return query.order_by(ContentOpportunity.opportunity_score.desc()).all()

    def toggle_bookmark(self, opportunity_id: str, workspace_id: str) -> ContentOpportunity | None:
        opp = (
            self.db.query(ContentOpportunity)
            .filter(ContentOpportunity.id == opportunity_id, ContentOpportunity.workspace_id == workspace_id)
            .first()
        )
        if opp:
            opp.is_bookmarked = not opp.is_bookmarked
            self.db.commit()
            self.db.refresh(opp)
        return opp
