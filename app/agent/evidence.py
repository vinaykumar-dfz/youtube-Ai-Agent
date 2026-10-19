from sqlalchemy.orm import Session
from app.database.models import EvidenceRecord, FindingRecord


class EvidenceService:
    def __init__(self, db: Session):
        self.db = db

    def register_evidence(
        self,
        workspace_id: str,
        job_id: str,
        source_type: str,
        title: str,
        snippet: str,
        url: str | None = None,
        confidence_score: float = 1.0,
    ) -> EvidenceRecord:
        record = EvidenceRecord(
            workspace_id=workspace_id,
            job_id=job_id,
            source_type=source_type,
            title=title,
            snippet=snippet,
            url=url,
            confidence_score=confidence_score,
        )
        self.db.add(record)
        self.db.flush()
        return record

    def register_finding(
        self,
        workspace_id: str,
        job_id: str,
        category: str,
        epistemic_tier: str,
        statement: str,
        explanation: str,
        recommendation: str,
        evidence_ids: list[str],
    ) -> FindingRecord:
        record = FindingRecord(
            workspace_id=workspace_id,
            job_id=job_id,
            category=category,
            epistemic_tier=epistemic_tier,
            statement=statement,
            explanation=explanation,
            recommendation=recommendation,
            evidence_ids=evidence_ids,
        )
        self.db.add(record)
        self.db.flush()
        return record

    def get_findings_for_job(self, job_id: str) -> list[FindingRecord]:
        return self.db.query(FindingRecord).filter(FindingRecord.job_id == job_id).all()

    def get_evidence_for_job(self, job_id: str) -> list[EvidenceRecord]:
        return self.db.query(EvidenceRecord).filter(EvidenceRecord.job_id == job_id).all()
