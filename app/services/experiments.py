from sqlalchemy.orm import Session
from app.database.models import ExperimentRecord


class ExperimentService:
    def __init__(self, db: Session):
        self.db = db

    def register_experiment(
        self,
        workspace_id: str,
        channel_id: str,
        hypothesis: str,
        target_metric: str,
        baseline_value: float,
        expected_value: float,
    ) -> ExperimentRecord:
        exp = ExperimentRecord(
            workspace_id=workspace_id,
            channel_id=channel_id,
            hypothesis=hypothesis,
            target_metric=target_metric,
            baseline_value=baseline_value,
            expected_value=expected_value,
            status="in_progress",
        )
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)
        return exp

    def list_experiments(self, workspace_id: str) -> list[ExperimentRecord]:
        return (
            self.db.query(ExperimentRecord)
            .filter(ExperimentRecord.workspace_id == workspace_id)
            .order_by(ExperimentRecord.created_at.desc())
            .all()
        )

    def record_outcome(
        self,
        experiment_id: str,
        workspace_id: str,
        observed_value: float,
        status: str,
        notes: str,
    ) -> ExperimentRecord | None:
        exp = (
            self.db.query(ExperimentRecord)
            .filter(ExperimentRecord.id == experiment_id, ExperimentRecord.workspace_id == workspace_id)
            .first()
        )
        if exp:
            exp.observed_value = observed_value
            exp.status = status
            exp.notes = notes
            self.db.commit()
            self.db.refresh(exp)
        return exp
