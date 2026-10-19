from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.database.session import get_db
from app.services.experiments import ExperimentService

router = APIRouter(prefix="/experiments", tags=["Hypothesis & Experiment Tracking"])


class CreateExperimentRequest(BaseModel):
    channel_id: str = Field(default="UC_x5XG1OV2P")
    hypothesis: str = Field(min_length=10)
    target_metric: str = Field(default="30s Retention")
    baseline_value: float
    expected_value: float


class RecordOutcomeRequest(BaseModel):
    observed_value: float
    status: str = Field(default="validated")  # validated, refuted
    notes: str


@router.get("")
def list_experiments(
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    service = ExperimentService(db)
    return service.list_experiments(tenant.workspace_id)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_experiment(
    payload: CreateExperimentRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    service = ExperimentService(db)
    return service.register_experiment(
        workspace_id=tenant.workspace_id,
        channel_id=payload.channel_id,
        hypothesis=payload.hypothesis,
        target_metric=payload.target_metric,
        baseline_value=payload.baseline_value,
        expected_value=payload.expected_value,
    )


@router.put("/{experiment_id}/outcome")
def record_outcome(
    experiment_id: str,
    payload: RecordOutcomeRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    service = ExperimentService(db)
    exp = service.record_outcome(
        experiment_id=experiment_id,
        workspace_id=tenant.workspace_id,
        observed_value=payload.observed_value,
        status=payload.status,
        notes=payload.notes,
    )
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return exp
