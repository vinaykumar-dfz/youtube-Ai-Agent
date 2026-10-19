from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.database.models import AnalysisReport, EvidenceRecord
from app.database.session import get_db

router = APIRouter(prefix="/reports", tags=["Analysis Reports"])


@router.get("/latest")
def get_latest_report(
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Retrieve the most recent analysis report for the workspace."""
    report = (
        db.query(AnalysisReport)
        .filter(AnalysisReport.workspace_id == tenant.workspace_id)
        .order_by(AnalysisReport.created_at.desc())
        .first()
    )
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis report found yet. Run an analysis on the Analyze page first.",
        )
    return _build_report_payload(report, db)


@router.get("/{job_id}")
def get_report_by_job_id(
    job_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Retrieve analysis report by job ID."""
    report = (
        db.query(AnalysisReport)
        .filter(AnalysisReport.job_id == job_id, AnalysisReport.workspace_id == tenant.workspace_id)
        .first()
    )
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report for job '{job_id}' not found.",
        )
    return _build_report_payload(report, db)


def _build_report_payload(report: AnalysisReport, db: Session) -> dict:
    evidence_records = (
        db.query(EvidenceRecord)
        .filter(EvidenceRecord.job_id == report.job_id)
        .all()
    )

    channel_info = report.metrics_summary.get("channel_info", {})
    subs_formatted = report.metrics_summary.get("subscribers", {}).get("value")
    subs_raw = report.metrics_summary.get("subscribers", {}).get("raw", 0)

    if subs_formatted and subs_formatted != "--":
        subscriber_str = f"{subs_formatted} subscribers"
    elif subs_raw:
        subscriber_str = f"{subs_raw:,} subscribers"
    else:
        subscriber_str = "Channel Verified"

    custom_url = channel_info.get("custom_url") or f"@{report.channel_title.lower().replace(' ', '')}"
    channel_url = channel_info.get("channel_url")
    if not channel_url:
        if custom_url.startswith("@"):
            channel_url = f"https://www.youtube.com/{custom_url}"
        elif report.channel_id:
            channel_url = f"https://www.youtube.com/channel/{report.channel_id}"
        else:
            channel_url = "https://www.youtube.com"

    thumbnail_url = channel_info.get("thumbnail_url") or "/assets/images/creator-avatar.svg"

    return {
        "id": report.id,
        "job_id": report.job_id,
        "channel_title": report.channel_title,
        "channel_id": report.channel_id,
        "custom_url": custom_url,
        "channel_url": channel_url,
        "thumbnail_url": thumbnail_url,
        "subscriber_count": subscriber_str,
        "date_range": "Observed 28-Day Window",
        "summary": report.summary,
        "metrics_summary": report.metrics_summary,
        "competitor_intelligence": report.metrics_summary.get("competitor_intelligence", {}),
        "analyzed_video": channel_info.get("analyzed_video"),
        "recent_videos": channel_info.get("recent_videos", []),
        "key_findings": report.key_findings,
        "recommendations": report.recommendations,
        "evidence_sources": [
            {
                "id": ev.id,
                "title": ev.title,
                "type": ev.source_type,
                "snippet": ev.snippet,
                "url": ev.url,
                "confidence": ev.confidence_score,
            }
            for ev in evidence_records
        ],
        "created_at": report.created_at,
    }

