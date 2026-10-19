from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_tenant_context
from app.database.models import AnalysisReport, ChannelConnection, ContentOpportunity
from app.database.session import get_db
from app.services.analytics import AnalyticsService

router = APIRouter(prefix="/channels", tags=["Channel Overview"])


@router.get("/overview")
def get_channel_overview(
    period: str = Query("28d"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Retrieve full dashboard data matching UI Reference Screen 2."""
    channel = (
        db.query(ChannelConnection)
        .filter(ChannelConnection.workspace_id == tenant.workspace_id)
        .first()
    )

    # Check latest report for additional rich metadata (e.g. analyzed video, recent uploads)
    latest_report = (
        db.query(AnalysisReport)
        .filter(AnalysisReport.workspace_id == tenant.workspace_id)
        .order_by(AnalysisReport.created_at.desc())
        .first()
    )
    report_ms = (latest_report.metrics_summary or {}) if latest_report else {}
    report_info = report_ms.get("channel_info", {}) if isinstance(report_ms, dict) else {}

    if channel:
        custom_url = channel.custom_url or report_info.get("custom_url") or ""
        if custom_url.startswith("@"):
            channel_url = f"https://www.youtube.com/{custom_url}"
        elif channel.channel_id:
            channel_url = f"https://www.youtube.com/channel/{channel.channel_id}"
        else:
            channel_url = report_info.get("channel_url") or "https://www.youtube.com"

        thumb = channel.thumbnail_url
        if not thumb or thumb == "/assets/images/creator-avatar.svg":
            thumb = report_info.get("thumbnail_url") or "/assets/images/creator-avatar.svg"

        channel_info = {
            "title": channel.title or "Connected Channel",
            "custom_url": custom_url,
            "channel_url": channel_url,
            "channel_id": channel.channel_id or "",
            "subscriber_count": channel.subscriber_count or 0,
            "video_count": channel.video_count or 0,
            "view_count": channel.view_count or 0,
            "thumbnail_url": thumb,
            "analyzed_video": report_info.get("analyzed_video"),
            "recent_videos": report_info.get("recent_videos", []),
        }
    else:
        if latest_report:
            clean_title = latest_report.channel_title or "Channel"
            handle = report_info.get("custom_url") or f"@{clean_title.lower().replace(' ', '')}"

            subs_entry = report_ms.get("subscribers", {})
            raw_subs = subs_entry.get("raw", 0) if isinstance(subs_entry, dict) else (int(subs_entry) if subs_entry else 0)

            views_entry = report_ms.get("total_views", {})
            raw_views = views_entry.get("raw", 0) if isinstance(views_entry, dict) else (int(views_entry) if views_entry else 0)

            channel_url = report_info.get("channel_url") or (
                f"https://www.youtube.com/{handle}" if handle.startswith("@")
                else f"https://www.youtube.com/channel/{latest_report.channel_id or ''}"
            )
            channel_info = {
                "title": clean_title,
                "custom_url": handle,
                "channel_url": channel_url,
                "channel_id": latest_report.channel_id or "",
                "subscriber_count": raw_subs,
                "video_count": report_info.get("video_count", 0),
                "view_count": raw_views,
                "thumbnail_url": report_info.get("thumbnail_url") or "/assets/images/creator-avatar.svg",
                "analyzed_video": report_info.get("analyzed_video"),
                "recent_videos": report_info.get("recent_videos", []),
            }
        else:
            channel_info = {
                "title": "No Channel Connected",
                "custom_url": "",
                "channel_url": "",
                "channel_id": "",
                "subscriber_count": 0,
                "video_count": 0,
                "view_count": 0,
                "thumbnail_url": "/assets/images/creator-avatar.svg",
                "analyzed_video": None,
                "recent_videos": [],
            }

    # Analytics calculation (extracts timeseries, recent_videos, and private metrics)
    metrics_source = {
        "subscribers": channel_info["subscriber_count"],
        "total_views": channel_info["view_count"],
        "recent_videos": channel_info.get("recent_videos", []),
        "analyzed_video": channel_info.get("analyzed_video"),
        "ctr": None,
        "avg_retention_percent": None,
    }
    if report_ms and isinstance(report_ms, dict):
        if "timeseries" in report_ms:
            metrics_source["timeseries"] = report_ms["timeseries"]
        if "watch_time_hours" in report_ms:
            metrics_source["watch_time_hours"] = report_ms["watch_time_hours"]
        if "ctr" in report_ms:
            metrics_source["ctr"] = report_ms["ctr"]
        if "avg_retention_percent" in report_ms:
            metrics_source["avg_retention_percent"] = report_ms["avg_retention_percent"]
        if "avg_duration_sec" in report_ms:
            metrics_source["avg_duration_sec"] = report_ms["avg_duration_sec"]

    analytics_data = AnalyticsService.compute_channel_analytics(metrics_source, period=period)

    # Competitor intelligence from latest report if available
    comp_intel = report_ms.get("competitor_intelligence", {}) if isinstance(report_ms, dict) else {}

    # Top opportunities
    top_opps = (
        db.query(ContentOpportunity)
        .filter(ContentOpportunity.workspace_id == tenant.workspace_id)
        .order_by(ContentOpportunity.opportunity_score.desc())
        .limit(4)
        .all()
    )

    data_freshness = {
        "channel_synced_at": channel.updated_at.isoformat() if (channel and hasattr(channel, "updated_at") and channel.updated_at) else None,
        "latest_report_at": latest_report.created_at.isoformat() if (latest_report and hasattr(latest_report, "created_at") and latest_report.created_at) else None,
        "is_oauth_connected": bool(channel and getattr(channel, "access_token", None)),
        "has_recent_report": bool(latest_report is not None),
        "source": "YouTube Data API v3 & Channel Analysis Reports",
    }

    return {
        "channel": channel_info,
        "data_freshness": data_freshness,
        "metric_cards": analytics_data.get("metric_cards", {}),
        "performance_panel": analytics_data.get("performance_panel", {}),
        "summary_cards": analytics_data.get("summary_cards", {}),
        "performance_chart": analytics_data.get("performance_chart", {}),
        "competitor_intelligence": comp_intel,
        "top_opportunities": [
            {
                "id": o.id,
                "title": o.title,
                "category": o.topic_category,
                "score": o.opportunity_score,
                "tags": o.tags,
            }
            for o in top_opps
        ],
    }
