from datetime import datetime, timezone
from typing import Any


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _format_count(val: int | float | None) -> str:
    if val is None:
        return "--"
    num = float(val)
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{int(num):,}" if num.is_integer() else f"{num:.1f}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def _parse_short_date(dt_str: Any) -> str:
    if not dt_str:
        return "Upload"
    try:
        clean = str(dt_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        return dt.strftime("%b %d")
    except Exception:
        parts = str(dt_str).split("T")[0].split("-")
        if len(parts) == 3:
            return f"{parts[1]}/{parts[2]}"
        return str(dt_str)[:10]


def _days_since(dt_str: Any) -> int:
    if not dt_str:
        return 1
    try:
        clean = str(dt_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        now = datetime.now(timezone.utc)
        diff = (now - dt).total_seconds() / 86400.0
        return max(1, int(diff))
    except Exception:
        return 1


class AnalyticsService:
    @staticmethod
    def compute_channel_analytics(
        metrics_raw: dict[str, Any] | None,
        period: str = "28d",
    ) -> dict[str, Any]:
        """Deterministic computation of channel overview baselines, 3-chart performance panel, and 4 metric cards.
        Supports both Studio daily timeseries and Idea 1: Recent Uploads Velocity Curve.
        """
        if not metrics_raw:
            metrics_raw = {}

        period_days = 28
        if period == "7d":
            period_days = 7
        elif period == "90d":
            period_days = 90

        subscribers = int(metrics_raw.get("subscribers") or 0)
        total_views = int(metrics_raw.get("total_views") or 0)
        watch_time_hours = _safe_float(metrics_raw.get("watch_time_hours"))
        ctr_val = _safe_float(metrics_raw.get("ctr"))
        avg_retention_pct = _safe_float(metrics_raw.get("avg_retention_percent"))
        avg_duration_sec = _safe_float(metrics_raw.get("avg_duration_sec"))

        # Check for Studio daily timeseries observations
        timeseries = metrics_raw.get("timeseries") or {}
        raw_dates: list[str] = timeseries.get("dates") or []
        raw_views: list[int | float] = timeseries.get("views") or timeseries.get("current_period_views") or []
        raw_wt: list[float] = timeseries.get("watch_time") or []
        raw_subs: list[int | float] = timeseries.get("subscribers") or timeseries.get("subscribers_gained") or []
        raw_subs_gained: list[int | float] = timeseries.get("subscribers_gained") or []

        has_studio_timeseries = bool(raw_dates and raw_views and len(raw_dates) > 1)

        # Check for Idea 1: Recent Uploads (Real data from YouTube Data API v3)
        recent_videos_input: list[dict] = metrics_raw.get("recent_videos") or []
        analyzed_video = metrics_raw.get("analyzed_video")

        all_vids = []
        if analyzed_video and isinstance(analyzed_video, dict) and analyzed_video.get("video_id"):
            all_vids.append(analyzed_video)
        if recent_videos_input and isinstance(recent_videos_input, list):
            for v in recent_videos_input:
                if isinstance(v, dict) and v.get("video_id"):
                    if not any(ex.get("video_id") == v.get("video_id") for ex in all_vids):
                        all_vids.append(v)

        # Sort videos chronologically by publish date (oldest to newest)
        def _get_sort_key(vid):
            return vid.get("published_at") or ""

        sorted_vids = sorted(all_vids, key=_get_sort_key)
        has_recent_uploads = len(sorted_vids) >= 1

        # Case A: YouTube Studio Daily Timeseries Exists
        if has_studio_timeseries:
            n = len(raw_dates)
            slice_len = min(period_days, n)
            current_dates = raw_dates[-slice_len:]
            current_views = [round(float(v), 1) for v in raw_views[-slice_len:]]
            current_wt = [round(float(w), 2) for w in raw_wt[-slice_len:]] if raw_wt else []
            current_subs = [round(float(s), 1) for s in raw_subs[-slice_len:]] if raw_subs else []
            current_subs_gained = [round(float(sg), 1) for sg in raw_subs_gained[-slice_len:]] if raw_subs_gained else []

            prev_views = [round(float(v), 1) for v in raw_views[-slice_len * 2 : -slice_len]] if n >= slice_len * 2 else None
            prev_wt = [round(float(w), 2) for w in raw_wt[-slice_len * 2 : -slice_len]] if (raw_wt and n >= slice_len * 2) else None
            prev_subs = [round(float(s), 1) for s in raw_subs[-slice_len * 2 : -slice_len]] if (raw_subs and n >= slice_len * 2) else None

            total_period_views = sum(current_views)
            latest_views = current_views[-1] if current_views else 0
            total_period_wt = sum(current_wt) if current_wt else None
            latest_wt = current_wt[-1] if current_wt else None
            net_subs = sum(current_subs_gained) if current_subs_gained else (current_subs[-1] - current_subs[0] if len(current_subs) > 1 else None)
            latest_subs = current_subs[-1] if current_subs else None

            def _calc_delta(curr: float, prev: float | None):
                if prev is None or prev == 0:
                    return None, "neutral"
                pct = ((curr - prev) / prev) * 100
                trend = "up" if pct > 0 else ("down" if pct < 0 else "neutral")
                return f"{'+' if pct > 0 else ''}{pct:.1f}%", trend

            views_delta_str, views_trend = _calc_delta(total_period_views, sum(prev_views) if prev_views else None)
            wt_delta_str, wt_trend = _calc_delta(total_period_wt or 0, sum(prev_wt) if prev_wt else None)
            subs_delta_str, subs_trend = _calc_delta(net_subs or 0, sum(prev_subs) if prev_subs else None)

            metric_cards = {
                "views": {
                    "id": "views",
                    "title": "Views",
                    "value": _format_count(total_period_views),
                    "unit": "views",
                    "period": f"Last {period_days} days",
                    "source": "YouTube Studio Export",
                    "badge": f"{period_days}-Day Verified",
                    "is_available": True,
                    "change_pct": views_delta_str,
                    "trend": views_trend,
                    "accent": "blue",
                },
                "watch_time": {
                    "id": "watch_time",
                    "title": "Watch Time",
                    "value": f"{total_period_wt:.1f} hrs" if total_period_wt is not None else "--",
                    "unit": "hours",
                    "period": f"Last {period_days} days",
                    "source": "YouTube Studio Export",
                    "badge": "Studio Analytics",
                    "is_available": total_period_wt is not None,
                    "change_pct": wt_delta_str,
                    "trend": wt_trend,
                    "accent": "green",
                },
                "ctr": {
                    "id": "ctr",
                    "title": "Click-Through Rate (CTR)",
                    "value": f"{ctr_val:.1f}%" if ctr_val is not None else "--",
                    "unit": "%",
                    "period": f"Last {period_days} days",
                    "source": "YouTube Studio Export",
                    "badge": "Studio Analytics",
                    "is_available": ctr_val is not None,
                    "change_pct": None,
                    "trend": "neutral",
                    "accent": "orange",
                },
                "avg_duration": {
                    "id": "avg_duration",
                    "title": "Avg. View Duration",
                    "value": _format_duration(avg_duration_sec) if avg_duration_sec is not None else (f"{avg_retention_pct:.0f}%" if avg_retention_pct is not None else "--"),
                    "unit": "min:sec" if avg_duration_sec is not None else "%",
                    "period": f"Last {period_days} days",
                    "source": "YouTube Studio Export",
                    "badge": "Studio Analytics",
                    "is_available": (avg_duration_sec is not None or avg_retention_pct is not None),
                    "change_pct": None,
                    "trend": "neutral",
                    "accent": "purple",
                },
            }

            performance_panel = {
                "has_timeseries": True,
                "mode": "daily_studio",
                "period": period,
                "panel_title": "Channel Performance",
                "period_badge": f"Last {period_days} Days",
                "subtitle": "Daily trajectory across views, watch time, and subscriber momentum",
                "legend_labels": {"views": "Views", "watch_time": "Watch Time", "subscribers": "Subscribers"},
                "views_chart": {
                    "id": "views",
                    "title": "Views",
                    "badge": "Daily Trend",
                    "color": "#3B82F6",
                    "gradient_start": "rgba(59, 130, 246, 0.25)",
                    "gradient_stop": "rgba(59, 130, 246, 0.0)",
                    "is_available": True,
                    "dates": current_dates,
                    "values": current_views,
                    "previous_values": prev_views,
                    "latest_value": _format_count(latest_views),
                    "latest_sub": "latest day",
                    "unit": "views",
                },
                "watch_time_chart": {
                    "id": "watch_time",
                    "title": "Watch Time",
                    "badge": "Hours",
                    "color": "#10B981",
                    "gradient_start": "rgba(16, 185, 129, 0.25)",
                    "gradient_stop": "rgba(16, 185, 129, 0.0)",
                    "is_available": len(current_wt) > 0,
                    "dates": current_dates,
                    "values": current_wt,
                    "previous_values": prev_wt,
                    "latest_value": f"{latest_wt:.1f} hrs" if latest_wt is not None else "--",
                    "latest_sub": "latest day",
                    "unit": "hours",
                },
                "subscribers_chart": {
                    "id": "subscribers",
                    "title": "Subscribers",
                    "badge": "Net Momentum",
                    "color": "#8B5CF6",
                    "gradient_start": "rgba(139, 92, 246, 0.25)",
                    "gradient_stop": "rgba(139, 92, 246, 0.0)",
                    "is_available": len(current_subs) > 0,
                    "dates": current_dates,
                    "values": current_subs,
                    "gained_values": current_subs_gained,
                    "previous_values": prev_subs,
                    "latest_value": _format_count(latest_subs),
                    "latest_sub": "period net gained" if net_subs else "period total",
                    "unit": "subscribers",
                },
            }

        # Case B: Idea 1 — Recent Uploads Velocity Curve (100% Real Public YouTube API Data)
        elif has_recent_uploads:
            u_dates = [_parse_short_date(v.get("published_at")) for v in sorted_vids]
            u_views = [int(v.get("view_count", 0) or 0) for v in sorted_vids]
            u_velocities = [
                round(int(v.get("view_count", 0) or 0) / _days_since(v.get("published_at")), 1)
                for v in sorted_vids
            ]
            u_likes = [int(v.get("like_count", 0) or 0) for v in sorted_vids]
            u_titles = [v.get("title", f"Video {i+1}") for i, v in enumerate(sorted_vids)]

            total_recent_views = sum(u_views)
            avg_views_per_vid = round(total_recent_views / len(u_views), 1) if u_views else 0
            total_recent_likes = sum(u_likes)
            avg_like_ratio = round((total_recent_likes / max(total_recent_views, 1)) * 100, 1)
            peak_velocity = max(u_velocities) if u_velocities else 0.0
            latest_velocity = u_velocities[-1] if u_velocities else 0.0

            views_display = _format_count(total_views if total_views > 0 else total_recent_views)

            metric_cards = {
                "views": {
                    "id": "views",
                    "title": "Views",
                    "value": views_display,
                    "unit": "views",
                    "period": "Lifetime Public",
                    "source": "YouTube Data API (Public)",
                    "badge": "Verified Public Count",
                    "is_available": True,
                    "change_pct": None,
                    "trend": "up",
                    "accent": "blue",
                },
                "watch_time": {
                    "id": "watch_time",
                    "title": "Avg. Views / Upload",
                    "value": _format_count(avg_views_per_vid),
                    "unit": "views/video",
                    "period": f"Recent {len(sorted_vids)} Uploads",
                    "source": "YouTube Data API (Public)",
                    "badge": "Observed Uploads",
                    "is_available": True,
                    "change_pct": None,
                    "trend": "neutral",
                    "accent": "green",
                },
                "ctr": {
                    "id": "ctr",
                    "title": "Observed Like Ratio",
                    "value": f"{avg_like_ratio:.1f}%",
                    "unit": "like/view",
                    "period": f"Recent {len(sorted_vids)} Uploads",
                    "source": "YouTube Data API (Public)",
                    "badge": "Public Engagement",
                    "is_available": True,
                    "change_pct": None,
                    "trend": "neutral",
                    "accent": "orange",
                },
                "avg_duration": {
                    "id": "avg_duration",
                    "title": "Peak Upload Velocity",
                    "value": f"~{peak_velocity:.0f}",
                    "unit": "views/day",
                    "period": "Peak Recent Upload",
                    "source": "YouTube Data API (Public)",
                    "badge": "Velocity Peak",
                    "is_available": True,
                    "change_pct": None,
                    "trend": "up",
                    "accent": "purple",
                },
            }

            performance_panel = {
                "has_timeseries": True,
                "mode": "recent_uploads",
                "period": period,
                "panel_title": "Channel Performance",
                "period_badge": f"Recent Uploads Trajectory ({len(sorted_vids)} Videos)",
                "subtitle": "Observed view counts, velocity, and engagement across recent channel uploads",
                "legend_labels": {
                    "views": "Views / Upload",
                    "watch_time": "Views / Day (Velocity)",
                    "subscribers": "Engagement (Likes)",
                },
                "views_chart": {
                    "id": "views",
                    "title": "Views per Upload",
                    "badge": "Observed Views",
                    "color": "#3B82F6",
                    "gradient_start": "rgba(59, 130, 246, 0.25)",
                    "gradient_stop": "rgba(59, 130, 246, 0.0)",
                    "is_available": True,
                    "dates": u_dates,
                    "values": u_views,
                    "titles": u_titles,
                    "previous_values": None,
                    "latest_value": _format_count(u_views[-1]),
                    "latest_sub": "latest upload views",
                    "unit": "views",
                },
                "watch_time_chart": {
                    "id": "watch_time",
                    "title": "Upload Velocity",
                    "badge": "Views / Day",
                    "color": "#10B981",
                    "gradient_start": "rgba(16, 185, 129, 0.25)",
                    "gradient_stop": "rgba(16, 185, 129, 0.0)",
                    "is_available": True,
                    "dates": u_dates,
                    "values": u_velocities,
                    "titles": u_titles,
                    "previous_values": None,
                    "latest_value": f"{latest_velocity:.0f} / day",
                    "latest_sub": "latest upload pace",
                    "unit": "views/day",
                },
                "subscribers_chart": {
                    "id": "subscribers",
                    "title": "Audience Engagement",
                    "badge": "Observed Likes",
                    "color": "#8B5CF6",
                    "gradient_start": "rgba(139, 92, 246, 0.25)",
                    "gradient_stop": "rgba(139, 92, 246, 0.0)",
                    "is_available": True,
                    "dates": u_dates,
                    "values": u_likes,
                    "titles": u_titles,
                    "previous_values": None,
                    "latest_value": _format_count(u_likes[-1]),
                    "latest_sub": "latest upload likes",
                    "unit": "likes",
                },
            }

        # Case C: Completely Empty (No channel analyzed yet)
        else:
            views_display = _format_count(total_views if total_views > 0 else None)
            metric_cards = {
                "views": {
                    "id": "views",
                    "title": "Views",
                    "value": views_display,
                    "unit": "views",
                    "period": "Lifetime Public",
                    "source": "YouTube Data API (Public)",
                    "badge": "Verified Public Count",
                    "is_available": total_views > 0,
                    "change_pct": None,
                    "trend": "neutral",
                    "accent": "blue",
                },
                "watch_time": {
                    "id": "watch_time",
                    "title": "Watch Time",
                    "value": f"{watch_time_hours:.1f} hrs" if watch_time_hours is not None else "--",
                    "unit": "hours",
                    "period": "Studio Analytics" if watch_time_hours is not None else "Unavailable",
                    "source": "YouTube Studio Export" if watch_time_hours is not None else "Requires Studio Export",
                    "badge": "Studio Analytics" if watch_time_hours is not None else "Requires Studio Export",
                    "is_available": watch_time_hours is not None,
                    "change_pct": None,
                    "trend": "neutral",
                    "accent": "green",
                },
                "ctr": {
                    "id": "ctr",
                    "title": "Click-Through Rate (CTR)",
                    "value": f"{ctr_val:.1f}%" if ctr_val is not None else "--",
                    "unit": "%",
                    "period": "Studio Analytics" if ctr_val is not None else "Unavailable",
                    "source": "YouTube Studio Export" if ctr_val is not None else "Requires Studio Export",
                    "badge": "Studio Analytics" if ctr_val is not None else "Requires Studio Export",
                    "is_available": ctr_val is not None,
                    "change_pct": None,
                    "trend": "neutral",
                    "accent": "orange",
                },
                "avg_duration": {
                    "id": "avg_duration",
                    "title": "Avg. View Duration",
                    "value": _format_duration(avg_duration_sec) if avg_duration_sec is not None else (f"{avg_retention_pct:.0f}%" if avg_retention_pct is not None else "--"),
                    "unit": "min:sec" if avg_duration_sec is not None else "%",
                    "period": "Studio Analytics" if (avg_duration_sec is not None or avg_retention_pct is not None) else "Unavailable",
                    "source": "YouTube Studio Export" if (avg_duration_sec is not None or avg_retention_pct is not None) else "Requires Studio Export",
                    "badge": "Studio Analytics" if (avg_duration_sec is not None or avg_retention_pct is not None) else "Requires Studio Export",
                    "is_available": (avg_duration_sec is not None or avg_retention_pct is not None),
                    "change_pct": None,
                    "trend": "neutral",
                    "accent": "purple",
                },
            }

            performance_panel = {
                "has_timeseries": False,
                "mode": "empty",
                "period": period,
                "panel_title": "Channel Performance",
                "period_badge": "No Data",
                "subtitle": "Submit a YouTube channel or video URL on the Analyze page to view performance curves",
                "legend_labels": {"views": "Views", "watch_time": "Watch Time", "subscribers": "Subscribers"},
                "views_chart": {"is_available": False, "dates": [], "values": [], "latest_value": "--", "unit": "views"},
                "watch_time_chart": {"is_available": False, "dates": [], "values": [], "latest_value": "--", "unit": "hours"},
                "subscribers_chart": {"is_available": False, "dates": [], "values": [], "latest_value": "--", "unit": "subscribers"},
                "empty_state": {
                    "title": "No Channel Analyzed Yet",
                    "reason": "Submit a YouTube channel or video URL on the Analyze page to generate live performance curves.",
                    "action_label": "+ Run Analysis",
                    "action_url": "/analyze",
                },
            }

        # Backward compatibility fields
        summary_cards = {
            "total_views": {
                "value": metric_cards["views"]["value"],
                "delta": "Verified Public Count" if total_views > 0 else "",
                "trend": "up",
                "raw": total_views,
            },
            "ctr": {
                "value": metric_cards["ctr"]["value"],
                "delta": metric_cards["ctr"]["badge"],
                "trend": "neutral",
                "raw": ctr_val,
            },
            "avg_retention": {
                "value": metric_cards["avg_duration"]["value"],
                "delta": metric_cards["avg_duration"]["badge"],
                "trend": "neutral",
                "raw": avg_retention_pct,
            },
            "subscribers": {
                "value": _format_count(subscribers if subscribers > 0 else None),
                "delta": "Verified Public Count" if subscribers > 0 else "",
                "trend": "up",
                "raw": subscribers,
            },
        }

        perf_dates = performance_panel.get("views_chart", {}).get("dates", [])
        perf_views = performance_panel.get("views_chart", {}).get("values", [])

        performance_chart = {
            "dates": perf_dates,
            "current_period_views": perf_views,
            "previous_period_views": [],
            "peak_day": max(perf_views) if perf_views else None,
        }

        return {
            "metric_cards": metric_cards,
            "performance_panel": performance_panel,
            "summary_cards": summary_cards,
            "performance_chart": performance_chart,
            "retention_breakdown": {
                "first_15s_drop": metrics_raw.get("first_15s_drop"),
                "midpoint_retention": metrics_raw.get("midpoint_retention"),
                "benchmark_average": None,
            },
        }
