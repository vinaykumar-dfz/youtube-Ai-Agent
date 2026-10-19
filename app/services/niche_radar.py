import re
from datetime import datetime, timezone, timedelta
from typing import Any
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.database.models import (
    ChannelConnection,
    CreatorMemory,
    VideoObservation,
    NicheSearchCache,
)
from app.agent.providers.youtube import YouTubeAdapter


class NicheRadarService:
    """Service for searching, scoring, and caching real YouTube niche videos and observations."""

    @classmethod
    def suggest_niches(cls, workspace_id: str, db: Session) -> dict[str, Any]:
        """Suggest relevant content niches based on connected channel or system topics."""
        suggestions = [
            "AI Agents",
            "Python Automation",
            "Autonomous Coding",
            "Machine Learning",
            "YouTube Growth Strategy",
            "SaaS Development",
        ]

        active_connection = (
            db.query(ChannelConnection)
            .filter(ChannelConnection.workspace_id == workspace_id, ChannelConnection.is_active == True)
            .first()
        )

        detected_niche = None
        channel_title = None

        if active_connection:
            channel_title = active_connection.title
            # Check creator memories
            niche_mem = (
                db.query(CreatorMemory)
                .filter(
                    CreatorMemory.workspace_id == workspace_id,
                    CreatorMemory.memory_type.in_(["niche_definition", "content_niche", "creator_niche"]),
                )
                .first()
            )
            if niche_mem and niche_mem.structured_payload.get("niche"):
                detected_niche = niche_mem.structured_payload.get("niche")
            else:
                # Infer from channel title
                clean_title = re.sub(r"[^a-zA-Z0-9\s]", "", active_connection.title).strip()
                if clean_title:
                    detected_niche = clean_title
                    if detected_niche not in suggestions:
                        suggestions.insert(0, detected_niche)

        return {
            "suggested_niches": suggestions[:8],
            "detected_niche": detected_niche or "AI Agents",
            "connected_channel": channel_title,
        }

    @classmethod
    def get_or_search_niche_radar(
        cls,
        workspace_id: str,
        niche: str,
        time_period: str = "all",
        category: str = "most_viewed",
        sort_by: str = "default",
        force_refresh: bool = False,
        page_token: str | None = None,
        db: Session = None,
    ) -> dict[str, Any]:
        """Retrieve, rank, and track real niche video performance metrics."""
        clean_niche = niche.strip()
        if not clean_niche:
            return {
                "videos": [],
                "niche": "",
                "time_period": time_period,
                "category": category,
                "total_count": 0,
                "last_refreshed_at": None,
                "is_cached": False,
                "data_coverage": {},
            }

        now = datetime.now(timezone.utc)
        cache_duration = timedelta(minutes=60)

        # 1. Check cache if not forcing refresh and no pagination token
        if not force_refresh and not page_token and db is not None:
            cache_entry = (
                db.query(NicheSearchCache)
                .filter(
                    NicheSearchCache.workspace_id == workspace_id,
                    NicheSearchCache.niche_keyword.ilike(clean_niche),
                    NicheSearchCache.time_period == time_period,
                    NicheSearchCache.category == category,
                )
                .first()
            )
            if cache_entry and cache_entry.last_refreshed_at:
                # Make timezone-aware if needed
                cache_time = cache_entry.last_refreshed_at
                if cache_time.tzinfo is None:
                    cache_time = cache_time.replace(tzinfo=timezone.utc)

                if now - cache_time < cache_duration:
                    payload = cache_entry.cached_payload
                    payload["is_cached"] = True
                    payload["last_refreshed_at"] = cache_time.isoformat()
                    return payload

        # 2. Query YouTube Data API v3
        yt = YouTubeAdapter()
        # Map categories to search sort parameters when relevant
        yt_sort = "relevance"
        if category == "most_viewed" or sort_by == "views":
            yt_sort = "viewCount"
        elif category == "recent" or sort_by == "recent":
            yt_sort = "date"

        raw_result = yt.search_niche_videos(
            query=clean_niche,
            time_period=time_period,
            sort_order=yt_sort,
            max_results=30,
            page_token=page_token,
        )

        raw_videos = raw_result.get("videos", [])
        if not raw_videos:
            empty_payload = {
                "videos": [],
                "niche": clean_niche,
                "time_period": time_period,
                "category": category,
                "total_count": 0,
                "next_page_token": None,
                "last_refreshed_at": now.isoformat(),
                "is_cached": False,
                "data_coverage": raw_result.get("data_coverage", {}),
            }
            return empty_payload

        # 3. Calculate derived metrics, track observations, and generate ranking labels
        processed_videos = []
        recorded_observations = 0

        for item in raw_videos:
            vid = item.get("video_id")
            pub_raw = item.get("published_at")
            view_count = int(item.get("view_count", 0))
            channel_id = item.get("channel_id")
            channel_title = item.get("channel_title")
            channel_subs = item.get("channel_subscriber_count")

            # A. Calculate elapsed time & video age
            elapsed_days = 1.0
            age_formatted = "Unknown"
            if pub_raw:
                try:
                    pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                    diff = now - pub_dt
                    elapsed_seconds = max(diff.total_seconds(), 3600.0)
                    elapsed_days = max(elapsed_seconds / 86400.0, 0.05)

                    if elapsed_days < 1:
                        hours = max(int(elapsed_seconds / 3600), 1)
                        age_formatted = f"{hours}h ago"
                    elif elapsed_days < 30:
                        days = int(elapsed_days)
                        age_formatted = f"{days}d ago"
                    elif elapsed_days < 365:
                        months = int(elapsed_days / 30)
                        age_formatted = f"{months}mo ago"
                    else:
                        years = round(elapsed_days / 365, 1)
                        age_formatted = f"{years}y ago"
                except Exception:
                    pass

            # B. Lifetime-average views per day: total_views / elapsed_days_since_publication
            views_per_day = round(view_count / max(elapsed_days, 1.0))

            # C. Recent view growth: latest_observed_views - previous_observed_views
            # Only calculated when prior observation snapshot exists
            recent_growth = None
            growth_delta_hours = None
            growth_label = "Baseline recorded"

            if db is not None and vid:
                try:
                    # Look up prior observation for this video taken at least 2 minutes prior
                    prev_obs = (
                        db.query(VideoObservation)
                        .filter(
                            VideoObservation.video_id == vid,
                            VideoObservation.observed_at <= now - timedelta(minutes=2),
                        )
                        .order_by(VideoObservation.observed_at.desc())
                        .first()
                    )

                    if prev_obs:
                        delta_v = view_count - prev_obs.view_count
                        # Growth cannot be negative unless YouTube purged views
                        recent_growth = delta_v
                        prev_time = prev_obs.observed_at
                        if prev_time.tzinfo is None:
                            prev_time = prev_time.replace(tzinfo=timezone.utc)
                        delta_sec = (now - prev_time).total_seconds()
                        growth_delta_hours = round(max(delta_sec / 3600.0, 0.1), 1)
                        if delta_v > 0:
                            growth_label = f"+{delta_v:,} in {growth_delta_hours}h"
                        elif delta_v == 0:
                            growth_label = f"0 views change ({growth_delta_hours}h)"
                        else:
                            growth_label = f"{delta_v:,} audit adjustment"

                    # Store new observation snapshot
                    new_obs = VideoObservation(
                        video_id=vid,
                        channel_id=channel_id,
                        channel_title=channel_title,
                        title=item.get("title", ""),
                        niche_keyword=clean_niche,
                        view_count=view_count,
                        like_count=item.get("like_count"),
                        comment_count=item.get("comment_count"),
                        observed_at=now,
                        metadata_json={"time_period": time_period},
                    )
                    db.add(new_obs)
                    recorded_observations += 1
                except Exception as obs_err:
                    logger.warning(f"Failed to record video observation for {vid}: {obs_err}")

            # D. Channel-relative multiplier (View to subscriber ratio)
            view_sub_ratio = None
            channel_baseline_text = "Channel subscriber count hidden"
            is_small_channel = False
            if channel_subs is not None and channel_subs > 0:
                view_sub_ratio = round(view_count / channel_subs, 2)
                channel_baseline_text = f"Channel baseline: {channel_subs:,} subs ({view_sub_ratio}x view multiplier)"
                if channel_subs < 100000 and view_sub_ratio >= 1.2:
                    is_small_channel = True
            elif channel_subs == 0:
                channel_baseline_text = "New channel (0 subs)"

            # Construct processed video card payload
            processed_videos.append({
                "video_id": vid,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "channel_id": channel_id,
                "channel_title": channel_title,
                "channel_subscriber_count": channel_subs,
                "channel_avatar_url": item.get("channel_avatar_url"),
                "channel_custom_url": item.get("channel_custom_url"),
                "thumbnail_url": item.get("thumbnail_url"),
                "url": item.get("url") or f"https://www.youtube.com/watch?v={vid}",
                "published_at": pub_raw,
                "published_date_formatted": pub_raw[:10] if pub_raw else "Recent",
                "video_age_formatted": age_formatted,
                "duration_formatted": item.get("duration_formatted", ""),
                "view_count": view_count,
                "view_count_formatted": f"{view_count:,}",
                "views_per_day": views_per_day,
                "views_per_day_formatted": f"{views_per_day:,}/day",
                "like_count": item.get("like_count"),
                "like_count_formatted": f"{item['like_count']:,}" if item.get("like_count") is not None else None,
                "comment_count": item.get("comment_count"),
                "comment_count_formatted": f"{item['comment_count']:,}" if item.get("comment_count") is not None else None,
                "recent_growth": recent_growth,
                "recent_growth_formatted": growth_label,
                "growth_delta_hours": growth_delta_hours,
                "view_sub_ratio": view_sub_ratio,
                "channel_baseline_text": channel_baseline_text,
                "is_small_channel": is_small_channel,
                "tags": item.get("tags", []),
            })

        if db is not None and recorded_observations > 0:
            try:
                db.commit()
            except Exception as commit_err:
                db.rollback()
                logger.warning(f"Database commit error for observations: {commit_err}")

        # 4. Deterministic Category Ranking & Sorting
        has_momentum_records = any(v.get("recent_growth") is not None for v in processed_videos)

        if category == "most_viewed":
            processed_videos.sort(key=lambda x: x["view_count"], reverse=True)
            for idx, v in enumerate(processed_videos, 1):
                v["rank_badge"] = f"👑 #{idx} Most Viewed"
                v["rank_reason"] = f"Highest total lifetime views in {clean_niche} ({v['view_count_formatted']} views)"

        elif category == "high_average":
            processed_videos.sort(key=lambda x: x["views_per_day"], reverse=True)
            for idx, v in enumerate(processed_videos, 1):
                v["rank_badge"] = f"🚀 #{idx} Velocity"
                v["rank_reason"] = f"Pacing at {v['views_per_day_formatted']} since upload ({v['video_age_formatted']})"

        elif category == "recent":
            processed_videos.sort(key=lambda x: x["published_at"] or "", reverse=True)
            for idx, v in enumerate(processed_videos, 1):
                v["rank_badge"] = f"⚡ #{idx} Recent"
                v["rank_reason"] = f"Published {v['video_age_formatted']} ({v['published_date_formatted']})"

        elif category == "momentum":
            # Rank by measured view growth from stored observations
            if has_momentum_records:
                # Put videos with measured growth first, sorted by delta views descending
                processed_videos.sort(
                    key=lambda x: (x["recent_growth"] is not None, x["recent_growth"] or 0),
                    reverse=True,
                )
                for idx, v in enumerate(processed_videos, 1):
                    if v["recent_growth"] is not None and v["recent_growth"] > 0:
                        v["rank_badge"] = f"📈 #{idx} Gaining Momentum"
                        v["rank_reason"] = f"+{v['recent_growth']:,} views observed over last {v['growth_delta_hours']}h"
                    else:
                        v["rank_badge"] = "🎯 Baseline Tracked"
                        v["rank_reason"] = "First observation recorded today; awaiting second snapshot"
            else:
                # No two observations yet — sort by daily velocity with clear explanation
                processed_videos.sort(key=lambda x: x["views_per_day"], reverse=True)
                for v in processed_videos:
                    v["rank_badge"] = "⏱ Baseline Recorded"
                    v["rank_reason"] = "Initial snapshot captured. Refresh in 1-2 hours to observe delta view growth."

        elif category == "small_channel":
            # Rank by view-to-subscriber multiplier for channels under 100k subscribers
            def small_channel_score(v):
                ratio = v["view_sub_ratio"] or 0.0
                subs = v["channel_subscriber_count"] or 99999999
                # Prioritize high ratio from smaller channels
                if subs < 100000:
                    return ratio * 2.0
                return ratio

            processed_videos.sort(key=small_channel_score, reverse=True)
            for idx, v in enumerate(processed_videos, 1):
                ratio = v["view_sub_ratio"]
                if ratio and ratio >= 1.0:
                    v["rank_badge"] = f"🌱 #{idx} Standout ({ratio}x Subs)"
                    v["rank_reason"] = f"Outperformed subscriber baseline by {ratio}x ({v['channel_baseline_text']})"
                else:
                    v["rank_badge"] = f"🌱 #{idx} Channel Match"
                    v["rank_reason"] = f"Niche competitor ({v['channel_baseline_text']})"

        # Respect secondary sort dropdown if user picked a specific sort
        if sort_by == "views":
            processed_videos.sort(key=lambda x: x["view_count"], reverse=True)
        elif sort_by == "velocity":
            processed_videos.sort(key=lambda x: x["views_per_day"], reverse=True)
        elif sort_by == "recent":
            processed_videos.sort(key=lambda x: x["published_at"] or "", reverse=True)
        elif sort_by == "likes":
            processed_videos.sort(key=lambda x: x["like_count"] or 0, reverse=True)

        coverage_info = {
            "source": "official_youtube_api_v3",
            "videos_retrieved": len(processed_videos),
            "time_period": time_period,
            "category": category,
            "has_momentum_delta_data": has_momentum_records,
            "momentum_explanation": (
                "Verified multi-snapshot view changes detected."
                if has_momentum_records
                else "Baseline snapshot captured for all videos. Two observation snapshots at different timestamps are required by YouTube to compute delta growth without fabrication."
            ),
            "quota_status": "active",
        }

        response_payload = {
            "videos": processed_videos,
            "niche": clean_niche,
            "time_period": time_period,
            "category": category,
            "sort_by": sort_by,
            "total_count": len(processed_videos),
            "next_page_token": raw_result.get("next_page_token"),
            "last_refreshed_at": now.isoformat(),
            "is_cached": False,
            "data_coverage": coverage_info,
        }

        # 5. Persist to cache if db provided and not pagination
        if db is not None and not page_token:
            try:
                existing_cache = (
                    db.query(NicheSearchCache)
                    .filter(
                        NicheSearchCache.workspace_id == workspace_id,
                        NicheSearchCache.niche_keyword.ilike(clean_niche),
                        NicheSearchCache.time_period == time_period,
                        NicheSearchCache.category == category,
                    )
                    .first()
                )
                if existing_cache:
                    existing_cache.cached_payload = response_payload
                    existing_cache.last_refreshed_at = now
                else:
                    new_cache = NicheSearchCache(
                        workspace_id=workspace_id,
                        niche_keyword=clean_niche,
                        time_period=time_period,
                        category=category,
                        cached_payload=response_payload,
                        last_refreshed_at=now,
                    )
                    db.add(new_cache)
                db.commit()
            except Exception as cache_err:
                db.rollback()
                logger.warning(f"Failed to persist niche radar cache: {cache_err}")

        return response_payload
