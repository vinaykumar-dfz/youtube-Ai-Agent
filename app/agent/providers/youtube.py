import re
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Any
import httpx

from app.core.config import settings
from app.core.errors import (
    InvalidInputException,
    YouTubeAPIException,
    YouTubeQuotaExceededException,
    YouTubeResourceNotFoundException,
)
from app.core.logging import logger


class YouTubeAdapter:
    """Production-grade adapter for the official Google YouTube Data API v3."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.YOUTUBE_API_KEY
        self.base_url = "https://www.googleapis.com/youtube/v3"

    def parse_youtube_target(self, url_or_id: str) -> dict[str, str]:
        """Classify and extract target identity from any valid YouTube URL or raw identifier."""
        target = url_or_id.strip()
        if not target:
            raise InvalidInputException("YouTube URL or target cannot be empty.")

        # 1. Video URL: youtube.com/watch?v=...
        watch_match = re.search(r"(?:youtube\.com\/watch\?(?:.*&)?v=)([a-zA-Z0-9_-]{11})", target)
        if watch_match:
            return {"type": "video", "id": watch_match.group(1)}

        # 2. Video URL: youtu.be/VIDEO_ID
        short_match = re.search(r"youtu\.be\/([a-zA-Z0-9_-]{11})", target)
        if short_match:
            return {"type": "video", "id": short_match.group(1)}

        # 3. Shorts URL: youtube.com/shorts/VIDEO_ID
        shorts_match = re.search(r"youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})", target)
        if shorts_match:
            return {"type": "video", "id": shorts_match.group(1)}

        # 4. Embed URL: youtube.com/embed/VIDEO_ID
        embed_match = re.search(r"youtube\.com\/embed\/([a-zA-Z0-9_-]{11})", target)
        if embed_match:
            return {"type": "video", "id": embed_match.group(1)}

        # 5. Channel ID URL: youtube.com/channel/UC... or raw UC...
        channel_id_match = re.search(r"(?:youtube\.com\/channel\/|^)(UC[a-zA-Z0-9_-]{22})", target)
        if channel_id_match:
            return {"type": "channel_id", "id": channel_id_match.group(1)}

        # 6. Handle URL: youtube.com/@handle or raw @handle
        handle_match = re.search(r"@([a-zA-Z0-9_.-]+)", target)
        if handle_match:
            return {"type": "handle", "id": f"@{handle_match.group(1)}"}

        # 7. Custom URL: youtube.com/c/NAME or youtube.com/user/NAME
        if "youtube.com/" in target:
            parsed = urllib.parse.urlparse(target)
            path_parts = [p for p in parsed.path.split("/") if p]
            if len(path_parts) >= 2 and path_parts[0] in ("c", "user"):
                return {"type": "custom", "id": path_parts[1]}
            elif path_parts:
                candidate = path_parts[-1]
                if candidate.startswith("@"):
                    return {"type": "handle", "id": candidate}
                return {"type": "custom", "id": candidate}

        # 8. Raw 11-char string might be a video ID, 24-char UC string is channel ID
        if re.match(r"^UC[a-zA-Z0-9_-]{22}$", target):
            return {"type": "channel_id", "id": target}
        if re.match(r"^[a-zA-Z0-9_-]{11}$", target):
            return {"type": "video", "id": target}

        # Fallback treat as handle search query
        clean_handle = target.lstrip("@").strip()
        return {"type": "handle", "id": f"@{clean_handle}"}

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute authenticated GET request to YouTube Data API v3 with robust error handling."""
        if not self.api_key:
            raise YouTubeAPIException(
                "YouTube Data API v3 key is not configured. "
                "Please add a valid YOUTUBE_API_KEY to your .env file."
            )

        full_params = {"key": self.api_key, **params}
        url = f"{self.base_url}/{endpoint}"

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, params=full_params)
        except httpx.RequestError as exc:
            logger.error(f"Network error contacting YouTube API: {exc}")
            raise YouTubeAPIException(f"Network connection to YouTube API failed: {exc}")

        if resp.status_code == 403:
            err_json = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            reasons = [e.get("reason", "") for e in err_json.get("error", {}).get("errors", [])]
            if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
                raise YouTubeQuotaExceededException()
            raise YouTubeAPIException(f"YouTube API access denied (403): {resp.text}")

        if resp.status_code == 404:
            raise YouTubeResourceNotFoundException(params.get("id", "endpoint"))

        if resp.status_code != 200:
            raise YouTubeAPIException(
                f"YouTube Data API request to '{endpoint}' failed with status {resp.status_code}: {resp.text}"
            )

        return resp.json()

    def fetch_video_details(self, video_id: str) -> dict[str, Any]:
        """Retrieve real metadata and statistics for a specific YouTube video."""
        data = self._get("videos", {
            "part": "snippet,contentDetails,statistics,status",
            "id": video_id,
        })
        items = data.get("items", [])
        if not items:
            raise YouTubeResourceNotFoundException(f"Video '{video_id}'")

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})
        status_info = item.get("status", {})
        duration_raw = content.get("duration", "")

        return {
            "video_id": item.get("id"),
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "published_at": snippet.get("publishedAt", ""),
            "channel_id": snippet.get("channelId", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "tags": snippet.get("tags", []),
            "thumbnail_url": (
                snippet.get("thumbnails", {}).get("maxres", {}).get("url")
                or snippet.get("thumbnails", {}).get("high", {}).get("url")
                or snippet.get("thumbnails", {}).get("default", {}).get("url")
            ),
            "duration": duration_raw,
            "duration_formatted": format_iso_duration(duration_raw),
            "privacy_status": status_info.get("privacyStatus", "public").capitalize(),
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)) if "likeCount" in stats else None,
            "comment_count": int(stats.get("commentCount", 0)) if "commentCount" in stats else None,
        }

    def get_uploads_playlist_id(self, channel_id: str) -> str | None:
        """Resolve or derive authoritative uploads playlist ID from channel ID."""
        if not channel_id:
            return None
        try:
            data = self._get("channels", {"part": "contentDetails", "id": channel_id})
            items = data.get("items", [])
            if items:
                uploads = items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
                if uploads:
                    return uploads
        except Exception as exc:
            logger.warning(f"Failed to query uploads playlist ID for channel '{channel_id}': {exc}")
        # Fallback rule of thumb for standard YouTube channel IDs: 'UC...' -> 'UU...'
        if channel_id.startswith("UC") and len(channel_id) == 24:
            return "UU" + channel_id[2:]
        return None

    def fetch_channel_video_inventory(
        self,
        uploads_playlist_id: str,
        max_results: int = 500,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch real channel videos with exact statistics, duration, and pagination metadata.

        Follows nextPageToken until all uploads are retrieved or safe operational limits
        are reached (up to max_results or 10 pages / 500 uploads). Preserves unique video IDs,
        batches statistics/details queries in chunks of 50, and reports actual retrieval completeness.
        """
        if not uploads_playlist_id:
            return {
                "videos": [],
                "next_page_token": None,
                "total_results": 0,
                "inspected_count": 0,
                "playlist_items_count": 0,
                "metadata_items_count": 0,
                "is_complete": False,
            }

        try:
            PAGE_SIZE = 50
            MAX_PAGES = 10  # Safe operational limit: up to 500 videos

            all_video_ids: list[str] = []
            seen_video_ids: set[str] = set()
            raw_playlist_items: dict[str, dict[str, Any]] = {}
            current_token: str | None = page_token
            page_count = 0
            total_results = 0

            # If a specific page_token was explicitly provided, only fetch that single page
            single_page_mode = page_token is not None

            while True:
                fetch_limit = min(PAGE_SIZE, max_results - len(all_video_ids)) if max_results > len(all_video_ids) else PAGE_SIZE
                params: dict[str, Any] = {
                    "part": "snippet,contentDetails",
                    "playlistId": uploads_playlist_id,
                    "maxResults": min(max(fetch_limit, 1), PAGE_SIZE),
                }
                if current_token:
                    params["pageToken"] = current_token

                playlist_data = self._get("playlistItems", params)
                page_count += 1

                if total_results == 0:
                    total_results = playlist_data.get("pageInfo", {}).get("totalResults", 0)

                items = playlist_data.get("items", [])
                for item in items:
                    vid = item.get("contentDetails", {}).get("videoId")
                    if vid and vid not in seen_video_ids:
                        seen_video_ids.add(vid)
                        all_video_ids.append(vid)
                        raw_playlist_items[vid] = item

                current_token = playlist_data.get("nextPageToken")

                if single_page_mode:
                    break
                if not current_token:
                    break
                if len(all_video_ids) >= max_results:
                    break
                if page_count >= MAX_PAGES:
                    logger.info(
                        "Reached safe max page limit (%d pages, %d videos) for playlist %s",
                        page_count,
                        len(all_video_ids),
                        uploads_playlist_id,
                    )
                    break

            if total_results == 0:
                total_results = len(all_video_ids)

            if not all_video_ids:
                return {
                    "videos": [],
                    "next_page_token": current_token,
                    "total_results": total_results,
                    "inspected_count": 0,
                    "playlist_items_count": 0,
                    "metadata_items_count": 0,
                    "is_complete": current_token is None,
                }

            # Batch fetch stats & contentDetails in chunks of 50
            stat_items: dict[str, dict[str, Any]] = {}
            for i in range(0, len(all_video_ids), 50):
                chunk_ids = all_video_ids[i:i + 50]
                try:
                    stats_data = self._get("videos", {
                        "part": "snippet,contentDetails,statistics",
                        "id": ",".join(chunk_ids),
                    })
                    for v in stats_data.get("items", []):
                        stat_items[v["id"]] = v
                except Exception as exc:
                    logger.warning(f"Error fetching video metadata batch for {len(chunk_ids)} videos: {exc}")

            videos = []
            for vid in all_video_ids:
                item = raw_playlist_items.get(vid, {})
                v_meta = stat_items.get(vid, {})
                s = v_meta.get("snippet", item.get("snippet", {}))
                st = v_meta.get("statistics", {})
                cd = v_meta.get("contentDetails", {})

                raw_views = st.get("viewCount")
                raw_likes = st.get("likeCount")
                raw_comments = st.get("commentCount")
                duration_raw = cd.get("duration", "")

                videos.append({
                    "video_id": vid,
                    "title": s.get("title", ""),
                    "description": s.get("description", ""),
                    "published_at": s.get("publishedAt", ""),
                    "view_count": int(raw_views) if raw_views is not None else 0,
                    "like_count": int(raw_likes) if raw_likes is not None else None,
                    "comment_count": int(raw_comments) if raw_comments is not None else None,
                    "thumbnail_url": (
                        s.get("thumbnails", {}).get("high", {}).get("url")
                        or s.get("thumbnails", {}).get("medium", {}).get("url")
                        or s.get("thumbnails", {}).get("default", {}).get("url")
                    ),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "duration": format_iso_duration(duration_raw),
                    "tags": s.get("tags", []),
                })

            playlist_count = len(all_video_ids)
            metadata_count = len(stat_items)
            is_complete = (current_token is None) and (metadata_count >= playlist_count)

            return {
                "videos": videos,
                "next_page_token": current_token,
                "total_results": total_results,
                "inspected_count": len(videos),
                "playlist_items_count": playlist_count,
                "metadata_items_count": metadata_count,
                "is_complete": is_complete,
                "uploads_playlist_id": uploads_playlist_id,
            }
        except Exception as exc:
            logger.warning(f"Error fetching channel video inventory from playlist '{uploads_playlist_id}': {exc}")
            return {
                "videos": [],
                "next_page_token": None,
                "total_results": 0,
                "inspected_count": 0,
                "playlist_items_count": 0,
                "metadata_items_count": 0,
                "is_complete": False,
                "error": str(exc),
            }

    def fetch_recent_channel_videos(self, uploads_playlist_id: str, max_results: int = 15) -> list[dict[str, Any]]:
        """Fetch real recent public videos uploaded to a channel via its uploads playlist."""
        inv = self.fetch_channel_video_inventory(uploads_playlist_id, max_results=max_results)
        return inv.get("videos", [])[:max_results]

    def fetch_channel_data(self, target_info: dict[str, str]) -> dict[str, Any]:
        """Fetch complete real channel metadata and recent uploads without fabricated fallbacks."""
        target_type = target_info.get("type", "handle")
        target_id = target_info.get("id", "")

        analyzed_video = None

        # If user supplied a video URL, first resolve the video to identify its channel
        if target_type == "video":
            analyzed_video = self.fetch_video_details(target_id)
            target_type = "channel_id"
            target_id = analyzed_video["channel_id"]

        params: dict[str, Any] = {"part": "snippet,statistics,contentDetails"}

        if target_type == "channel_id":
            params["id"] = target_id
        elif target_type == "handle":
            params["forHandle"] = target_id.lstrip("@")
        elif target_type == "custom":
            # Try forUsername first, fallback to search if not found
            try:
                temp_params = {**params, "forUsername": target_id}
                data = self._get("channels", temp_params)
                if not data.get("items"):
                    # Fallback search for channel ID
                    search_res = self._get("search", {"part": "snippet", "type": "channel", "q": target_id, "maxResults": 1})
                    search_items = search_res.get("items", [])
                    if search_items:
                        params["id"] = search_items[0]["id"]["channelId"]
                    else:
                        params["forHandle"] = target_id
                else:
                    params = temp_params
            except Exception:
                params["forHandle"] = target_id

        data = self._get("channels", params)
        items = data.get("items", [])

        # If handle lookup failed, try searching for the handle name
        if not items and "forHandle" in params:
            handle_name = params["forHandle"]
            search_res = self._get("search", {"part": "snippet", "type": "channel", "q": handle_name, "maxResults": 1})
            search_items = search_res.get("items", [])
            if search_items:
                channel_id_found = search_items[0].get("id", {}).get("channelId")
                if channel_id_found:
                    data = self._get("channels", {"part": "snippet,statistics,contentDetails", "id": channel_id_found})
                    items = data.get("items", [])

        if not items:
            raise YouTubeResourceNotFoundException(f"Channel for target '{target_id}'")

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content_details = item.get("contentDetails", {})
        uploads_playlist_id = content_details.get("relatedPlaylists", {}).get("uploads")

        # Fetch actual recent uploads using uploads playlist
        recent_videos = self.fetch_recent_channel_videos(uploads_playlist_id, max_results=15) if uploads_playlist_id else []

        # If a specific video was submitted, ensure it is included or highlighted in analyzed_video
        raw_subs = stats.get("subscriberCount")
        subscriber_count = int(raw_subs) if raw_subs is not None else None

        return {
            "channel_id": item.get("id"),
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "subscriber_count": subscriber_count,  # None if hidden by creator
            "video_count": int(stats.get("videoCount", 0)),
            "view_count": int(stats.get("viewCount", 0)),
            "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url") or snippet.get("thumbnails", {}).get("default", {}).get("url"),
            "custom_url": snippet.get("customUrl", f"@{snippet.get('title', '').replace(' ', '').lower()}"),
            "published_at": snippet.get("publishedAt", ""),
            "uploads_playlist_id": uploads_playlist_id,
            "recent_videos": recent_videos,
            "analyzed_video": analyzed_video,
            "source": "official_youtube_api_v3",
        }

    def search_competitor_videos(self, query: str, max_results: int = 15) -> list[dict[str, Any]]:
        """Search and retrieve real candidate competitor videos for a specific topic query."""
        if not self.api_key or not query.strip():
            return []

        try:
            # 1. Search for video items matching query
            search_data = self._get("search", {
                "part": "snippet",
                "type": "video",
                "q": query.strip(),
                "maxResults": min(max_results, 50),
            })
            search_items = search_data.get("items", [])
            video_ids = [
                item["id"]["videoId"]
                for item in search_items
                if item.get("id", {}).get("videoId")
            ]
            if not video_ids:
                return []

            # 2. Batch fetch full video metadata (contentDetails, statistics)
            videos_data = self._get("videos", {
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(video_ids),
            })
            items = videos_data.get("items", [])

            results = []
            for item in items:
                v_id = item.get("id")
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                content = item.get("contentDetails", {})

                raw_views = stats.get("viewCount")
                raw_likes = stats.get("likeCount")
                raw_comments = stats.get("commentCount")

                thumb = (
                    snippet.get("thumbnails", {}).get("high", {}).get("url")
                    or snippet.get("thumbnails", {}).get("medium", {}).get("url")
                    or snippet.get("thumbnails", {}).get("default", {}).get("url")
                )

                duration_raw = content.get("duration", "")
                results.append({
                    "video_id": v_id,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "thumbnail_url": thumb,
                    "duration_raw": duration_raw,
                    "duration_formatted": format_iso_duration(duration_raw),
                    "view_count": int(raw_views) if raw_views is not None else 0,
                    "like_count": int(raw_likes) if raw_likes is not None else None,
                    "comment_count": int(raw_comments) if raw_comments is not None else None,
                    "tags": snippet.get("tags", []),
                })

            return results
        except Exception as exc:
            logger.warning(f"Error searching competitor videos for query '{query}': {exc}")
            return []

    def search_niche_videos(
        self,
        query: str,
        time_period: str = "all",
        sort_order: str = "relevance",
        max_results: int = 24,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """Search and batch-hydrate real YouTube videos and channels for Niche Radar."""
        clean_query = query.strip()
        if not clean_query:
            return {"videos": [], "next_page_token": None, "total_results": 0, "query": "", "data_coverage": {}}

        published_after = None
        now = datetime.now(timezone.utc)
        if time_period == "7d":
            published_after = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif time_period == "30d":
            published_after = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif time_period == "90d":
            published_after = (now - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")

        order_map = {
            "viewCount": "viewCount",
            "date": "date",
            "relevance": "relevance",
            "rating": "rating",
        }
        api_order = order_map.get(sort_order, "relevance")

        search_params: dict[str, Any] = {
            "part": "snippet",
            "type": "video",
            "q": clean_query,
            "maxResults": min(max_results, 50),
            "order": api_order,
        }
        if published_after:
            search_params["publishedAfter"] = published_after
        if page_token:
            search_params["pageToken"] = page_token

        search_data = self._get("search", search_params)
        search_items = search_data.get("items", [])
        next_page_token = search_data.get("nextPageToken")
        total_results = search_data.get("pageInfo", {}).get("totalResults", len(search_items))

        video_ids = []
        for it in search_items:
            vid = it.get("id", {}).get("videoId")
            if vid and vid not in video_ids:
                video_ids.append(vid)

        if not video_ids:
            return {
                "videos": [],
                "next_page_token": next_page_token,
                "total_results": total_results,
                "query": clean_query,
                "time_period": time_period,
                "retrieved_at": now.isoformat(),
                "data_coverage": {"videos_retrieved": 0, "source": "official_youtube_api_v3"},
            }

        # Batch fetch full video stats & contentDetails
        videos_data = self._get("videos", {
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(video_ids),
        })
        video_items_dict = {it["id"]: it for it in videos_data.get("items", [])}

        # Extract unique channel IDs and batch fetch channel statistics
        channel_ids = list(dict.fromkeys([
            it.get("snippet", {}).get("channelId")
            for it in video_items_dict.values()
            if it.get("snippet", {}).get("channelId")
        ]))

        channels_dict = {}
        if channel_ids:
            try:
                ch_data = self._get("channels", {
                    "part": "snippet,statistics",
                    "id": ",".join(channel_ids[:50]),
                })
                channels_dict = {ch["id"]: ch for ch in ch_data.get("items", [])}
            except Exception as ch_err:
                logger.warning(f"Failed to batch query channels for niche search: {ch_err}")

        results = []
        for vid in video_ids:
            item = video_items_dict.get(vid)
            if not item:
                continue

            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            channel_id = snippet.get("channelId", "")
            ch_info = channels_dict.get(channel_id, {})
            ch_stats = ch_info.get("statistics", {})
            ch_snippet = ch_info.get("snippet", {})

            raw_subs = ch_stats.get("subscriberCount")
            ch_subs = int(raw_subs) if raw_subs is not None else None
            ch_avatar = (
                ch_snippet.get("thumbnails", {}).get("default", {}).get("url")
                or ch_snippet.get("thumbnails", {}).get("high", {}).get("url")
            )

            raw_views = stats.get("viewCount")
            raw_likes = stats.get("likeCount")
            raw_comments = stats.get("commentCount")

            thumb = (
                snippet.get("thumbnails", {}).get("maxres", {}).get("url")
                or snippet.get("thumbnails", {}).get("high", {}).get("url")
                or snippet.get("thumbnails", {}).get("medium", {}).get("url")
                or snippet.get("thumbnails", {}).get("default", {}).get("url")
            )

            duration_raw = content.get("duration", "")
            results.append({
                "video_id": vid,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_id": channel_id,
                "channel_title": snippet.get("channelTitle", ""),
                "channel_subscriber_count": ch_subs,
                "channel_avatar_url": ch_avatar,
                "channel_custom_url": ch_snippet.get("customUrl", ""),
                "published_at": snippet.get("publishedAt", ""),
                "thumbnail_url": thumb,
                "duration_raw": duration_raw,
                "duration_formatted": format_iso_duration(duration_raw),
                "view_count": int(raw_views) if raw_views is not None else 0,
                "like_count": int(raw_likes) if raw_likes is not None else None,
                "comment_count": int(raw_comments) if raw_comments is not None else None,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "tags": snippet.get("tags", []),
            })

        return {
            "videos": results,
            "next_page_token": next_page_token,
            "total_results": total_results,
            "query": clean_query,
            "time_period": time_period,
            "retrieved_at": now.isoformat(),
            "data_coverage": {
                "source": "official_youtube_api_v3",
                "videos_retrieved": len(results),
                "channels_hydrated": len(channels_dict),
                "has_exact_stats": True,
            },
        }


def format_iso_duration(iso_duration: str) -> str:
    """Convert ISO 8601 duration (e.g., PT1H2M30S, PT14M32S) to HH:MM:SS or MM:SS."""
    if not iso_duration:
        return "Unknown"
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
    if not match:
        return iso_duration
    hours, minutes, seconds = match.groups()
    h = int(hours) if hours else 0
    m = int(minutes) if minutes else 0
    s = int(seconds) if seconds else 0
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
    return f"{m}:{s:02d}"
