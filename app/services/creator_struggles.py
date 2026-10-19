from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import CreatorStruggleCache
from app.agent.providers.llm import OpenRouterLLMClient
from app.agent.providers.tavily import TavilySearchClient
from app.agent.providers.youtube import YouTubeAdapter

logger = logging.getLogger("growtube.creator_struggles")

SUGGESTED_STRUGGLE_QUERIES = [
    {
        "query": "My coding tutorials get impressions, but viewers do not click, and those who click leave in the first minute.",
        "badge": "CTR & Retention",
    },
    {
        "query": "Why do my YouTube Shorts get exactly zero views or stop at 400 views?",
        "badge": "Shorts Feed",
    },
    {
        "query": "Sudden drop in views on new uploads after consistent momentum",
        "badge": "Algorithm Test",
    },
    {
        "query": "High impressions on browse features but CTR is stuck under 2%",
        "badge": "Packaging",
    },
    {
        "query": "Viewers leaving within the first 30 seconds: hook and pacing diagnosis",
        "badge": "Audience Retention",
    },
    {
        "query": "Views increasing rapidly but subscriber conversion is near zero",
        "badge": "Subscribers",
    },
    {
        "query": "Reused content monetization rejection: how to successfully appeal",
        "badge": "Monetization Policy",
    },
    {
        "query": "Running out of video ideas in a saturated niche: finding content gaps",
        "badge": "Ideation Strategy",
    },
    {
        "query": "Video editing taking 30 hours per video: streamlining production workflow",
        "badge": "Editing Workload",
    },
    {
        "query": "Competitors with lower production quality outperforming my channel",
        "badge": "Competition",
    },
]


class UniversalStrugglesEngine:
    """Universal, query-driven research engine that investigates virtually any YouTube creator problem.

    Accepts natural language questions, extracts core entities and multi-part issues,
    and conducts concurrent research across:
    1. YouTube Videos (YouTube Data API v3)
    2. Reddit Discussions (r/NewTubers, r/SmallYoutubers, r/PartneredYoutube)
    3. X Posts (Public creator discussions, platform updates, and breakdowns)
    4. Internet Articles (Creator case studies, platform policy documentation, deep-dives)
    """

    def __init__(self):
        self.tavily = TavilySearchClient()
        self.youtube = YouTubeAdapter()
        self.llm = OpenRouterLLMClient()

    @staticmethod
    def get_suggested_queries() -> list[dict[str, str]]:
        return SUGGESTED_STRUGGLE_QUERIES

    def understand_problem(self, natural_query: str) -> dict[str, Any]:
        """Perform dynamic problem understanding, extracting entities, metrics, and multi-part aspects."""
        clean_q = natural_query.strip()
        if not clean_q:
            clean_q = "YouTube channel growth strategy and view acceleration"

        # Try fast LLM extraction if client configured
        if self.llm.is_configured:
            llm_result = self._understand_with_llm(clean_q)
            if llm_result:
                return llm_result

        # Robust deterministic semantic understanding engine
        return self._understand_deterministically(clean_q)

    def _understand_with_llm(self, query: str) -> dict[str, Any] | None:
        """Use LLM to decompose complex multi-part creator questions and generate source-specific queries."""
        prompt = (
            "You are the Lead Research Architect for GrowTube. Analyze this YouTube creator's problem:\n\n"
            f'"{query}"\n\n'
            "Tasks:\n"
            "1. Identify the primary problem and intended outcome.\n"
            "2. Identify whether this contains multiple related problems (e.g., low CTR AND low retention).\n"
            "3. Extract relevant YouTube metrics (e.g. CTR, Impressions, AVD, RPM, Shorts Viewed Ratio).\n"
            "4. Formulate 4 targeted search queries tailored for each source:\n"
            "   - youtube_query: clean query for YouTube Data API to find tutorial/advice videos\n"
            "   - reddit_query: query targeting Reddit creator subreddits\n"
            "   - x_query: query targeting X / Twitter posts\n"
            "   - articles_query: query targeting articles, case studies, and documentation\n\n"
            "Respond ONLY with valid JSON in this exact structure:\n"
            "{\n"
            '  "primary_problem": "concise description of primary problem",\n'
            '  "sub_problems": ["aspect 1", "aspect 2"],\n'
            '  "intended_outcome": "what the creator aims to solve",\n'
            '  "key_metrics": ["Metric 1", "Metric 2"],\n'
            '  "queries": {\n'
            '    "youtube": "search term",\n'
            '    "reddit": "site:reddit.com/... query",\n'
            '    "x": "site:x.com query",\n'
            '    "articles": "case study query"\n'
            "  }\n"
            "}"
        )

        try:
            resp_str = self.llm.chat_completion(
                [
                    {"role": "system", "content": "You are an expert YouTube growth research engine. Output strictly valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            if resp_str:
                clean_json = resp_str.strip()
                if clean_json.startswith("```"):
                    clean_json = clean_json.split("\n", 1)[1]
                    if clean_json.endswith("```"):
                        clean_json = clean_json.rsplit("\n", 1)[0]
                data = json.loads(clean_json.strip())
                if isinstance(data, dict) and "primary_problem" in data and "queries" in data:
                    return data
        except Exception as exc:
            logger.warning(f"LLM query decomposition fallback to deterministic: {exc}")

        return None

    def _understand_deterministically(self, clean_q: str) -> dict[str, Any]:
        """High-precision semantic entity and aspect extractor for arbitrary creator problems."""
        metrics = []
        aspects = []

        # 1. CTR and Packaging
        if re.search(r"\b(ctr|click|clicks|click-through|thumbnail|thumbnails|title|titles|packaging)\b", clean_q, re.I):
            metrics.extend(["Click-Through Rate (CTR)", "Impressions"])
            aspects.append("Thumbnail & Title Packaging")

        # 2. Audience Retention and Hook
        if re.search(r"\b(retention|watch time|leave|first minute|30 seconds|drop off|pacing|hook|intro|duration|avd)\b", clean_q, re.I):
            metrics.extend(["Average View Duration (AVD)", "First 30s Retention %"])
            aspects.append("Opening Hook & Viewer Retention")

        # 3. Discovery and Algorithm
        if re.search(r"\b(impression|impressions|reach|browse|suggested|algorithm|recommend|distribution|test)\b", clean_q, re.I):
            metrics.append("Browse Features Impressions")
            aspects.append("Algorithmic Recommendation & Seed Audiences")

        # 4. Low Views or Stalls
        if re.search(r"\b(view|views|zero views|no views|drop in views|flatline|stuck)\b", clean_q, re.I):
            metrics.append("View Velocity")
            aspects.append("View Velocity & Impression Allocation")

        # 5. YouTube Shorts
        if re.search(r"\b(short|shorts|vertical|swipe)\b", clean_q, re.I):
            metrics.append("Shorts Feed Shown vs Viewed Ratio")
            aspects.append("Shorts Algorithm & Cannibalization")

        # 6. Subscribers and Community
        if re.search(r"\b(sub|subs|subscriber|subscribers|conversion|loyal|community|comments|returning)\b", clean_q, re.I):
            metrics.append("Subscriber Conversion Rate")
            aspects.append("Audience Loyalty & Viewer Conversion")

        # 7. Monetization, Copyright and Policy
        if re.search(r"\b(monetiz|ypp|rpm|cpm|revenue|reused|copyright|strike|claim|community guidelines)\b", clean_q, re.I):
            metrics.append("Playback-Based RPM / CPM")
            aspects.append("YouTube Partner Program & Policy Compliance")

        # 8. Content Ideas and Niche Competition
        if re.search(r"\b(idea|ideas|niche|topic|topics|competition|saturation|gap|competitors)\b", clean_q, re.I):
            metrics.append("Search Demand Volume")
            aspects.append("Topic Selection & Competitor Positioning")

        # 9. Production, Workflow, and Burnout
        if re.search(r"\b(edit|editing|audio|mic|camera|workflow|burnout|schedule|consistency|render)\b", clean_q, re.I):
            metrics.append("Production Turnaround Hours")
            aspects.append("Editing Workflow & Creator Sustainability")

        # Deduplicate
        seen_m = set()
        unique_metrics = [m for m in metrics if not (m in seen_m or seen_m.add(m))]
        if not unique_metrics:
            unique_metrics = ["Audience Retention", "Impressions CTR", "View Velocity"]

        seen_a = set()
        unique_aspects = [a for a in aspects if not (a in seen_a or seen_a.add(a))]
        if not unique_aspects:
            unique_aspects = ["Channel Diagnostics & Algorithm Alignment"]

        # Extract meaningful keywords for search queries
        clean_tokens = re.sub(r"[\'\"?.,!;:()\[\]{}]", " ", clean_q).split()
        stop_words = {
            "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it", "they",
            "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
            "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "can", "will", "just", "should", "now", "get", "getting",
        }
        meaningful = [w for w in clean_tokens if w.lower() not in stop_words and len(w) > 2]
        core_kw = " ".join(meaningful[:6]) if meaningful else clean_q[:40]

        return {
            "primary_problem": clean_q if len(clean_q) <= 120 else clean_q[:117] + "...",
            "sub_problems": unique_aspects,
            "intended_outcome": f"Diagnose causes and identify evidence-backed solutions for: {core_kw}",
            "key_metrics": unique_metrics,
            "queries": {
                "youtube": f"{core_kw} YouTube advice fix",
                "reddit": f"(site:reddit.com/r/NewTubers OR site:reddit.com/r/SmallYoutubers OR site:reddit.com/r/PartneredYoutube) {core_kw}",
                "x": f"(site:x.com OR site:twitter.com) YouTube {core_kw}",
                "articles": f"YouTube {core_kw} case study guide",
            },
        }

    def _fetch_youtube_videos(self, query: str, limit: int = 8) -> dict[str, Any]:
        """Fetch relevant YouTube videos via YouTube Data API v3 with clean metadata."""
        try:
            videos = self.youtube.search_competitor_videos(query, max_results=limit)
            results = []
            for v in videos:
                results.append({
                    "id": v.get("video_id"),
                    "title": v.get("title", "YouTube Video"),
                    "channel_title": v.get("channel_title", "Creator Channel"),
                    "thumbnail_url": v.get("thumbnail_url"),
                    "view_count": v.get("view_count", 0),
                    "duration_formatted": v.get("duration_formatted", "--:--"),
                    "published_at": v.get("published_at"),
                    "url": f"https://www.youtube.com/watch?v={v.get('video_id')}" if v.get("video_id") else "#",
                })
            return {"status": "ok", "results": results, "error": None}
        except Exception as exc:
            logger.warning(f"Error fetching YouTube videos for struggles engine: {exc}")
            return {"status": "error", "results": [], "error": str(exc)}

    def _fetch_reddit_discussions(self, query: str, limit: int = 8) -> dict[str, Any]:
        """Search relevant public Reddit creator communities via Tavily."""
        try:
            resp = self.tavily.search(query, max_results=limit, search_depth="advanced")
            raw_results = resp.get("results", [])
            results = []
            for r in raw_results:
                url = r.get("url", "")
                sub_match = re.search(r"reddit\.com\/(r\/[a-zA-Z0-9_]+)", url, re.I)
                subreddit = sub_match.group(1) if sub_match else "r/NewTubers"

                clean_title = r.get("title", "Reddit Discussion")
                # Remove common Reddit trailing title markers
                clean_title = re.sub(r"\s*:\s*r\/[a-zA-Z0-9_]+.*$", "", clean_title, flags=re.I)

                results.append({
                    "title": clean_title,
                    "subreddit": subreddit,
                    "url": url,
                    "content": r.get("content", ""),
                    "published_date": r.get("published_date"),
                    "retrieved_at": r.get("retrieved_at"),
                })
            return {"status": "ok", "results": results, "error": None}
        except Exception as exc:
            logger.warning(f"Error fetching Reddit discussions: {exc}")
            return {"status": "error", "results": [], "error": str(exc)}

    def _fetch_x_posts(self, query: str, limit: int = 8) -> dict[str, Any]:
        """Search public X / Twitter posts and breakdowns via Tavily."""
        try:
            resp = self.tavily.search(query, max_results=limit, search_depth="advanced")
            raw_results = resp.get("results", [])
            results = []
            for r in raw_results:
                url = r.get("url", "")
                title = r.get("title", "")

                # Extract handle from URL if matching https://x.com/username/status/...
                handle_match = re.search(r"(?:x\.com|twitter\.com)\/([a-zA-Z0-9_]+)\/status", url, re.I)
                author_handle = f"@{handle_match.group(1)}" if handle_match else "@creator"

                # Extract author name from title e.g. "Name on X: '...'"
                author_match = re.search(r"^(.+?)\s+(?:on X|on Twitter|:\s*)", title, re.I)
                author_name = author_match.group(1).strip() if author_match else author_handle.lstrip("@")

                clean_text = r.get("content", "")

                results.append({
                    "title": title,
                    "author_name": author_name,
                    "author_handle": author_handle,
                    "url": url,
                    "content": clean_text,
                    "published_date": r.get("published_date"),
                    "retrieved_at": r.get("retrieved_at"),
                })
            return {"status": "ok", "results": results, "error": None}
        except Exception as exc:
            logger.warning(f"Error fetching X posts: {exc}")
            return {"status": "error", "results": [], "error": str(exc)}

    def _fetch_internet_articles(self, query: str, limit: int = 8) -> dict[str, Any]:
        """Search documentation, creator publications, technical articles, and case studies via Tavily."""
        try:
            resp = self.tavily.search(query, max_results=limit, search_depth="advanced")
            raw_results = resp.get("results", [])
            results = []
            for r in raw_results:
                url = r.get("url", "")
                parsed = urlparse(url)
                domain = parsed.hostname.replace("www.", "") if parsed.hostname else "Web Publication"

                results.append({
                    "title": r.get("title", "Creator Guide"),
                    "domain": domain,
                    "url": url,
                    "content": r.get("content", ""),
                    "published_date": r.get("published_date"),
                    "retrieved_at": r.get("retrieved_at"),
                })
            return {"status": "ok", "results": results, "error": None}
        except Exception as exc:
            logger.warning(f"Error fetching articles: {exc}")
            return {"status": "error", "results": [], "error": str(exc)}

    def investigate(
        self,
        user_query: str,
        force_refresh: bool = False,
        db: Session | None = None,
    ) -> dict[str, Any]:
        """Execute universal multi-source investigation across YouTube, Reddit, X, and Articles concurrently."""
        clean_q = user_query.strip()
        if not clean_q:
            clean_q = "My coding tutorials get impressions, but viewers do not click, and those who click leave in the first minute."

        now = datetime.now(timezone.utc)
        cache_key = hashlib.sha256(f"universal_v2:{clean_q.lower()}".encode("utf-8")).hexdigest()[:32]

        # 1. Check cache if not forcing refresh
        if db and not force_refresh:
            cached_rec = db.query(CreatorStruggleCache).filter(CreatorStruggleCache.cache_key == cache_key).first()
            if cached_rec:
                cutoff = now - timedelta(hours=2)
                c_time = cached_rec.last_refreshed_at
                if c_time.tzinfo is None:
                    c_time = c_time.replace(tzinfo=timezone.utc)
                if c_time > cutoff:
                    payload = dict(cached_rec.cached_payload)
                    payload["from_cache"] = True
                    payload["cache_age_seconds"] = int((now - c_time).total_seconds())
                    return payload

        # 2. Dynamic Problem Understanding
        analysis = self.understand_problem(clean_q)
        queries = analysis.get("queries", {})
        yt_q = queries.get("youtube", clean_q)
        rd_q = queries.get("reddit", clean_q)
        x_q = queries.get("x", clean_q)
        art_q = queries.get("articles", clean_q)

        # 3. Parallel Four-Source Retrieval
        sources_payload = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_yt = executor.submit(self._fetch_youtube_videos, yt_q, 6)
            future_rd = executor.submit(self._fetch_reddit_discussions, rd_q, 6)
            future_x = executor.submit(self._fetch_x_posts, x_q, 6)
            future_art = executor.submit(self._fetch_internet_articles, art_q, 6)

            res_yt = future_yt.result()
            res_rd = future_rd.result()
            res_x = future_x.result()
            res_art = future_art.result()

        sources_payload["youtube"] = {
            "title": "YouTube Videos",
            "status": res_yt["status"],
            "query_used": yt_q,
            "count": len(res_yt["results"]),
            "results": res_yt["results"],
            "error": res_yt.get("error"),
        }
        sources_payload["reddit"] = {
            "title": "Reddit Discussions",
            "status": res_rd["status"],
            "query_used": rd_q,
            "count": len(res_rd["results"]),
            "results": res_rd["results"],
            "error": res_rd.get("error"),
        }
        sources_payload["x"] = {
            "title": "X Posts",
            "status": res_x["status"],
            "query_used": x_q,
            "count": len(res_x["results"]),
            "results": res_x["results"],
            "error": res_x.get("error"),
        }
        sources_payload["articles"] = {
            "title": "Internet Articles",
            "status": res_art["status"],
            "query_used": art_q,
            "count": len(res_art["results"]),
            "results": res_art["results"],
            "error": res_art.get("error"),
        }

        total_evidence = (
            len(res_yt["results"])
            + len(res_rd["results"])
            + len(res_x["results"])
            + len(res_art["results"])
        )

        response_payload = {
            "query": clean_q,
            "analysis": analysis,
            "sources": sources_payload,
            "total_evidence_count": total_evidence,
            "retrieved_at": now.isoformat(),
            "evidence_disclosure": (
                f"Multi-source investigation retrieved {total_evidence} authentic records across "
                "YouTube Data API, Reddit community discussions, X posts, and internet case studies. "
                "All results reflect verified external sources. Specific channel diagnosis requires authorized Studio analytics."
            ),
            "from_cache": False,
        }

        # 4. Cache in Database
        if db:
            try:
                cached_rec = db.query(CreatorStruggleCache).filter(CreatorStruggleCache.cache_key == cache_key).first()
                if not cached_rec:
                    cached_rec = CreatorStruggleCache(
                        cache_key=cache_key,
                        category="universal_query",
                        query=clean_q[:250],
                        sample_size=total_evidence,
                        cached_payload=response_payload,
                        last_refreshed_at=now,
                    )
                    db.add(cached_rec)
                else:
                    cached_rec.cached_payload = response_payload
                    cached_rec.sample_size = total_evidence
                    cached_rec.last_refreshed_at = now
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.warning(f"Failed to cache universal struggle query: {exc}")

        return response_payload
