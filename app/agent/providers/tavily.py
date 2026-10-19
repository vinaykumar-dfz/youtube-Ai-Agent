from datetime import datetime, timezone
import logging
import os
from typing import Any
import httpx

from app.core.config import settings

logger = logging.getLogger("growtube.tavily")


class TavilySearchClient:
    """Production client for Tavily Web & Community Search API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = (
            api_key
            or settings.TAVILY_API_KEY
            or os.environ.get("TAVILY_API_KEY", "")
        ).strip()
        self.base_url = "https://api.tavily.com"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("tvly-"))

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        include_answer: bool = False,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        """Execute web search across creator communities and authoritative case studies.

        Returns structured results with provenance timestamps, platform identification,
        and clean text snippets.
        """
        clean_query = query.strip()
        if not clean_query:
            return {"query": "", "results": [], "total_results": 0, "error": None}

        if not self.is_configured:
            logger.warning("Tavily API key is not configured. Web discovery cannot execute.")
            return {
                "query": clean_query,
                "results": [],
                "total_results": 0,
                "error": "Tavily API key not configured. Set TAVILY_API_KEY in environment.",
            }

        payload = {
            "api_key": self.api_key,
            "query": clean_query,
            "search_depth": search_depth,
            "max_results": min(max(max_results, 1), 20),
            "include_answer": include_answer,
            "include_raw_content": False,
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if resp.status_code == 200:
                    data = resp.json()
                    raw_results = data.get("results", [])
                    now_iso = datetime.now(timezone.utc).isoformat()

                    formatted_results = []
                    seen_urls = set()

                    for item in raw_results:
                        url = item.get("url", "").strip()
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)

                        platform = "reddit" if "reddit.com" in url.lower() else "web"
                        formatted_results.append({
                            "title": item.get("title", "Untitled Discussion").strip(),
                            "url": url,
                            "content": (item.get("content") or "").strip(),
                            "score": float(item.get("score") or 0.0),
                            "published_date": item.get("published_date"),
                            "retrieved_at": now_iso,
                            "platform": platform,
                        })

                    return {
                        "query": clean_query,
                        "results": formatted_results,
                        "total_results": len(formatted_results),
                        "answer": data.get("answer"),
                        "error": None,
                    }
                else:
                    err_msg = f"Tavily API returned HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning(err_msg)
                    return {
                        "query": clean_query,
                        "results": [],
                        "total_results": 0,
                        "error": err_msg,
                    }

        except httpx.TimeoutException:
            logger.warning("Tavily search request timed out for query: %s", clean_query)
            return {
                "query": clean_query,
                "results": [],
                "total_results": 0,
                "error": "Tavily search request timed out. Please try again.",
            }
        except Exception as exc:
            logger.error("Unexpected error in Tavily search: %s", exc)
            return {
                "query": clean_query,
                "results": [],
                "total_results": 0,
                "error": str(exc),
            }
