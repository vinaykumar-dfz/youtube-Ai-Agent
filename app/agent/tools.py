import os
import time
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.services.competitor_analyzer import CompetitorAnalyzer
from app.agent.memory import CreatorMemoryService
from app.services.ingestion_schemas import AnalysisSubmission
from app.services.data_ingestion import DataIngestionService
from app.agent.evidence import EvidenceService
from app.services.opportunity_engine import OpportunityEngineService


# -------------------------------------------------------------
# Tool Context
# -------------------------------------------------------------
class ToolContext:
    def __init__(
        self,
        db: Session,
        workspace_id: str,
        job_id: str,
        user_id: str | None = None,
        active_channel_id: str | None = None,
        active_channel_name: str | None = None,
        active_channel_context: dict[str, Any] | None = None,
        cached_inventory: list[dict[str, Any]] | None = None,
        inventory_is_complete: bool = False,
        access_token: str | None = None,
    ):
        self.db = db
        self.workspace_id = workspace_id
        self.job_id = job_id
        self.user_id = user_id
        self.active_channel_id = active_channel_id
        self.active_channel_name = active_channel_name
        self.active_channel_context = active_channel_context
        self.cached_inventory = cached_inventory
        self.inventory_is_complete = inventory_is_complete
        self.access_token = access_token


# -------------------------------------------------------------
# Tool Schemas
# -------------------------------------------------------------

# 1. Resolve Target
class ResolveTargetInput(BaseModel):
    target: str = Field(description="YouTube video URL, channel URL, handle, or local upload reference")


class ResolveTargetOutput(BaseModel):
    success: bool
    target_type: str
    channel_id: str | None
    channel_title: str | None
    custom_url: str | None
    video_id: str | None
    title: str | None
    error: str | None = None


# 2. Fetch Metadata
class FetchMetadataInput(BaseModel):
    target: str = Field(description="Resolved video ID, channel ID, or uploaded path")


class FetchMetadataOutput(BaseModel):
    success: bool
    video_data: dict[str, Any] | None = None
    channel_info: dict[str, Any] | None = None
    metrics_data: dict[str, Any] = Field(default_factory=dict)
    recent_videos: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# 3. Search Competitors
class SearchCompetitorsInput(BaseModel):
    query: str = Field(description="Search terms representing the primary topic")
    max_results: int = Field(default=12, ge=3, le=25, description="Number of candidate videos to retrieve")


class SearchCompetitorsOutput(BaseModel):
    success: bool
    query: str
    candidates_count: int
    candidate_videos: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# 4. Retrieve Transcript
class RetrieveTranscriptInput(BaseModel):
    video_id: str | None = Field(default=None, description="YouTube Video ID")
    upload_path: str | None = Field(default=None, description="Path to uploaded transcript file (.txt, .srt, .vtt)")


class RetrieveTranscriptOutput(BaseModel):
    success: bool
    has_transcript: bool
    source_type: str
    summary_text: str | None = None
    sample_segments: list[str] = Field(default_factory=list)
    error: str | None = None


# 5. Calculate Video Metrics
class CalculateMetricsInput(BaseModel):
    submitted_video: dict[str, Any] = Field(description="Target video data dict")
    candidate_videos: list[dict[str, Any]] = Field(description="List of candidate competitor videos")
    max_competitors: int = Field(default=5, ge=1, le=8)


class CalculateMetricsOutput(BaseModel):
    success: bool
    selected_competitors: list[dict[str, Any]] = Field(default_factory=list)
    target_velocity_daily: float = 0.0
    top_competitor_velocity_daily: float = 0.0
    error: str | None = None


# 6. Compare Videos Matrix
class CompareVideosMatrixInput(BaseModel):
    submitted_video: dict[str, Any] = Field(description="Submitted video metadata")
    selected_competitors: list[dict[str, Any]] = Field(description="Selected benchmark competitors")


class CompareVideosMatrixOutput(BaseModel):
    success: bool
    side_by_side_matrix: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# 7. Extract Keywords & Hashtags
class ExtractKeywordsInput(BaseModel):
    submitted_video: dict[str, Any] = Field(description="Submitted video metadata")
    selected_competitors: list[dict[str, Any]] = Field(description="Selected benchmark competitors")


class ExtractKeywordsOutput(BaseModel):
    success: bool
    keyword_analysis: dict[str, Any] = Field(default_factory=dict)
    hashtag_analysis: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# 8. Identify Content Gaps
class IdentifyGapsInput(BaseModel):
    submitted_video: dict[str, Any] = Field(description="Target video data")
    selected_competitors: list[dict[str, Any]] = Field(description="Selected benchmark competitors")
    keyword_table: dict[str, Any] = Field(default_factory=dict)
    hashtag_data: dict[str, Any] = Field(default_factory=dict)


class IdentifyGapsOutput(BaseModel):
    success: bool
    gaps: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# 9. Generate Recommendations
class GenerateRecommendationsInput(BaseModel):
    gaps: list[dict[str, Any]] = Field(description="Identified content gaps")


class GenerateRecommendationsOutput(BaseModel):
    success: bool
    action_plan: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# 10. Retrieve Creator Memories
class RetrieveMemoriesInput(BaseModel):
    query: str = Field(description="Query keywords or topic to search relevant creator memories")
    categories: list[str] | None = Field(default=None, description="Optional categories: creator_profile, episodic, semantic")


class RetrieveMemoriesOutput(BaseModel):
    success: bool
    memories: list[dict[str, Any]] = Field(default_factory=list)
    formatted_context: str = ""
    error: str | None = None


# 11. Save Analysis Results & Memories
class SaveResultsInput(BaseModel):
    channel_name: str
    channel_id: str
    analyzed_video: dict[str, Any]
    selected_competitors: list[dict[str, Any]]
    gaps: list[dict[str, Any]]
    action_plan: list[dict[str, Any]]
    topic_intent: dict[str, Any]
    metrics_summary: dict[str, Any]
    eligible_memories: list[dict[str, Any]] = Field(default_factory=list)


class SaveResultsOutput(BaseModel):
    success: bool
    report_id: str | None = None
    evidence_count: int = 0
    opportunities_count: int = 0
    memories_saved_count: int = 0
    error: str | None = None


# 12. Get Latest Analysis Report
class GetLatestReportInput(BaseModel):
    pass


class GetLatestReportOutput(BaseModel):
    success: bool
    has_report: bool
    channel_title: str | None = None
    summary: str | None = None
    top_recommendations: list[str] = Field(default_factory=list)
    key_findings: list[dict[str, Any]] = Field(default_factory=list)
    competitor_intelligence: dict[str, Any] = Field(default_factory=dict)
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    error: str | None = None


# 13. Get Connected Channel Summary
class GetChannelSummaryInput(BaseModel):
    pass


class GetChannelSummaryOutput(BaseModel):
    success: bool
    has_channel: bool
    title: str | None = None
    channel_name: str | None = None
    channel_id: str | None = None
    custom_url: str | None = None
    subscriber_count: int = 0
    video_count: int = 0
    view_count: int = 0
    recent_videos: list[dict[str, Any]] = Field(default_factory=list)
    is_oauth_connected: bool = False
    active_conversation_channel: dict[str, Any] | None = None
    error: str | None = None


# 14. Get Channel Video Inventory
class GetChannelInventoryInput(BaseModel):
    channel_id: str | None = Field(default=None, description="YouTube Channel ID. If omitted, uses the workspace's connected channel.")
    max_results: int = Field(default=500, ge=5, le=500, description="Number of channel videos to inspect (fetches pages up to this operational limit)")
    page_token: str | None = Field(default=None, description="Optional pagination token to fetch a single specific page")


class GetChannelInventoryOutput(BaseModel):
    success: bool
    channel_title: str | None = None
    channel_id: str | None = None
    videos: list[dict[str, Any]] = Field(default_factory=list)
    total_channel_videos: int = 0
    inspected_count: int = 0
    playlist_items_count: int = 0
    metadata_items_count: int = 0
    is_complete: bool = False
    playlist_coverage_percentage: float = 0.0
    metadata_coverage_percentage: float = 0.0
    coverage_percentage: float = 0.0
    coverage_notes: str = ""
    observation_period: str = "Lifetime public metrics"
    error: str | None = None


# 15. Rank and Calculate Video Metrics
class RankVideosInput(BaseModel):
    metric: str = Field(default="views", description="Ranking metric: 'views', 'likes', 'comments', 'engagement_rate', 'views_per_day'")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of top videos to return (e.g. 1 for winning video, 5 or 10 for top list)")
    days_filter: int | None = Field(
        default=None,
        description=(
            "Optional date filter in days. Defaults to None for all-time / lifetime ranking. "
            "DO NOT set this parameter unless the creator explicitly requests a specific timeframe "
            "(e.g. 'in the last 30 days', 'this month', 'past week'). "
            "For all-time, lifetime, or general 'best' or 'most viewed' video requests, leave as None."
        ),
    )
    content_type: str | None = Field(
        default="all",
        description="Filter by content format: 'all' (default), 'shorts' (videos <= 60 seconds), or 'long_form' (videos > 60 seconds).",
    )
    channel_id: str | None = Field(default=None, description="Channel ID to rank. If omitted, uses connected channel.")


class RankVideosOutput(BaseModel):
    success: bool
    ranking_metric: str
    winning_video: dict[str, Any] | None = None
    ranked_videos: list[dict[str, Any]] = Field(default_factory=list)
    inspected_count: int = 0
    total_channel_videos: int = 0
    coverage_summary: str = ""
    explanation: str = ""
    error: str | None = None


# 16. Compare Videos Detailed
class CompareVideosDetailedInput(BaseModel):
    target_video_id: str | None = Field(default=None, description="Target video ID (or 'best' for top video)")
    benchmark_video_id: str | None = Field(default=None, description="Benchmark video ID (or 'latest' for latest upload)")
    channel_id: str | None = Field(default=None, description="Channel ID context")


class CompareVideosDetailedOutput(BaseModel):
    success: bool
    video_a: dict[str, Any] | None = None
    video_b: dict[str, Any] | None = None
    metric_comparison: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
# 17. Get Video Details
class GetVideoDetailsInput(BaseModel):
    video_id: str = Field(description="11-character YouTube video ID to inspect")


class GetVideoDetailsOutput(BaseModel):
    success: bool
    video_id: str = ""
    title: str = ""
    channel_title: str = ""
    channel_id: str = ""
    view_count: int = 0
    like_count: int | None = None
    comment_count: int | None = None
    duration: str = ""
    duration_formatted: str = ""
    published_at: str = ""
    url: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    error: str | None = None


# 18. Search YouTube Videos
class SearchYouTubeVideosInput(BaseModel):
    query: str = Field(description="Search keywords, niche topic, or competitor channel query")
    max_results: int = Field(default=10, ge=1, le=25, description="Number of results to retrieve")


class SearchYouTubeVideosOutput(BaseModel):
    success: bool
    query: str
    videos: list[dict[str, Any]] = Field(default_factory=list)
    total_found: int = 0
    error: str | None = None


# 19. Search YouTube Growth Strategies (Tavily)
class SearchGrowthStrategiesInput(BaseModel):
    topic: str = Field(description="Specific YouTube growth topic (e.g. hook design, retention pacing, packaging, CTR, shorts conversion)")
    focus_area: str | None = Field(default=None, description="Optional focus area: hooks, thumbnails, titles, pacing, SEO, audience engagement")


class SearchGrowthStrategiesOutput(BaseModel):
    success: bool
    topic: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    direct_answer: str | None = None
    error: str | None = None


# 20. Research YouTube Algorithm Updates (Tavily)
class ResearchAlgorithmUpdatesInput(BaseModel):
    query: str = Field(default="YouTube algorithm update policy creator insider", description="Topic or policy area to research")


class ResearchAlgorithmUpdatesOutput(BaseModel):
    success: bool
    query: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    direct_answer: str | None = None
    error: str | None = None


# 21. Research Niche and Trends (Tavily)
class ResearchNicheTrendsInput(BaseModel):
    niche: str = Field(description="Niche, industry, or content domain to analyze")
    topic: str | None = Field(default=None, description="Optional subtopic or angle")


class ResearchNicheTrendsOutput(BaseModel):
    success: bool
    niche: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# 22. Search Competitor Content Strategies (Tavily)
class SearchCompetitorStrategiesInput(BaseModel):
    competitor_channel_or_niche: str = Field(description="Competitor channel name or niche topic")


class SearchCompetitorStrategiesOutput(BaseModel):
    success: bool
    target: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# 23. Search External Web (Tavily)
class SearchExternalWebInput(BaseModel):
    query: str = Field(description="Web search query")
    max_results: int = Field(default=5, ge=1, le=10)


class SearchExternalWebOutput(BaseModel):
    success: bool
    query: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# 24. Get Channel Details (Public & Connected)
class GetChannelDetailsInput(BaseModel):
    channel_id: str | None = Field(default=None, description="YouTube Channel ID, handle (e.g. '@AlexBuildsAI'), or URL. If omitted, uses the active conversation channel.")


class GetChannelDetailsOutput(BaseModel):
    success: bool
    channel_id: str | None = None
    channel_name: str | None = None
    custom_url: str | None = None
    subscriber_count: int | None = None
    video_count: int = 0
    view_count: int = 0
    description: str = ""
    recent_videos: list[dict[str, Any]] = Field(default_factory=list)
    is_oauth_connected: bool = False
    source: str = "official_youtube_api_v3"
    error: str | None = None


# 25. Investigate Creator Struggles
class InvestigateStrugglesInput(BaseModel):
    query: str = Field(description="Natural language description of the YouTube creator struggle or problem")
    force_refresh: bool = Field(default=False, description="Whether to bypass cache and query live sources")


class InvestigateStrugglesOutput(BaseModel):
    success: bool
    query: str = ""
    analysis: dict[str, Any] = Field(default_factory=dict)
    sources: dict[str, Any] = Field(default_factory=dict)
    total_evidence_count: int = 0
    error: str | None = None


# 26. Search Niche Radar
class SearchNicheRadarInput(BaseModel):
    niche: str = Field(description="Niche keyword or topic to explore and rank")
    time_period: str = Field(default="all", description="Time period filter: 7d, 30d, 90d, all")
    category: str = Field(default="most_viewed", description="Category: most_viewed, high_average, recent, momentum, small_channel")


class SearchNicheRadarOutput(BaseModel):
    success: bool
    niche: str = ""
    videos: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    is_cached: bool = False
    error: str | None = None


# Tool Execution Logic
# -------------------------------------------------------------

def tool_resolve_youtube_target(args: ResolveTargetInput, ctx: ToolContext) -> ResolveTargetOutput:
    try:
        ingestion = DataIngestionService()
        sub = AnalysisSubmission(input_target=args.target)
        res = ingestion.process_submission(sub)
        v = res.analyzed_video or (res.recent_videos[0] if res.recent_videos else None)

        return ResolveTargetOutput(
            success=True,
            target_type=res.source_type,
            channel_id=res.channel_id,
            channel_title=res.channel_title,
            custom_url=res.custom_url,
            video_id=v.get("video_id") if v else None,
            title=v.get("title") if v else (f"{res.channel_title} Channel Content"),
        )
    except Exception as e:
        logger.warning(f"Error in tool_resolve_youtube_target: {e}")
        return ResolveTargetOutput(
            success=False,
            target_type="unknown",
            channel_id=None,
            channel_title=None,
            custom_url=None,
            video_id=None,
            title=None,
            error=str(e),
        )


def tool_fetch_youtube_metadata(args: FetchMetadataInput, ctx: ToolContext) -> FetchMetadataOutput:
    try:
        ingestion = DataIngestionService()
        sub = AnalysisSubmission(input_target=args.target)
        res = ingestion.process_submission(sub)

        analyzed_v = res.analyzed_video
        if not analyzed_v and res.recent_videos:
            analyzed_v = res.recent_videos[0]

        if not analyzed_v:
            analyzed_v = {
                "video_id": None,
                "title": f"{res.channel_title or 'Channel'} Content",
                "description": res.description or "",
                "channel_id": res.channel_id,
                "channel_title": res.channel_title or "Creator Channel",
                "published_at": "",
                "view_count": int(res.metrics_data.get("total_views") or 0),
                "like_count": None,
                "comment_count": None,
                "thumbnail_url": res.thumbnail_url,
                "tags": [],
            }

        ch_info = {
            "title": res.channel_title or "Creator Channel",
            "channel_id": res.channel_id,
            "custom_url": res.custom_url or f"@{res.channel_title or 'channel'}",
            "channel_url": res.channel_url or "https://www.youtube.com",
            "thumbnail_url": res.thumbnail_url or "/assets/images/creator-avatar.svg",
            "subscribers": int(res.metrics_data.get("subscribers") or 0),
            "total_views": int(res.metrics_data.get("total_views") or 0),
            "video_count": int(res.video_count or res.metrics_data.get("videos") or 1),
            "analyzed_video": analyzed_v,
            "recent_videos": res.recent_videos[:6] if res.recent_videos else [],
        }

        return FetchMetadataOutput(
            success=True,
            video_data=analyzed_v,
            channel_info=ch_info,
            metrics_data=res.metrics_data,
            recent_videos=res.recent_videos[:6] if res.recent_videos else [],
        )
    except Exception as e:
        logger.warning(f"Error in tool_fetch_youtube_metadata: {e}")
        return FetchMetadataOutput(success=False, error=str(e))


def tool_search_competitor_videos(args: SearchCompetitorsInput, ctx: ToolContext) -> SearchCompetitorsOutput:
    try:
        ingestion = DataIngestionService()
        clean_q = args.query.strip()
        candidates = ingestion.yt_adapter.search_competitor_videos(clean_q, max_results=args.max_results)

        # If too few, fallback search with truncated words
        if len(candidates) < 3 and len(clean_q.split()) > 2:
            alt_q = " ".join(clean_q.split()[:3])
            more = ingestion.yt_adapter.search_competitor_videos(alt_q, max_results=10)
            seen_ids = {c["video_id"] for c in candidates}
            for m in more:
                if m["video_id"] not in seen_ids:
                    candidates.append(m)

        return SearchCompetitorsOutput(
            success=True,
            query=clean_q,
            candidates_count=len(candidates),
            candidate_videos=candidates,
        )
    except Exception as e:
        logger.warning(f"Error in tool_search_competitor_videos: {e}")
        return SearchCompetitorsOutput(success=False, query=args.query, candidates_count=0, error=str(e))


def tool_retrieve_video_transcript(args: RetrieveTranscriptInput, ctx: ToolContext) -> RetrieveTranscriptOutput:
    try:
        # Check uploaded file path first
        if args.upload_path and os.path.exists(args.upload_path):
            with open(args.upload_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read(8000)

            # Clean out SRT / VTT timestamp markers if present
            clean_lines = []
            for line in raw_text.splitlines():
                line = line.strip()
                if not line or line.isdigit() or "-->" in line:
                    continue
                clean_lines.append(line)

            text_body = " ".join(clean_lines)[:1500]
            return RetrieveTranscriptOutput(
                success=True,
                has_transcript=True,
                source_type="uploaded_file",
                summary_text=text_body,
                sample_segments=clean_lines[:5],
            )

        # Public YouTube API does not provide full transcript captions without OAuth or special third-party proxies
        return RetrieveTranscriptOutput(
            success=True,
            has_transcript=False,
            source_type="unavailable_public_api",
            summary_text="Transcript captions are not accessible via public unauthenticated YouTube Data API v3 without an uploaded file.",
        )
    except Exception as e:
        return RetrieveTranscriptOutput(success=False, has_transcript=False, source_type="error", error=str(e))


def tool_calculate_video_metrics(args: CalculateMetricsInput, ctx: ToolContext) -> CalculateMetricsOutput:
    try:
        selected = CompetitorAnalyzer.rank_and_select_competitors(
            submitted_video=args.submitted_video,
            candidate_videos=args.candidate_videos,
            max_competitors=args.max_competitors,
        )
        target_views = float(args.submitted_video.get("view_count") or 0)
        top_comp_velocity = selected[0].get("views_per_day", 0.0) if selected else 0.0

        return CalculateMetricsOutput(
            success=True,
            selected_competitors=selected,
            target_velocity_daily=target_views / 30.0,
            top_competitor_velocity_daily=top_comp_velocity,
        )
    except Exception as e:
        logger.warning(f"Error in tool_calculate_video_metrics: {e}")
        return CalculateMetricsOutput(success=False, error=str(e))


def tool_compare_videos_matrix(args: CompareVideosMatrixInput, ctx: ToolContext) -> CompareVideosMatrixOutput:
    try:
        matrix = CompetitorAnalyzer.build_side_by_side_matrix(
            args.submitted_video,
            args.selected_competitors,
        )
        return CompareVideosMatrixOutput(success=True, side_by_side_matrix=matrix)
    except Exception as e:
        logger.warning(f"Error in tool_compare_videos_matrix: {e}")
        return CompareVideosMatrixOutput(success=False, error=str(e))


def tool_extract_keywords_and_hashtags(args: ExtractKeywordsInput, ctx: ToolContext) -> ExtractKeywordsOutput:
    try:
        kw_table = CompetitorAnalyzer.analyze_keywords_side_by_side(args.submitted_video, args.selected_competitors)
        ht_data = CompetitorAnalyzer.analyze_hashtags_side_by_side(args.submitted_video, args.selected_competitors)
        return ExtractKeywordsOutput(
            success=True,
            keyword_analysis=kw_table,
            hashtag_analysis=ht_data,
        )
    except Exception as e:
        logger.warning(f"Error in tool_extract_keywords_and_hashtags: {e}")
        return ExtractKeywordsOutput(success=False, error=str(e))


def tool_identify_content_gaps(args: IdentifyGapsInput, ctx: ToolContext) -> IdentifyGapsOutput:
    try:
        gaps = CompetitorAnalyzer.generate_gap_analysis(
            args.submitted_video,
            args.selected_competitors,
            args.keyword_table,
            args.hashtag_data,
        )
        return IdentifyGapsOutput(success=True, gaps=gaps)
    except Exception as e:
        logger.warning(f"Error in tool_identify_content_gaps: {e}")
        return IdentifyGapsOutput(success=False, error=str(e))


def tool_generate_prioritized_recommendations(args: GenerateRecommendationsInput, ctx: ToolContext) -> GenerateRecommendationsOutput:
    try:
        plan = CompetitorAnalyzer.generate_prioritized_action_plan(args.gaps)
        return GenerateRecommendationsOutput(success=True, action_plan=plan)
    except Exception as e:
        logger.warning(f"Error in tool_generate_prioritized_recommendations: {e}")
        return GenerateRecommendationsOutput(success=False, error=str(e))


def tool_retrieve_creator_memories(args: RetrieveMemoriesInput, ctx: ToolContext) -> RetrieveMemoriesOutput:
    try:
        mem_service = CreatorMemoryService(ctx.db)
        memories = mem_service.retrieve_relevant_memories(
            workspace_id=ctx.workspace_id,
            query_context=args.query,
            categories=args.categories,
            limit=5,
        )
        ctx_str = mem_service.format_memory_context_for_prompt(memories)
        mem_dicts = [
            {
                "id": m.id,
                "category": m.category,
                "type": m.memory_type,
                "content": m.content,
                "verification_status": m.verification_status,
            }
            for m in memories
        ]
        return RetrieveMemoriesOutput(
            success=True,
            memories=mem_dicts,
            formatted_context=ctx_str,
        )
    except Exception as e:
        logger.warning(f"Error in tool_retrieve_creator_memories: {e}")
        return RetrieveMemoriesOutput(success=False, error=str(e))


def tool_save_analysis_results_and_memories(args: SaveResultsInput, ctx: ToolContext) -> SaveResultsOutput:
    try:
        from app.database.models import AnalysisReport, ChannelConnection

        ev_service = EvidenceService(ctx.db)
        opp_service = OpportunityEngineService(ctx.db)
        mem_service = CreatorMemoryService(ctx.db)

        # 1. Register Evidence for Target
        v_title = args.analyzed_video.get("title", "Target Video")
        v_id = args.analyzed_video.get("video_id")
        v_url = f"https://www.youtube.com/watch?v={v_id}" if v_id else "https://www.youtube.com"

        ev_target = ev_service.register_evidence(
            workspace_id=ctx.workspace_id,
            job_id=ctx.job_id,
            source_type="submitted_video_audit",
            title=f"Submitted Target: {v_title}",
            snippet=(
                f"Verified metrics for '{v_title}': {args.analyzed_video.get('view_count', 0):,} views, "
                f"{args.analyzed_video.get('like_count') or 0} likes, duration: {args.analyzed_video.get('duration', 'Unknown')}."
            ),
            url=v_url,
            confidence_score=1.0,
        )

        # 2. Register Evidence for Competitors
        comp_evidence_ids = []
        for comp in args.selected_competitors:
            ev_comp = ev_service.register_evidence(
                workspace_id=ctx.workspace_id,
                job_id=ctx.job_id,
                source_type="competitor_performance_benchmark",
                title=f"Competitor: {comp.get('title')} ({comp.get('channel_title')})",
                snippet=(
                    f"Public YouTube API metrics: {comp.get('view_count', 0):,} views, "
                    f"pacing at {int(comp.get('views_per_day', 0)):,} views/day over {comp.get('age_days', 0)} days."
                ),
                url=f"https://www.youtube.com/watch?v={comp.get('video_id')}",
                confidence_score=comp.get("confidence_score", 0.95),
            )
            comp_evidence_ids.append(ev_comp.id)

        # 3. Register Findings from Gaps
        registered_findings = []
        for idx, gap in enumerate(args.gaps[:4]):
            f_rec = ev_service.register_finding(
                workspace_id=ctx.workspace_id,
                job_id=ctx.job_id,
                category=gap["category"],
                epistemic_tier="measured_observation" if "Velocity" in gap["title"] or "Markers" in gap["title"] else "inferred_hypothesis",
                statement=gap["title"],
                explanation=f"{gap.get('submitted_evidence', '')} {gap.get('competitor_evidence', '')}",
                recommendation=gap.get("recommended_action", ""),
                evidence_ids=[ev_target.id] + comp_evidence_ids[:2],
            )
            registered_findings.append({
                "id": idx + 1,
                "title": f_rec.statement,
                "description": f_rec.explanation,
                "recommendation": f_rec.recommendation,
                "category": gap["category"],
                "priority": gap.get("priority", "HIGH"),
                "evidence_id": ev_target.id,
            })

        # 4. Generate Content Opportunities
        opp_count = 0
        topic_str = args.topic_intent.get("primary_topic", "Content Opportunity")
        for idx, gap in enumerate(args.gaps[:3], 1):
            opp_service.create_opportunity(
                workspace_id=ctx.workspace_id,
                job_id=ctx.job_id,
                title=f"{topic_str}: {gap.get('recommended_action', '')}"[:100],
                topic_category=gap["category"],
                demand_score=round(8.5 + (idx * 0.3), 1),
                saturation_score=round(3.8 + (idx * 0.4), 1),
                authority_score=round(8.2 + (idx * 0.2), 1),
                differentiation_angle=gap.get("expected_benefit", "Differentiation angle based on competitor audit"),
                target_audience=args.topic_intent.get("intended_audience", "General YouTube Audience"),
                tags=[gap["category"], "Competitor Gap", "Actionable"],
                is_bookmarked=(idx == 1),
            )
            opp_count += 1

        # 5. Persist AnalysisReport
        top_recs = [step["action"] for step in args.action_plan[:3]] or [
            "Incorporate high-intent keyword phrases into title opening.",
            "Add timestamp chapters into description for search indexing.",
            "Expand description to provide comprehensive context.",
        ]

        report = AnalysisReport(
            workspace_id=ctx.workspace_id,
            job_id=ctx.job_id,
            channel_title=args.channel_name,
            channel_id=args.channel_id,
            summary=(
                f"Agentic audit complete for '{v_title}' on {args.channel_name}. "
                f"Benchmarked {len(args.selected_competitors)} topical competitors. "
                f"Discovered {len(args.gaps)} evidence-grounded content gaps."
            ),
            metrics_summary=args.metrics_summary,
            key_findings=registered_findings,
            recommendations=top_recs,
        )
        ctx.db.add(report)

        # 6. Save eligible creator memories
        memories_count = 0
        for m in args.eligible_memories:
            try:
                mem_service.save_memory(
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    category=m.get("category", "semantic"),
                    memory_type=m.get("memory_type", "verified_insight"),
                    content=m.get("content", ""),
                    structured_payload=m.get("structured_payload", {}),
                    source_job_id=ctx.job_id,
                    source_evidence_id=ev_target.id,
                    confidence_score=float(m.get("confidence_score", 0.9)),
                    verification_status=m.get("verification_status", "verified_observation"),
                )
                memories_count += 1
            except Exception as mem_err:
                logger.warning(f"Could not persist candidate memory: {mem_err}")

        # 7. Update or create ChannelConnection
        ch_conn = ctx.db.query(ChannelConnection).filter(ChannelConnection.workspace_id == ctx.workspace_id).first()
        if ch_conn:
            ch_conn.title = args.channel_name
            ch_conn.channel_id = args.channel_id
            ch_conn.is_active = True
        else:
            ch_conn = ChannelConnection(
                workspace_id=ctx.workspace_id,
                channel_id=args.channel_id,
                title=args.channel_name,
                is_active=True,
            )
            ctx.db.add(ch_conn)

        ctx.db.commit()

        return SaveResultsOutput(
            success=True,
            report_id=report.id,
            evidence_count=1 + len(comp_evidence_ids),
            opportunities_count=opp_count,
            memories_saved_count=memories_count,
        )
    except Exception as e:
        ctx.db.rollback()
        logger.exception(f"Error in tool_save_analysis_results_and_memories: {e}")
        return SaveResultsOutput(success=False, error=str(e))


def tool_get_latest_analysis_report(args: GetLatestReportInput, ctx: ToolContext) -> GetLatestReportOutput:
    try:
        from app.database.models import AnalysisReport
        report = (
            ctx.db.query(AnalysisReport)
            .filter(AnalysisReport.workspace_id == ctx.workspace_id)
            .order_by(AnalysisReport.created_at.desc())
            .first()
        )
        if not report:
            return GetLatestReportOutput(
                success=True,
                has_report=False,
                summary="No previous analysis reports exist for this workspace.",
            )
        ms = report.metrics_summary or {}
        comp_intel = ms.get("competitor_intelligence", {})
        return GetLatestReportOutput(
            success=True,
            has_report=True,
            channel_title=report.channel_title,
            summary=report.summary,
            top_recommendations=report.recommendations or [],
            key_findings=report.key_findings or [],
            competitor_intelligence=comp_intel,
            metrics_summary=ms,
            created_at=report.created_at.isoformat() if report.created_at else None,
        )
    except Exception as e:
        logger.warning(f"Error in tool_get_latest_analysis_report: {e}")
        return GetLatestReportOutput(success=False, has_report=False, error=str(e))


def tool_get_connected_channel_summary(args: GetChannelSummaryInput, ctx: ToolContext) -> GetChannelSummaryOutput:
    try:
        from app.database.models import ChannelConnection, AnalysisReport
        from app.agent.providers.youtube import YouTubeAdapter

        # Deterministic precedence:
        # If an active channel is established in this conversation, check if IT has an OAuth connection.
        # Fall back to arbitrary workspace OAuth connection ONLY IF no active conversation channel exists.
        conn = None
        if ctx.active_channel_id:
            conn = ctx.db.query(ChannelConnection).filter(
                ChannelConnection.workspace_id == ctx.workspace_id,
                ChannelConnection.channel_id == ctx.active_channel_id,
            ).first()
        else:
            conn = ctx.db.query(ChannelConnection).filter(
                ChannelConnection.workspace_id == ctx.workspace_id
            ).first()

        if not conn and not ctx.active_channel_id:
            rep = (
                ctx.db.query(AnalysisReport)
                .filter(AnalysisReport.workspace_id == ctx.workspace_id)
                .order_by(AnalysisReport.created_at.desc())
                .first()
            )
            if rep:
                ctx.active_channel_id = rep.channel_id
                ctx.active_channel_name = rep.channel_title
                r_ms = rep.metrics_summary or {}
                r_info = r_ms.get("channel_info", {}) if isinstance(r_ms, dict) else {}
                ctx.active_channel_context = {
                    "channel_id": rep.channel_id,
                    "channel_name": rep.channel_title,
                    "custom_url": r_info.get("custom_url"),
                    "subscriber_count": r_info.get("subscriber_count") or (r_ms.get("subscribers", {}).get("raw") if isinstance(r_ms.get("subscribers"), dict) else 0),
                    "video_count": len(r_info.get("recent_videos", [])),
                    "view_count": r_info.get("view_count") or (r_ms.get("total_views", {}).get("raw") if isinstance(r_ms.get("total_views"), dict) else 0),
                    "context_source": "workspace_analysis_report",
                }

        if not conn:
            active_info = None
            ch_name = ctx.active_channel_name or ctx.active_channel_id
            recent = []
            sub_count = 0
            vid_count = 0
            view_count = 0
            cust_url = None
            if ctx.active_channel_id:
                if ctx.active_channel_context:
                    ch_name = ctx.active_channel_context.get("channel_name") or ch_name
                    cust_url = ctx.active_channel_context.get("custom_url")
                    sub_count = ctx.active_channel_context.get("subscriber_count") or 0
                    vid_count = ctx.active_channel_context.get("video_count") or 0
                    view_count = ctx.active_channel_context.get("view_count") or 0

                try:
                    yt = YouTubeAdapter()
                    t_info = yt.parse_youtube_target(ctx.active_channel_id)
                    ch_d = yt.fetch_channel_data(t_info)
                    if ch_d:
                        recent = ch_d.get("recent_videos", [])[:8]
                        if not cust_url:
                            cust_url = ch_d.get("custom_url")
                        if not sub_count:
                            sub_count = ch_d.get("subscriber_count") or 0
                        if not vid_count:
                            vid_count = ch_d.get("video_count") or 0
                        if not view_count:
                            view_count = ch_d.get("view_count") or 0
                except Exception as e:
                    logger.debug(f"Could not fetch recent uploads for active channel in summary: {e}")

                active_info = {
                    "channel_id": ctx.active_channel_id,
                    "channel_name": ch_name,
                    "custom_url": cust_url,
                    "subscriber_count": sub_count,
                    "video_count": vid_count,
                    "view_count": view_count,
                    "recent_videos": recent,
                    "context_source": ctx.active_channel_context.get("context_source", "conversation_active_channel") if ctx.active_channel_context else "conversation_active_channel",
                }

            if ctx.active_channel_id or active_info:
                return GetChannelSummaryOutput(
                    success=True,
                    has_channel=True,
                    is_oauth_connected=False,
                    title=ch_name,
                    channel_name=ch_name,
                    channel_id=ctx.active_channel_id,
                    custom_url=cust_url,
                    subscriber_count=sub_count,
                    video_count=vid_count,
                    view_count=view_count,
                    recent_videos=recent,
                    active_conversation_channel=active_info,
                    error=None,
                )

            return GetChannelSummaryOutput(
                success=True,
                has_channel=False,
                is_oauth_connected=False,
                error="No connected channel or active channel found in this workspace. Please select or analyze a channel.",
            )

        recent = []
        report = (
            ctx.db.query(AnalysisReport)
            .filter(AnalysisReport.workspace_id == ctx.workspace_id)
            .order_by(AnalysisReport.created_at.desc())
            .first()
        )
        if report and report.channel_id == conn.channel_id and report.metrics_summary:
            ch_info = report.metrics_summary.get("channel_info", {})
            recent = ch_info.get("recent_videos", [])

        if not recent:
            try:
                yt = YouTubeAdapter()
                t_info = yt.parse_youtube_target(conn.channel_id)
                ch_d = yt.fetch_channel_data(t_info)
                if ch_d:
                    recent = ch_d.get("recent_videos", [])[:8]
            except Exception as e:
                logger.debug(f"Could not fetch recent uploads for connected channel: {e}")

        is_oauth = bool(getattr(conn, "access_token", None))
        return GetChannelSummaryOutput(
            success=True,
            has_channel=True,
            is_oauth_connected=is_oauth,
            title=conn.title,
            channel_name=conn.title,
            channel_id=conn.channel_id,
            custom_url=conn.custom_url,
            subscriber_count=conn.subscriber_count,
            video_count=conn.video_count,
            view_count=conn.view_count,
            recent_videos=recent,
        )
    except Exception as e:
        logger.warning(f"Error in tool_get_connected_channel_summary: {e}")
        return GetChannelSummaryOutput(success=False, has_channel=False, error=str(e))


def tool_get_channel_video_inventory(args: GetChannelInventoryInput, ctx: ToolContext) -> GetChannelInventoryOutput:
    try:
        from app.database.models import ChannelConnection, AnalysisReport
        from app.agent.providers.youtube import YouTubeAdapter

        # Protected active channel resolution:
        # Default to ctx.active_channel_id if args.channel_id is omitted or references active channel.
        # Otherwise respect explicitly supplied external/competitor channel ID.
        arg_cid = args.channel_id.strip() if (args.channel_id and isinstance(args.channel_id, str)) else ""
        if ctx.active_channel_id and (not arg_cid or arg_cid.lower() in ("my_channel", "active_channel", "connected_channel", "current_channel", ctx.active_channel_id.lower())):
            target_channel_id = ctx.active_channel_id
        else:
            target_channel_id = arg_cid or ctx.active_channel_id

        # Performance guard: reuse complete cached inventory from context if available for active channel
        if (
            ctx.cached_inventory
            and ctx.inventory_is_complete
            and target_channel_id == ctx.active_channel_id
            and (not args.page_token)
        ):
            tot = len(ctx.cached_inventory)
            ch_title = ctx.active_channel_name or "Channel"
            return GetChannelInventoryOutput(
                success=True,
                channel_title=ch_title,
                channel_id=target_channel_id,
                videos=ctx.cached_inventory,
                total_channel_videos=tot,
                inspected_count=tot,
                playlist_items_count=tot,
                metadata_items_count=tot,
                is_complete=True,
                playlist_coverage_percentage=100.0,
                metadata_coverage_percentage=100.0,
                coverage_percentage=100.0,
                coverage_notes="Reused verified complete inventory from working memory cache.",
                observation_period=f"Channel uploads inventory ({tot}/{tot} uploads, 100.0% coverage)",
            )

        ch_conn = None
        if target_channel_id:
            ch_conn = ctx.db.query(ChannelConnection).filter(
                ChannelConnection.workspace_id == ctx.workspace_id,
                ChannelConnection.channel_id == target_channel_id,
            ).first()
        if not ch_conn and not target_channel_id:
            ch_conn = ctx.db.query(ChannelConnection).filter(
                ChannelConnection.workspace_id == ctx.workspace_id,
            ).first()

        channel_id = target_channel_id or (ch_conn.channel_id if ch_conn else None)
        if not channel_id:
            rep = (
                ctx.db.query(AnalysisReport)
                .filter(AnalysisReport.workspace_id == ctx.workspace_id)
                .order_by(AnalysisReport.created_at.desc())
                .first()
            )
            if rep:
                channel_id = rep.channel_id
                target_channel_id = rep.channel_id
                if not ctx.active_channel_name:
                    ctx.active_channel_name = rep.channel_title
        if not channel_id:
            return GetChannelInventoryOutput(
                success=False,
                error="No connected channel or active conversation channel found. Connect a channel or specify a channel ID.",
            )

        channel_title = (
            ch_conn.title if ch_conn
            else (ctx.active_channel_name or (ctx.active_channel_context.get("channel_name") if ctx.active_channel_context else "Channel"))
        )
        db_video_count = (
            int(ch_conn.video_count) if (ch_conn and ch_conn.video_count)
            else (int(ctx.active_channel_context.get("video_count", 0)) if ctx.active_channel_context else 0)
        )

        yt = YouTubeAdapter()
        uploads_playlist_id = yt.get_uploads_playlist_id(channel_id)

        inv = yt.fetch_channel_video_inventory(
            uploads_playlist_id=uploads_playlist_id,
            max_results=args.max_results,
            page_token=args.page_token,
        )
        videos = inv.get("videos", [])
        is_complete = bool(inv.get("is_complete", False))
        api_total = int(inv.get("total_results", 0) or 0)
        playlist_items_count = int(inv.get("playlist_items_count", len(videos)) or 0)
        metadata_items_count = int(inv.get("metadata_items_count", len(videos)) or 0)

        # Fallback to AnalysisReport cached recent uploads if YouTube API returned 0 or failed
        if not videos:
            report = (
                ctx.db.query(AnalysisReport)
                .filter(AnalysisReport.workspace_id == ctx.workspace_id)
                .order_by(AnalysisReport.created_at.desc())
                .first()
            )
            if report and report.metrics_summary:
                ch_info = report.metrics_summary.get("channel_info", {})
                videos = ch_info.get("recent_videos", [])
                metadata_items_count = len(videos)
                playlist_items_count = len(videos)

        inspected = len(videos)

        # Trustworthy total denominator: API pageInfo.totalResults first, then DB channel video_count
        authoritative_total = api_total if api_total > 0 else db_video_count

        if authoritative_total <= 0:
            # Denominator unknown: NEVER claim 100%, report 0.0% coverage
            coverage_pct = 0.0
            playlist_cov_pct = 0.0
            metadata_cov_pct = 0.0
            coverage_notes = "Channel total upload count is unknown; cannot verify inventory completeness."
            total_count = inspected
        else:
            total_count = max(authoritative_total, inspected)
            raw_playlist_pct = round((playlist_items_count / total_count) * 100, 1)
            raw_metadata_pct = round((metadata_items_count / total_count) * 100, 1)
            raw_cov_pct = round((inspected / total_count) * 100, 1)

            # Strict rule: NEVER report 100% if pagination is incomplete, metadata batches failed, or inspected < total_count
            if not is_complete or inspected < total_count or metadata_items_count < playlist_items_count:
                coverage_pct = min(raw_cov_pct, 99.0) if raw_cov_pct >= 100.0 else raw_cov_pct
                playlist_cov_pct = min(raw_playlist_pct, 99.0) if raw_playlist_pct >= 100.0 and not is_complete else raw_playlist_pct
                metadata_cov_pct = min(raw_metadata_pct, 99.0) if raw_metadata_pct >= 100.0 and metadata_items_count < total_count else raw_metadata_pct

                notes_reasons = []
                if not is_complete:
                    notes_reasons.append("pagination limit reached before playlist end")
                if metadata_items_count < playlist_items_count:
                    notes_reasons.append(f"{playlist_items_count - metadata_items_count} videos failed metadata retrieval")
                if inspected < total_count:
                    notes_reasons.append(f"retrieved {inspected} of {total_count} uploads")
                coverage_notes = f"Partial coverage ({coverage_pct}%): " + ", ".join(notes_reasons) + "."
            else:
                coverage_pct = 100.0
                playlist_cov_pct = 100.0
                metadata_cov_pct = 100.0
                coverage_notes = f"Verified 100% complete coverage: All {inspected} uploads successfully retrieved and analyzed."

        return GetChannelInventoryOutput(
            success=True,
            channel_title=channel_title,
            channel_id=channel_id,
            videos=videos,
            total_channel_videos=total_count,
            inspected_count=inspected,
            playlist_items_count=playlist_items_count,
            metadata_items_count=metadata_items_count,
            is_complete=is_complete,
            playlist_coverage_percentage=playlist_cov_pct,
            metadata_coverage_percentage=metadata_cov_pct,
            coverage_percentage=coverage_pct,
            coverage_notes=coverage_notes,
            observation_period=f"Channel uploads inventory ({inspected}/{total_count} uploads, {coverage_pct}% coverage)",
        )
    except Exception as e:
        logger.warning(f"Error in tool_get_channel_video_inventory: {e}")
        return GetChannelInventoryOutput(success=False, error=str(e))


def tool_rank_and_calculate_video_metrics(args: RankVideosInput, ctx: ToolContext) -> RankVideosOutput:
    try:
        arg_cid = args.channel_id.strip() if (args.channel_id and isinstance(args.channel_id, str)) else ""
        if ctx.active_channel_id and (not arg_cid or arg_cid.lower() in ("my_channel", "active_channel", "connected_channel", "current_channel", ctx.active_channel_id.lower())):
            target_cid = ctx.active_channel_id
        else:
            target_cid = arg_cid or ctx.active_channel_id

        inv_args = GetChannelInventoryInput(channel_id=target_cid, max_results=500)
        inv = tool_get_channel_video_inventory(inv_args, ctx)
        if not inv.success or not inv.videos:
            return RankVideosOutput(
                success=False,
                ranking_metric=args.metric,
                error=inv.error or "No channel videos found to rank.",
            )

        videos = list(inv.videos)
        now = datetime.now(timezone.utc)

        enriched_videos = []
        for v in videos:
            v_copy = dict(v)
            pub_str = v.get("published_at", "")
            age_days = 1
            if pub_str:
                try:
                    clean_pub = pub_str.replace("Z", "+00:00")
                    pub_dt = datetime.fromisoformat(clean_pub)
                    age_days = max(1, (now - pub_dt).days)
                except Exception:
                    age_days = 1
            v_copy["age_days"] = age_days

            vc = int(v.get("view_count", 0) or 0)
            lc = int(v.get("like_count") or 0) if v.get("like_count") is not None else 0
            cc = int(v.get("comment_count") or 0) if v.get("comment_count") is not None else 0

            v_copy["views_per_day"] = round(vc / age_days, 1)
            v_copy["engagement_rate"] = round(((lc + cc) / max(vc, 1)) * 100, 2)
            enriched_videos.append(v_copy)

        # Content format filtering (Shorts vs Long-form)
        c_type = (args.content_type or "all").lower().strip()
        if c_type in ("shorts", "short"):
            def is_short_vid(v_item: dict[str, Any]) -> bool:
                dur = str(v_item.get("duration", ""))
                if ":" in dur:
                    parts = dur.split(":")
                    if len(parts) == 2:
                        try:
                            return (int(parts[0]) * 60 + int(parts[1])) <= 60
                        except ValueError:
                            pass
                return False
            enriched_videos = [v for v in enriched_videos if is_short_vid(v)]
        elif c_type in ("long_form", "longform", "videos", "long"):
            def is_long_vid(v_item: dict[str, Any]) -> bool:
                dur = str(v_item.get("duration", ""))
                if ":" in dur:
                    parts = dur.split(":")
                    if len(parts) >= 3:
                        return True
                    if len(parts) == 2:
                        try:
                            return (int(parts[0]) * 60 + int(parts[1])) > 60
                        except ValueError:
                            pass
                return True
            enriched_videos = [v for v in enriched_videos if is_long_vid(v)]

        # Date filtering ONLY if explicitly requested with a valid positive integer
        if args.days_filter is not None and isinstance(args.days_filter, int) and args.days_filter > 0:
            enriched_videos = [v for v in enriched_videos if v.get("age_days", 0) <= args.days_filter]

        metric_key = args.metric.lower().strip()
        if metric_key in ("views", "view_count", "most_viewed"):
            sort_fn = lambda x: int(x.get("view_count", 0) or 0)
            metric_label = "Lifetime Views"
        elif metric_key in ("engagement", "engagement_rate"):
            sort_fn = lambda x: float(x.get("engagement_rate", 0.0) or 0.0)
            metric_label = "Engagement Rate (%)"
        elif metric_key in ("velocity", "views_per_day", "views/day"):
            sort_fn = lambda x: float(x.get("views_per_day", 0.0) or 0.0)
            metric_label = "Views/Day Velocity"
        elif metric_key in ("likes", "like_count"):
            sort_fn = lambda x: int(x.get("like_count", 0) or 0)
            metric_label = "Like Count"
        elif metric_key in ("comments", "comment_count"):
            sort_fn = lambda x: int(x.get("comment_count", 0) or 0)
            metric_label = "Comment Count"
        else:
            sort_fn = lambda x: int(x.get("view_count", 0) or 0)
            metric_label = "Lifetime Views"

        # Honest response if filtering produced zero matches
        if not enriched_videos:
            period_note = f"in the last {args.days_filter} days" if args.days_filter else "matching criteria"
            fmt_note = f" ({c_type})" if c_type != "all" else ""
            return RankVideosOutput(
                success=True,
                ranking_metric=metric_label,
                winning_video=None,
                ranked_videos=[],
                inspected_count=0,
                total_channel_videos=inv.total_channel_videos,
                coverage_summary=f"Inspected {inv.inspected_count} uploads; 0 matched the active filter {period_note}{fmt_note}.",
                explanation=f"No uploads found on this channel {period_note}{fmt_note}.",
            )

        enriched_videos.sort(key=sort_fn, reverse=True)

        ranked_list = []
        for idx, v in enumerate(enriched_videos[:args.top_n]):
            v["rank"] = idx + 1
            if idx == 0:
                v["badge"] = f"Top #1 by {metric_label}"
            else:
                v["badge"] = f"Rank #{idx + 1}"
            ranked_list.append(v)

        winner = ranked_list[0] if ranked_list else None
        explanation = ""
        if winner:
            if metric_key in ("engagement", "engagement_rate"):
                val_str = f"{winner.get('engagement_rate', 0)}%"
            else:
                val = winner.get("view_count" if "views" in metric_key else metric_key, winner.get("view_count", 0))
                val_str = f"{val:,}" if isinstance(val, (int, float)) else str(val)

            explanation = (
                f"The highest-performing video is '{winner.get('title')}' with {val_str} {metric_label}. "
                f"Published on {str(winner.get('published_at', ''))[:10]} ({winner.get('age_days')} days ago), "
                f"pacing at {winner.get('views_per_day', 0):,} views/day."
            )

        cov_summary = (
            f"Ranked across {len(enriched_videos)} analyzed channel uploads "
            f"out of {inv.total_channel_videos} total uploads ({inv.coverage_percentage}% inventory coverage: {inv.coverage_notes})."
        )

        return RankVideosOutput(
            success=True,
            ranking_metric=metric_label,
            winning_video=winner,
            ranked_videos=ranked_list,
            inspected_count=len(enriched_videos),
            total_channel_videos=inv.total_channel_videos,
            coverage_summary=cov_summary,
            explanation=explanation,
        )
    except Exception as e:
        logger.warning(f"Error in tool_rank_and_calculate_video_metrics: {e}")
        return RankVideosOutput(success=False, ranking_metric=args.metric, error=str(e))


def tool_compare_videos_detailed(args: CompareVideosDetailedInput, ctx: ToolContext) -> CompareVideosDetailedOutput:
    try:
        arg_cid = args.channel_id.strip() if (args.channel_id and isinstance(args.channel_id, str)) else ""
        if ctx.active_channel_id and (not arg_cid or arg_cid.lower() in ("my_channel", "active_channel", "connected_channel", "current_channel", ctx.active_channel_id.lower())):
            target_cid = ctx.active_channel_id
        else:
            target_cid = arg_cid or ctx.active_channel_id

        inv_args = GetChannelInventoryInput(channel_id=target_cid, max_results=500)
        inv = tool_get_channel_video_inventory(inv_args, ctx)
        if not inv.success or not inv.videos:
            return CompareVideosDetailedOutput(success=False, error="Channel videos inventory unavailable for comparison.")

        videos = inv.videos
        v_a = None
        v_b = None

        if args.target_video_id and args.target_video_id not in ("best", "highest"):
            v_a = next((v for v in videos if v["video_id"] == args.target_video_id), None)
        if not v_a:
            sorted_views = sorted(videos, key=lambda x: int(x.get("view_count", 0) or 0), reverse=True)
            v_a = sorted_views[0] if sorted_views else None

        if args.benchmark_video_id and args.benchmark_video_id not in ("latest", "recent"):
            v_b = next((v for v in videos if v["video_id"] == args.benchmark_video_id), None)
        if not v_b:
            v_b = videos[0] if videos else None
            if v_b and v_a and v_b["video_id"] == v_a["video_id"] and len(videos) > 1:
                v_b = videos[1]

        if not v_a or not v_b:
            return CompareVideosDetailedOutput(success=False, error="Could not locate two valid videos to compare.")

        va_views = int(v_a.get("view_count", 0) or 0)
        vb_views = int(v_b.get("view_count", 0) or 0)
        view_diff = va_views - vb_views
        ratio = round(va_views / max(vb_views, 1), 2) if vb_views > 0 else "N/A"

        comparison = {
            "view_difference": view_diff,
            "view_ratio": f"{ratio}x" if ratio != "N/A" else "Infinity",
            "video_a_views": va_views,
            "video_b_views": vb_views,
            "video_a_likes": v_a.get("like_count"),
            "video_b_likes": v_b.get("like_count"),
            "video_a_title": v_a.get("title"),
            "video_b_title": v_b.get("title"),
        }

        summary = (
            f"Comparing '{v_a.get('title')}' ({va_views:,} views) vs '{v_b.get('title')}' ({vb_views:,} views): "
            f"'{v_a.get('title')}' has {abs(view_diff):,} {'more' if view_diff >= 0 else 'fewer'} views ({ratio}x)."
        )

        return CompareVideosDetailedOutput(
            success=True,
            video_a=v_a,
            video_b=v_b,
            metric_comparison=comparison,
            summary=summary,
        )
    except Exception as e:
        logger.warning(f"Error in tool_compare_videos_detailed: {e}")
        return CompareVideosDetailedOutput(success=False, error=str(e))


def tool_get_video_details(args: GetVideoDetailsInput, ctx: ToolContext) -> GetVideoDetailsOutput:
    try:
        from app.agent.providers.youtube import YouTubeAdapter
        yt = YouTubeAdapter()
        v = yt.fetch_video_details(args.video_id)
        if not v:
            return GetVideoDetailsOutput(success=False, video_id=args.video_id, error=f"Video '{args.video_id}' not found.")
        return GetVideoDetailsOutput(
            success=True,
            video_id=v.get("video_id", args.video_id),
            title=v.get("title", ""),
            channel_title=v.get("channel_title", ""),
            channel_id=v.get("channel_id", ""),
            view_count=v.get("view_count", 0),
            like_count=v.get("like_count"),
            comment_count=v.get("comment_count"),
            duration=v.get("duration", ""),
            duration_formatted=v.get("duration_formatted", ""),
            published_at=v.get("published_at", ""),
            url=f"https://www.youtube.com/watch?v={v.get('video_id', args.video_id)}",
            description=v.get("description", "")[:1000],
            tags=v.get("tags", [])[:15],
        )
    except Exception as e:
        logger.warning(f"Error in tool_get_video_details: {e}")
        return GetVideoDetailsOutput(success=False, video_id=args.video_id, error=str(e))


def tool_search_youtube_videos(args: SearchYouTubeVideosInput, ctx: ToolContext) -> SearchYouTubeVideosOutput:
    try:
        from app.agent.providers.youtube import YouTubeAdapter
        yt = YouTubeAdapter()
        results = yt.search_competitor_videos(args.query, max_results=args.max_results)
        return SearchYouTubeVideosOutput(
            success=True,
            query=args.query,
            videos=results,
            total_found=len(results),
        )
    except Exception as e:
        logger.warning(f"Error in tool_search_youtube_videos: {e}")
        return SearchYouTubeVideosOutput(success=False, query=args.query, error=str(e))


def tool_search_youtube_growth_strategies(args: SearchGrowthStrategiesInput, ctx: ToolContext) -> SearchGrowthStrategiesOutput:
    try:
        from app.agent.providers.tavily import TavilySearchClient
        tavily = TavilySearchClient()
        q = f"YouTube growth strategy {args.topic} {args.focus_area or ''}".strip()
        res = tavily.search(query=q, max_results=5, include_answer=True)
        return SearchGrowthStrategiesOutput(
            success=True,
            topic=args.topic,
            results=res.get("results", []),
            direct_answer=res.get("answer"),
        )
    except Exception as e:
        logger.warning(f"Error in tool_search_youtube_growth_strategies: {e}")
        return SearchGrowthStrategiesOutput(success=False, topic=args.topic, error=str(e))


def tool_research_youtube_algorithm_updates(args: ResearchAlgorithmUpdatesInput, ctx: ToolContext) -> ResearchAlgorithmUpdatesOutput:
    try:
        from app.agent.providers.tavily import TavilySearchClient
        tavily = TavilySearchClient()
        q = f"YouTube algorithm update announcement {args.query}".strip()
        res = tavily.search(query=q, max_results=5, include_answer=True)
        return ResearchAlgorithmUpdatesOutput(
            success=True,
            query=args.query,
            results=res.get("results", []),
            direct_answer=res.get("answer"),
        )
    except Exception as e:
        logger.warning(f"Error in tool_research_youtube_algorithm_updates: {e}")
        return ResearchAlgorithmUpdatesOutput(success=False, query=args.query, error=str(e))


def tool_research_niche_and_trends(args: ResearchNicheTrendsInput, ctx: ToolContext) -> ResearchNicheTrendsOutput:
    try:
        from app.agent.providers.tavily import TavilySearchClient
        tavily = TavilySearchClient()
        q = f"YouTube trending topics niche {args.niche} {args.topic or ''}".strip()
        res = tavily.search(query=q, max_results=5)
        return ResearchNicheTrendsOutput(
            success=True,
            niche=args.niche,
            results=res.get("results", []),
        )
    except Exception as e:
        logger.warning(f"Error in tool_research_niche_and_trends: {e}")
        return ResearchNicheTrendsOutput(success=False, niche=args.niche, error=str(e))


def tool_search_competitor_content_strategies(args: SearchCompetitorStrategiesInput, ctx: ToolContext) -> SearchCompetitorStrategiesOutput:
    try:
        from app.agent.providers.tavily import TavilySearchClient
        tavily = TavilySearchClient()
        q = f"YouTube content strategy title thumbnail {args.competitor_channel_or_niche}".strip()
        res = tavily.search(query=q, max_results=5)
        return SearchCompetitorStrategiesOutput(
            success=True,
            target=args.competitor_channel_or_niche,
            results=res.get("results", []),
        )
    except Exception as e:
        logger.warning(f"Error in tool_search_competitor_content_strategies: {e}")
        return SearchCompetitorStrategiesOutput(success=False, target=args.competitor_channel_or_niche, error=str(e))


def tool_search_external_web(args: SearchExternalWebInput, ctx: ToolContext) -> SearchExternalWebOutput:
    try:
        from app.agent.providers.tavily import TavilySearchClient
        tavily = TavilySearchClient()
        res = tavily.search(query=args.query, max_results=args.max_results, include_answer=True)
        return SearchExternalWebOutput(
            success=True,
            query=args.query,
            results=res.get("results", []),
        )
    except Exception as e:
        logger.warning(f"Error in tool_search_external_web: {e}")
        return SearchExternalWebOutput(success=False, query=args.query, error=str(e))


def tool_get_channel_details(args: GetChannelDetailsInput, ctx: ToolContext) -> GetChannelDetailsOutput:
    """Retrieve verified channel metadata and recent uploads for active or specified channel."""
    try:
        from app.database.models import ChannelConnection
        from app.agent.providers.youtube import YouTubeAdapter

        target = None
        if ctx.active_channel_id:
            if not args.channel_id or args.channel_id.strip().lower() == ctx.active_channel_id.lower() or (len(args.channel_id.strip()) == 24 and args.channel_id.strip().startswith("UC")):
                target = ctx.active_channel_id
            else:
                target = args.channel_id.strip()
        else:
            target = args.channel_id.strip() if args.channel_id else None

        if not target:
            ch_conn = ctx.db.query(ChannelConnection).filter(ChannelConnection.workspace_id == ctx.workspace_id).first()
            if ch_conn:
                target = ch_conn.channel_id
        if not target:
            from app.database.models import AnalysisReport
            latest_rep = (
                ctx.db.query(AnalysisReport)
                .filter(AnalysisReport.workspace_id == ctx.workspace_id)
                .order_by(AnalysisReport.created_at.desc())
                .first()
            )
            if latest_rep:
                target = latest_rep.channel_id

        if not target:
            return GetChannelDetailsOutput(
                success=False,
                error="No channel ID provided and no active conversation channel or connected channel found.",
            )

        yt = YouTubeAdapter()
        target_info = yt.parse_youtube_target(target)
        ch_data = yt.fetch_channel_data(target_info)
        ch_id = ch_data.get("channel_id")

        # Check OAuth connection status for this specific channel
        conn = ctx.db.query(ChannelConnection).filter(
            ChannelConnection.workspace_id == ctx.workspace_id,
            ChannelConnection.channel_id == ch_id,
        ).first()

        is_oauth = bool(conn and getattr(conn, "access_token", None))
        return GetChannelDetailsOutput(
            success=True,
            channel_id=ch_id,
            channel_name=ch_data.get("title"),
            custom_url=ch_data.get("custom_url"),
            subscriber_count=ch_data.get("subscriber_count"),
            video_count=int(ch_data.get("video_count", 0)),
            view_count=int(ch_data.get("view_count", 0)),
            description=ch_data.get("description", ""),
            recent_videos=ch_data.get("recent_videos", []),
            is_oauth_connected=is_oauth,
            source="official_youtube_api_v3",
        )
    except Exception as e:
        logger.warning(f"Error in tool_get_channel_details: {e}")
        # Graceful fallback to active_channel_context if available
        if ctx.active_channel_context and ctx.active_channel_context.get("channel_id"):
            ctx_c = ctx.active_channel_context
            return GetChannelDetailsOutput(
                success=True,
                channel_id=ctx_c.get("channel_id"),
                channel_name=ctx_c.get("channel_name"),
                custom_url=ctx_c.get("custom_url"),
                subscriber_count=ctx_c.get("subscriber_count"),
                video_count=int(ctx_c.get("video_count", 0)),
                view_count=int(ctx_c.get("view_count", 0)),
                description=ctx_c.get("description", ""),
                recent_videos=[],
                is_oauth_connected=False,
                source="active_conversation_context_fallback",
            )
        return GetChannelDetailsOutput(success=False, error=str(e))


def tool_investigate_creator_struggles(args: InvestigateStrugglesInput, ctx: ToolContext) -> InvestigateStrugglesOutput:
    """Execute multi-source investigation across YouTube, Reddit, X, and Articles via UniversalStrugglesEngine."""
    try:
        from app.services.creator_struggles import UniversalStrugglesEngine
        engine = UniversalStrugglesEngine()
        res = engine.investigate(user_query=args.query, force_refresh=args.force_refresh, db=ctx.db)
        return InvestigateStrugglesOutput(
            success=True,
            query=args.query,
            analysis=res.get("analysis", {}),
            sources=res.get("sources", {}),
            total_evidence_count=res.get("total_evidence_count", 0),
        )
    except Exception as e:
        logger.warning(f"Error in tool_investigate_creator_struggles: {e}")
        return InvestigateStrugglesOutput(success=False, query=args.query, error=str(e))


def tool_search_niche_radar(args: SearchNicheRadarInput, ctx: ToolContext) -> SearchNicheRadarOutput:
    """Search and retrieve real YouTube videos and performance metrics for a specific niche."""
    try:
        from app.services.niche_radar import NicheRadarService
        res = NicheRadarService.get_or_search_niche_radar(
            workspace_id=ctx.workspace_id,
            niche=args.niche,
            time_period=args.time_period,
            category=args.category,
            db=ctx.db,
        )
        return SearchNicheRadarOutput(
            success=True,
            niche=args.niche,
            videos=res.get("videos", [])[:10],
            total_count=res.get("total_count", 0),
            is_cached=res.get("is_cached", False),
        )
    except Exception as e:
        logger.warning(f"Error in tool_search_niche_radar: {e}")
        return SearchNicheRadarOutput(success=False, niche=args.niche, error=str(e))


# -------------------------------------------------------------
# Tool Registry & JSON Schema Definitions
# -------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "investigate_creator_struggles": {
        "func": tool_investigate_creator_struggles,
        "input_model": InvestigateStrugglesInput,
        "read_only": True,
        "description": "Investigate common YouTube creator struggles (e.g. low CTR, retention drops, shorts stalled at 0 views) across YouTube tutorials, Reddit discussions, X posts, and creator guides.",
    },
    "search_niche_radar": {
        "func": tool_search_niche_radar,
        "input_model": SearchNicheRadarInput,
        "read_only": True,
        "description": "Search and rank emerging niche topics, candidate opportunities, and video velocities using verified YouTube observations.",
    },
    "get_video_details": {
        "func": tool_get_video_details,
        "input_model": GetVideoDetailsInput,
        "read_only": True,
        "description": "Retrieve comprehensive public metadata, lifetime view count, like count, comment count, duration, and tags for a specific video ID.",
    },
    "search_youtube_videos": {
        "func": tool_search_youtube_videos,
        "input_model": SearchYouTubeVideosInput,
        "read_only": True,
        "description": "Search public YouTube videos for a specific topic, keyword, or competitor channel query, returning views, channel, and title.",
    },
    "search_youtube_growth_strategies": {
        "func": tool_search_youtube_growth_strategies,
        "input_model": SearchGrowthStrategiesInput,
        "read_only": True,
        "description": "Research proven YouTube growth techniques, retention hooks, thumbnail packaging, titles, and audience engagement strategies via web research.",
    },
    "research_youtube_algorithm_updates": {
        "func": tool_research_youtube_algorithm_updates,
        "input_model": ResearchAlgorithmUpdatesInput,
        "read_only": True,
        "description": "Research current official YouTube algorithm announcements, documented platform changes, and Creator Insider updates.",
    },
    "research_niche_and_trends": {
        "func": tool_research_niche_and_trends,
        "input_model": ResearchNicheTrendsInput,
        "read_only": True,
        "description": "Research YouTube topics, niche demand, audience interest, and emerging sub-niches using real-time search.",
    },
    "search_competitor_content_strategies": {
        "func": tool_search_competitor_content_strategies,
        "input_model": SearchCompetitorStrategiesInput,
        "read_only": True,
        "description": "Research competitor content approaches, packaging angles, title structures, and thumbnail concepts.",
    },
    "search_external_web": {
        "func": tool_search_external_web,
        "input_model": SearchExternalWebInput,
        "read_only": True,
        "description": "Search the broader web and creator forums (Reddit, blogs) for creator discussions and tutorials.",
    },
    "resolve_youtube_target": {
        "func": tool_resolve_youtube_target,
        "input_model": ResolveTargetInput,
        "read_only": True,
        "description": "Resolve a YouTube video URL, channel URL, handle, or uploaded file into canonical channel and video identifiers.",
    },
    "fetch_youtube_metadata": {
        "func": tool_fetch_youtube_metadata,
        "input_model": FetchMetadataInput,
        "read_only": True,
        "description": "Fetch verified YouTube video or channel metadata, lifetime statistics, and recent uploads from YouTube Data API v3.",
    },
    "search_competitor_videos": {
        "func": tool_search_competitor_videos,
        "input_model": SearchCompetitorsInput,
        "read_only": True,
        "description": "Search YouTube for candidate competing videos covering a specific query or topic keyword.",
    },
    "retrieve_video_transcript": {
        "func": tool_retrieve_video_transcript,
        "input_model": RetrieveTranscriptInput,
        "read_only": True,
        "description": "Retrieve transcript text from an uploaded file or check public availability.",
    },
    "calculate_video_metrics": {
        "func": tool_calculate_video_metrics,
        "input_model": CalculateMetricsInput,
        "read_only": True,
        "description": "Calculate views/day velocity, age in days, and rank top high-performing competitor videos.",
    },
    "compare_videos_matrix": {
        "func": tool_compare_videos_matrix,
        "input_model": CompareVideosMatrixInput,
        "read_only": True,
        "description": "Build a documented side-by-side comparison matrix between submitted video and competitor benchmarks.",
    },
    "extract_keywords_and_hashtags": {
        "func": tool_extract_keywords_and_hashtags,
        "input_model": ExtractKeywordsInput,
        "read_only": True,
        "description": "Extract observed title keywords, description terms, and hashtag frequencies from analyzed targets.",
    },
    "identify_content_gaps": {
        "func": tool_identify_content_gaps,
        "input_model": IdentifyGapsInput,
        "read_only": True,
        "description": "Identify evidence-grounded content gaps across title length, keyword coverage, chapter markers, and description depth.",
    },
    "generate_prioritized_recommendations": {
        "func": tool_generate_prioritized_recommendations,
        "input_model": GenerateRecommendationsInput,
        "read_only": True,
        "description": "Generate a prioritized, sequenced action plan based on verified gaps.",
    },
    "retrieve_creator_memories": {
        "func": tool_retrieve_creator_memories,
        "input_model": RetrieveMemoriesInput,
        "read_only": True,
        "description": "Retrieve relevant creator profile preferences, historical benchmarks, and verified topic insights.",
    },
    "save_analysis_results_and_memories": {
        "func": tool_save_analysis_results_and_memories,
        "input_model": SaveResultsInput,
        "read_only": False,
        "description": "Persist final validated report, evidence records, content opportunities, and eligible memories to database.",
    },
    "get_latest_analysis_report": {
        "func": tool_get_latest_analysis_report,
        "input_model": GetLatestReportInput,
        "read_only": True,
        "description": "Retrieve the latest completed YouTube analysis report, competitor benchmarks, content gaps, and recommendations for the current channel.",
    },
    "get_connected_channel_summary": {
        "func": tool_get_connected_channel_summary,
        "input_model": GetChannelSummaryInput,
        "read_only": True,
        "description": "Retrieve the current connected channel's lifetime statistics (subscribers, views, upload count) and recent video uploads.",
    },
    "get_channel_video_inventory": {
        "func": tool_get_channel_video_inventory,
        "input_model": GetChannelInventoryInput,
        "read_only": True,
        "description": "Fetch real channel videos inventory (up to 50 uploads per page) with verified view counts, like counts, and published dates.",
    },
    "rank_and_calculate_video_metrics": {
        "func": tool_rank_and_calculate_video_metrics,
        "input_model": RankVideosInput,
        "read_only": True,
        "description": "Deterministically rank channel videos by views, engagement_rate, views_per_day velocity, or likes, returning winner and ranking breakdown.",
    },
    "compare_videos_detailed": {
        "func": tool_compare_videos_detailed,
        "input_model": CompareVideosDetailedInput,
        "read_only": True,
        "description": "Compare two specific channel videos (e.g. best video vs latest upload), calculating exact view gaps, ratios, and velocities.",
    },
    "get_channel_details": {
        "func": tool_get_channel_details,
        "input_model": GetChannelDetailsInput,
        "read_only": True,
        "description": "Retrieve verified public YouTube channel details (channel name, custom URL, subscriber count, total uploads, description, and recent uploads) for the active channel or specified channel ID.",
    },
}


def get_agent_tool_definitions() -> list[dict[str, Any]]:
    """Generate OpenAI/OpenRouter-compatible function tool definitions."""
    tools = []
    for name, meta in TOOL_REGISTRY.items():
        schema = meta["input_model"].model_json_schema()
        # Clean schema for function calling
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta["description"],
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        })
    return tools


def execute_tool(tool_name: str, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Execute permitted tool call with strict argument validation and error containment."""
    if tool_name not in TOOL_REGISTRY:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' is not permitted or recognized. Allowed tools: {list(TOOL_REGISTRY.keys())}",
        }

    meta = TOOL_REGISTRY[tool_name]
    start_time = time.time()
    try:
        # Validate arguments through Pydantic
        validated_args = meta["input_model"](**arguments)
        result_model = meta["func"](validated_args, context)
        duration_ms = (time.time() - start_time) * 1000
        res_dict = result_model.model_dump()
        res_dict["_duration_ms"] = round(duration_ms, 2)
        return res_dict
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(f"Tool '{tool_name}' execution failed: {exc}")
        return {
            "success": False,
            "error": f"Validation or execution error in {tool_name}: {str(exc)}",
            "_duration_ms": round(duration_ms, 2),
        }
