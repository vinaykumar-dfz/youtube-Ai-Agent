import time
from typing import Any
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.database.models import AnalysisJob, Workspace
from app.agent.state import (
    AnalysisWorkflowState,
    WorkflowStage,
    can_transition,
)
from app.agent.tools import (
    ToolContext,
    execute_tool,
    get_agent_tool_definitions,
)
from app.agent.memory import CreatorMemoryService
from app.services.competitor_analyzer import CompetitorAnalyzer
from app.services.analytics import AnalyticsService
from app.agent.providers.llm import OpenRouterLLMClient
from app.services.job_management import JobManagementService


class AgentOrchestrator:
    """Stateful, memory-enabled, tool-using agentic orchestrator for YouTube growth and competitor analysis."""

    def __init__(self, db: Session):
        self.db = db
        self.job_service = JobManagementService(db)
        self.memory_service = CreatorMemoryService(db)
        self.llm = OpenRouterLLMClient()
        self.tool_definitions = get_agent_tool_definitions()

    def execute_workflow(self, job_id: str) -> None:
        """Execute or resume the stateful, memory-guided agentic workflow."""
        job = self.db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return

        workspace_id = job.workspace_id
        target = job.input_target

        # Check for checkpoint recovery
        state = self._initialize_or_resume_state(job)
        ctx = ToolContext(self.db, workspace_id, job.id, user_id=state.user_id)

        try:
            # -------------------------------------------------------------
            # Stage 1: RESOLVING_INPUT
            # -------------------------------------------------------------
            if state.stage in (WorkflowStage.RECEIVED, WorkflowStage.RESOLVING_INPUT):
                self._transition_and_checkpoint(job, state, WorkflowStage.RESOLVING_INPUT)
                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    f"Agent initialized. Resolving target identity for: {target}",
                )

                # Tool 1: Resolve target
                resolve_res = self._run_tool(state, "resolve_youtube_target", {"target": target}, ctx)
                if not resolve_res.get("success"):
                    raise ValueError(resolve_res.get("error", "Failed to resolve target identity"))

                # Tool 2: Fetch metadata
                meta_res = self._run_tool(state, "fetch_youtube_metadata", {"target": target}, ctx)
                if not meta_res.get("success"):
                    raise ValueError(meta_res.get("error", "Failed to fetch YouTube metadata"))

                state.resolved_target = meta_res.get("video_data")
                state.channel_info = meta_res.get("channel_info")
                state.metrics = meta_res.get("metrics_data", {})

                v_title = state.resolved_target.get("title", "Submitted Video")
                ch_title = state.channel_info.get("title", "Channel")
                v_views = state.resolved_target.get("view_count", 0)

                state.completed_steps.append("RESOLVE_METADATA")
                state.short_term_memory.add_observation(
                    f"Target resolved: '{v_title}' on channel '{ch_title}' ({v_views:,} views)"
                )
                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    f"Resolved target: '{v_title}' on {ch_title} ({v_views:,} views). Verified metadata ingested.",
                )

            # -------------------------------------------------------------
            # Stage 2: PLANNING (Memory Retrieval & Research Plan Formulation)
            # -------------------------------------------------------------
            if state.stage in (WorkflowStage.RESOLVING_INPUT, WorkflowStage.PLANNING):
                self._transition_and_checkpoint(job, state, WorkflowStage.PLANNING)

                # Retrieve relevant long-term creator memories
                v_title = state.resolved_target.get("title", "") if state.resolved_target else target
                mem_res = self._run_tool(
                    state,
                    "retrieve_creator_memories",
                    {"query": f"{v_title} {target}", "categories": ["creator_profile", "episodic", "semantic"]},
                    ctx,
                )
                memory_context = mem_res.get("formatted_context", "")
                state.memory_references = mem_res.get("memories", [])
                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    f"Retrieved {len(state.memory_references)} relevant creator memories and channel preferences.",
                )

                # LLM Research Plan Generation
                plan = self._formulate_research_plan(state, memory_context)
                state.research_plan = plan
                state.completed_steps.append("FORMULATE_PLAN")
                primary_topic = plan.get("primary_topic", v_title[:50] or "youtube growth")
                state.short_term_memory.add_observation(f"Primary Topic formulated: '{primary_topic}'")

                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    f"Research Plan established: Focus Topic '{primary_topic}'. Search intent: {plan.get('search_intent', 'Educational')}.",
                )

            # -------------------------------------------------------------
            # Stage 3: COLLECTING_EVIDENCE (Tool-Calling Bounded Decision Loop)
            # -------------------------------------------------------------
            if state.stage in (WorkflowStage.PLANNING, WorkflowStage.COLLECTING_EVIDENCE):
                self._transition_and_checkpoint(job, state, WorkflowStage.COLLECTING_EVIDENCE)

                primary_topic = state.research_plan.get("primary_topic") if isinstance(state.research_plan, dict) else ""
                if not primary_topic and state.resolved_target:
                    primary_topic = state.resolved_target.get("title", "")[:50]

                # 1. Search candidate competitors
                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    f"Agent querying YouTube Data API v3 for high-velocity competing videos on: '{primary_topic}'...",
                )
                search_res = self._run_tool(
                    state,
                    "search_competitor_videos",
                    {"query": primary_topic, "max_results": 15},
                    ctx,
                )
                candidates = search_res.get("candidate_videos", [])
                state.competitor_candidates = candidates

                # 2. Check for optional transcript
                if state.resolved_target and state.resolved_target.get("video_id"):
                    v_id = state.resolved_target["video_id"]
                    t_res = self._run_tool(state, "retrieve_video_transcript", {"video_id": v_id}, ctx)
                    if t_res.get("has_transcript"):
                        state.short_term_memory.add_observation(f"Transcript excerpt available: {t_res.get('summary_text')[:120]}")

                # 3. Calculate reproducible velocity metrics & select top competitors
                calc_res = self._run_tool(
                    state,
                    "calculate_video_metrics",
                    {"submitted_video": state.resolved_target, "candidate_videos": candidates, "max_competitors": 5},
                    ctx,
                )
                state.selected_competitors = calc_res.get("selected_competitors", [])
                state.completed_steps.append("COLLECT_COMPETITORS")

                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    f"Benchmarked {len(state.selected_competitors)} verified topical competitors ranked by views/day velocity.",
                )

                # Agent Reflection Loop: Evaluate if more evidence is needed
                more_needed = self._evaluate_evidence_sufficiency(state)
                if more_needed and state.tool_call_count < state.max_tool_calls:
                    self.job_service.add_event(
                        job.id,
                        workspace_id,
                        state.get_frontend_stage(),
                        "Agent reflection: Requesting supplementary competitor search with refined keyword query...",
                    )
                    alt_q = f"{primary_topic} tutorial"
                    supp_res = self._run_tool(
                        state,
                        "search_competitor_videos",
                        {"query": alt_q, "max_results": 8},
                        ctx,
                    )
                    seen_ids = {c.get("video_id") for c in state.competitor_candidates}
                    for sc in supp_res.get("candidate_videos", []):
                        if sc.get("video_id") not in seen_ids:
                            state.competitor_candidates.append(sc)
                    # Recalculate
                    calc_res2 = self._run_tool(
                        state,
                        "calculate_video_metrics",
                        {"submitted_video": state.resolved_target, "candidate_videos": state.competitor_candidates, "max_competitors": 5},
                        ctx,
                    )
                    state.selected_competitors = calc_res2.get("selected_competitors", state.selected_competitors)

            # -------------------------------------------------------------
            # Stage 4: ANALYZING (Matrices, Keywords, Hashtags, and Content Gaps)
            # -------------------------------------------------------------
            if state.stage in (WorkflowStage.COLLECTING_EVIDENCE, WorkflowStage.ANALYZING):
                self._transition_and_checkpoint(job, state, WorkflowStage.ANALYZING)
                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    "Synthesizing side-by-side matrices, keyword distribution, and hashtag gap metrics...",
                )

                # Tool 6: Compare videos matrix
                matrix_res = self._run_tool(
                    state,
                    "compare_videos_matrix",
                    {"submitted_video": state.resolved_target, "selected_competitors": state.selected_competitors},
                    ctx,
                )
                matrix = matrix_res.get("side_by_side_matrix", [])

                # Tool 7: Extract keywords & hashtags
                kw_res = self._run_tool(
                    state,
                    "extract_keywords_and_hashtags",
                    {"submitted_video": state.resolved_target, "selected_competitors": state.selected_competitors},
                    ctx,
                )
                kw_table = kw_res.get("keyword_analysis", {})
                ht_data = kw_res.get("hashtag_analysis", {})

                # Tool 8: Identify content gaps
                gap_res = self._run_tool(
                    state,
                    "identify_content_gaps",
                    {
                        "submitted_video": state.resolved_target,
                        "selected_competitors": state.selected_competitors,
                        "keyword_table": kw_table,
                        "hashtag_data": ht_data,
                    },
                    ctx,
                )
                state.gaps = gap_res.get("gaps", [])

                # Tool 9: Generate recommendations
                rec_res = self._run_tool(
                    state,
                    "generate_prioritized_recommendations",
                    {"gaps": state.gaps},
                    ctx,
                )
                state.recommendations = rec_res.get("action_plan", [])
                state.completed_steps.append("ANALYZE_GAPS")

                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    f"Identified {len(state.gaps)} verified gaps and sequenced {len(state.recommendations)} actionable recommendations.",
                )

            # -------------------------------------------------------------
            # Stage 5: VERIFYING & COMPLETED (Evidence Grounding, Memory Extraction & Persistence)
            # -------------------------------------------------------------
            if state.stage in (WorkflowStage.ANALYZING, WorkflowStage.VERIFYING):
                self._transition_and_checkpoint(job, state, WorkflowStage.VERIFYING)
                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    "Verifying epistemic grounding of recommendations and extracting reusable creator memories...",
                )

                ch_name = state.channel_info.get("title", "Creator Channel") if state.channel_info else "Creator Channel"
                ch_id = state.channel_info.get("channel_id") if state.channel_info else f"UC_{abs(hash(ch_name)):x}"

                # Compute clean analytics
                raw_metrics = state.metrics or {}
                analytics = AnalyticsService.compute_channel_analytics(raw_metrics)

                topic_intent = (
                    state.research_plan
                    if isinstance(state.research_plan, dict) and "primary_topic" in state.research_plan
                    else {
                        "primary_topic": state.resolved_target.get("title", "")[:40] if state.resolved_target else "General Topic",
                        "search_intent": "Educational",
                        "intended_audience": "Target YouTube Viewers",
                    }
                )

                comp_intel = {
                    "topic_intent": topic_intent,
                    "selected_competitors": state.selected_competitors,
                    "side_by_side_matrix": matrix if 'matrix' in locals() else [],
                    "keyword_analysis": kw_table if 'kw_table' in locals() else {},
                    "hashtag_analysis": ht_data if 'ht_data' in locals() else {},
                    "gap_analysis": state.gaps,
                    "action_plan": state.recommendations,
                }

                metrics_summary = {
                    **analytics.get("summary_cards", {}),
                    "channel_info": state.channel_info,
                    "competitor_intelligence": comp_intel,
                }

                # Extract eligible memories (verified observations only)
                eligible_memories = []
                # 1. Topic niche memory
                eligible_memories.append({
                    "category": "creator_profile",
                    "memory_type": "channel_niche",
                    "content": f"Channel operates in niche: {topic_intent.get('primary_topic')}",
                    "confidence_score": 0.95,
                    "verification_status": "creator_stated",
                })
                # 2. Competitor benchmark observation
                if state.selected_competitors:
                    top_c = state.selected_competitors[0]
                    eligible_memories.append({
                        "category": "episodic",
                        "memory_type": "top_competitor_benchmark",
                        "content": f"Top competitor benchmark for {topic_intent.get('primary_topic')} is '{top_c.get('title')}' by {top_c.get('channel_title')} ({int(top_c.get('views_per_day', 0)):,} views/day).",
                        "structured_payload": {"competitor_id": top_c.get("video_id"), "velocity": top_c.get("views_per_day")},
                        "confidence_score": 1.0,
                        "verification_status": "verified_observation",
                    })
                # 3. Actionable gap insight
                if state.gaps:
                    top_gap = state.gaps[0]
                    eligible_memories.append({
                        "category": "semantic",
                        "memory_type": "verified_content_gap",
                        "content": f"Observed recurring gap in {topic_intent.get('primary_topic')}: {top_gap.get('title')}. Action: {top_gap.get('recommended_action')}",
                        "structured_payload": top_gap,
                        "confidence_score": 0.9,
                        "verification_status": "verified_observation",
                    })

                # Tool 11: Save validated results & memories
                save_res = self._run_tool(
                    state,
                    "save_analysis_results_and_memories",
                    {
                        "channel_name": ch_name,
                        "channel_id": ch_id or f"UC_{abs(hash(ch_name)):x}",
                        "analyzed_video": state.resolved_target or {},
                        "selected_competitors": state.selected_competitors,
                        "gaps": state.gaps,
                        "action_plan": state.recommendations,
                        "topic_intent": topic_intent,
                        "metrics_summary": metrics_summary,
                        "eligible_memories": eligible_memories,
                    },
                    ctx,
                )

                if not save_res.get("success"):
                    raise ValueError(save_res.get("error", "Failed to save analysis results"))

                report_id = save_res.get("report_id")

                self._transition_and_checkpoint(job, state, WorkflowStage.COMPLETED)
                self.job_service.add_event(
                    job.id,
                    workspace_id,
                    state.get_frontend_stage(),
                    f"Analysis Complete! Saved report, {save_res.get('evidence_count')} evidence records, {save_res.get('opportunities_count')} opportunities, and {save_res.get('memories_saved_count')} eligible memories.",
                )
                self.job_service.update_job_progress(
                    job.id,
                    state.get_frontend_stage(),
                    100,
                    status="succeeded",
                    result_summary={"report_id": report_id, "channel_title": ch_name},
                )
                logger.info(f"Agent workflow for Job {job.id} completed successfully.")

        except Exception as e:
            logger.exception(f"Job {job.id} encountered error in stage {state.stage}: {e}")
            state.errors.append(str(e))
            self._transition_and_checkpoint(job, state, WorkflowStage.FAILED)
            self.job_service.add_event(
                job.id,
                workspace_id,
                state.get_frontend_stage(),
                f"Workflow execution stopped: {str(e)}",
                level="error",
            )
            self.job_service.update_job_progress(
                job.id,
                state.get_frontend_stage(),
                state.get_progress_percent(),
                status="failed",
                error_message=str(e),
            )

    def _initialize_or_resume_state(self, job: AnalysisJob) -> AnalysisWorkflowState:
        """Load from existing checkpoint if available or initialize brand new state."""
        if job.checkpoint_data and isinstance(job.checkpoint_data, dict):
            try:
                state = AnalysisWorkflowState.from_dict(job.checkpoint_data)
                # If was failed or completed, start fresh
                if state.stage not in (WorkflowStage.COMPLETED, WorkflowStage.FAILED, WorkflowStage.CANCELLED):
                    logger.info(f"Resuming Job {job.id} from checkpoint at stage {state.stage}")
                    return state
            except Exception as e:
                logger.warning(f"Failed to restore checkpoint for job {job.id}: {e}")

        # New state
        return AnalysisWorkflowState(
            job_id=job.id,
            workspace_id=job.workspace_id,
            input_type=job.input_type,
            input_target=job.input_target,
        )

    def _transition_and_checkpoint(
        self,
        job: AnalysisJob,
        state: AnalysisWorkflowState,
        new_stage: WorkflowStage,
    ) -> None:
        """Validate state transition, update job progress, and persist checkpoint."""
        if state.stage != new_stage:
            state.transition_to(new_stage)

        job.current_stage = state.get_frontend_stage()
        job.progress_percent = state.get_progress_percent()
        job.checkpoint_data = state.to_dict()

        if new_stage == WorkflowStage.COMPLETED:
            job.status = "succeeded"
        elif new_stage == WorkflowStage.FAILED:
            job.status = "failed"
        elif job.status == "queued":
            job.status = "running"

        self.db.commit()

    def _run_tool(
        self,
        state: AnalysisWorkflowState,
        tool_name: str,
        args: dict[str, Any],
        ctx: ToolContext,
    ) -> dict[str, Any]:
        """Execute tool, measure duration, update short-term state and history."""
        t0 = time.time()
        res = execute_tool(tool_name, args, ctx)
        duration_ms = (time.time() - t0) * 1000
        state.record_tool_call(tool_name, args, res, duration_ms)
        return res

    def _formulate_research_plan(
        self,
        state: AnalysisWorkflowState,
        memory_context: str,
    ) -> dict[str, Any]:
        """Ask LLM to formulate structured research plan incorporating creator memories."""
        target_v = state.resolved_target or {}
        v_title = target_v.get("title", "")
        v_desc = (target_v.get("description", "") or "")[:500]

        # Use CompetitorAnalyzer fallback extraction as baseline
        fallback = CompetitorAnalyzer.analyze_topic_and_intent(target_v)

        if not self.llm.is_configured:
            return fallback

        system_prompt = (
            "You are the GrowTube AI Agent Architect. You are formulating an evidence-grounded research plan.\n"
            "Analyze the target video and the historical creator memories below. "
            "Formulate the primary topic, search intent, intended audience, and 2-3 focused competitor search queries.\n\n"
            f"Creator Memories:\n{memory_context}\n\n"
            "Return valid JSON matching this schema:\n"
            "{\n"
            '  "primary_topic": "string",\n'
            '  "search_intent": "string",\n'
            '  "intended_audience": "string",\n'
            '  "search_queries": ["query1", "query2"]\n'
            "}"
        )
        user_prompt = f"Video Title: {v_title}\nDescription excerpt: {v_desc}"

        raw = self.llm.chat_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.2, max_tokens=350)

        if not raw:
            return fallback

        try:
            import json
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(clean)
            return {
                **fallback,
                "primary_topic": parsed.get("primary_topic") or fallback["primary_topic"],
                "search_intent": parsed.get("search_intent") or fallback["search_intent"],
                "intended_audience": parsed.get("intended_audience") or fallback["intended_audience"],
                "search_queries": parsed.get("search_queries") or [fallback["primary_topic"]],
            }
        except Exception:
            return fallback

    def _evaluate_evidence_sufficiency(self, state: AnalysisWorkflowState) -> bool:
        """Evaluate whether candidate evidence is sufficient or if an additional research query is warranted."""
        # If fewer than 2 competitors found, additional search is needed
        if len(state.selected_competitors) < 2:
            return True

        if not self.llm.is_configured:
            return False

        # Ask LLM if research is sufficient
        summary = state.compact_summary_for_llm()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the GrowTube Evidence Evaluator. Inspect the current research summary.\n"
                    "Determine whether additional competitor research is required to draw valid conclusions.\n"
                    'Respond with JSON: {"needs_more_research": true/false, "rationale": "short explanation"}'
                ),
            },
            {"role": "user", "content": summary},
        ]
        resp = self.llm.chat_completion(messages, temperature=0.1, max_tokens=150)
        if resp and '"needs_more_research": true' in resp.lower():
            return True
        return False
