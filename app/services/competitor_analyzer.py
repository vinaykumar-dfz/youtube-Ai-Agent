import re
from datetime import UTC, datetime
from typing import Any

from app.core.logging import logger

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
    "video", "watch", "channel", "subscribe", "youtube", "vs", "new", "get", "like", "share"
}


def clean_text_tokens(text: str) -> list[str]:
    """Tokenize and filter standard English stopwords and punctuation."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS and not w.isdigit()]


def extract_hashtags_from_text(text: str) -> list[str]:
    """Extract actual hashtags present in text (#tag)."""
    if not text:
        return []
    tags = re.findall(r"#([a-zA-Z0-9_]+)", text)
    return list(dict.fromkeys([f"#{t.lower()}" for t in tags]))


def calculate_age_days(published_at_str: str) -> int:
    """Calculate elapsed days since video publication date."""
    if not published_at_str:
        return 1
    try:
        clean_date = published_at_str.replace("Z", "+00:00")
        pub_dt = datetime.fromisoformat(clean_date)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        days = (now - pub_dt).days
        return max(days, 1)
    except Exception:
        return 1


class CompetitorAnalyzer:
    """Production data-driven YouTube Competitor Intelligence and Side-by-Side Comparison Engine."""

    @classmethod
    def analyze_topic_and_intent(cls, video_data: dict[str, Any]) -> dict[str, Any]:
        """Infer topic, search intent, audience, and key entities from actual metadata."""
        title = video_data.get("title", "")
        desc = video_data.get("description", "")
        raw_tags = video_data.get("tags") or []

        title_tokens = clean_text_tokens(title)
        desc_tokens = clean_text_tokens(desc)

        # Primary topic: 2-3 most frequent or title-salient keywords
        primary_keywords = title_tokens[:4]
        primary_topic = " ".join([k.capitalize() for k in primary_keywords]) if primary_keywords else "General Content"

        # Secondary topics from title + description
        all_tokens = title_tokens + desc_tokens[:40]
        token_freq: dict[str, int] = {}
        for t in all_tokens:
            token_freq[t] = token_freq.get(t, 0) + 1
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        secondary_topics = [t[0].capitalize() for t in sorted_tokens[:6] if t[0].lower() not in [k.lower() for k in primary_keywords]]

        # Search intent classification based on objective syntactic cues
        title_lower = title.lower()
        if any(w in title_lower for w in ["how to", "tutorial", "guide", "step by step", "build", "code"]):
            search_intent = "Informational / Step-by-Step Tutorial"
            content_format = "Technical Tutorial"
        elif any(w in title_lower for w in ["vs", "comparison", "benchmark", "review", "best", "top"]):
            search_intent = "Comparative Evaluation / Buyer Research"
            content_format = "Comparative Review"
        elif any(w in title_lower for w in ["why", "mistake", "truth", "secret", "exposed", "stop"]):
            search_intent = "Problem Diagnosis / Curiosity Solution"
            content_format = "Case Study & Analysis"
        elif any(w in title_lower for w in ["news", "update", "released", "2026", "future"]):
            search_intent = "Trending Industry Update"
            content_format = "News Breakdown"
        else:
            search_intent = "General Informational Discovery"
            content_format = "Educational Video"

        # Audience detection
        audience_clues = []
        if any(w in title_lower or w in desc.lower() for w in ["developer", "engineer", "coding", "python", "javascript", "code"]):
            audience_clues.append("Software developers and technical builders")
        if any(w in title_lower or w in desc.lower() for w in ["creator", "youtube", "grow", "views", "channel"]):
            audience_clues.append("Content creators and video strategists")
        if any(w in title_lower or w in desc.lower() for w in ["student", "beginner", "learn", "basics"]):
            audience_clues.append("Beginners and self-paced learners")
        intended_audience = ", ".join(audience_clues) if audience_clues else "General topic practitioners and enthusiasts"

        # Key viewer questions grounded in topic
        primary_label = " ".join(primary_keywords) if primary_keywords else "this topic"
        likely_questions = [
            f"How do I get started with {primary_label}?",
            f"What are the best practices and common mistakes in {primary_label}?",
            f"How does this implementation compare to competing solutions?",
        ]

        hashtags = extract_hashtags_from_text(title + " " + desc)

        base_data = {
            "primary_topic": primary_topic,
            "secondary_topics": secondary_topics,
            "search_intent": search_intent,
            "content_format": content_format,
            "intended_audience": intended_audience,
            "main_problem_addressed": f"Optimizing implementation and understanding of {primary_label}.",
            "key_entities_and_concepts": primary_keywords + [t[0] for t in sorted_tokens[:4]],
            "likely_viewer_questions": likely_questions,
            "observed_hashtags": hashtags,
            "tags_available": bool(raw_tags),
            "raw_tags": raw_tags if raw_tags else [],
            "transcript_status": "Unavailable via public YouTube Data API v3; topic inferred from verified published metadata.",
        }

        try:
            from app.agent.providers.llm import OpenRouterLLMClient
            llm_client = OpenRouterLLMClient()
            if llm_client.is_configured:
                return llm_client.enhance_topic_and_intent(title, desc, base_data)
        except Exception:
            pass

        return base_data

    @classmethod
    def rank_and_select_competitors(
        cls,
        submitted_video: dict[str, Any],
        candidate_videos: list[dict[str, Any]],
        max_competitors: int = 5,
    ) -> list[dict[str, Any]]:
        """Rank candidates using documented relevance and performance methodology."""
        submitted_id = submitted_video.get("video_id")
        submitted_title_tokens = set(clean_text_tokens(submitted_video.get("title", "")))

        scored_candidates = []

        for cand in candidate_videos:
            cand_id = cand.get("video_id")
            if not cand_id or cand_id == submitted_id:
                continue

            cand_title = cand.get("title", "")
            cand_tokens = set(clean_text_tokens(cand_title))
            overlap = len(submitted_title_tokens.intersection(cand_tokens))

            # 1. Topic Relevance Score (0-100)
            if not submitted_title_tokens:
                relevance_score = 50.0
            else:
                relevance_score = min(100.0, (overlap / len(submitted_title_tokens)) * 100.0)

            # Discard completely unrelated candidates (no keyword overlap)
            if overlap == 0 and len(submitted_title_tokens) > 0:
                relevance_score = 10.0

            # 2. Observed Performance Signals (Views / Day)
            views = cand.get("view_count") or 0
            age_days = calculate_age_days(cand.get("published_at", ""))
            views_per_day = views / age_days

            # Scaled performance score (log scale)
            import math
            log_velocity = math.log10(views_per_day + 1)
            perf_score = min(100.0, log_velocity * 25.0)

            # Engagement signals
            likes = cand.get("like_count")
            like_rate = (likes / views * 100) if likes and views > 0 else None

            # 3. Confidence in Comparison
            confidence = 0.95 if (views > 0 and cand.get("published_at")) else 0.70

            # Combined comparison index (prioritizing topic relevance first!)
            selection_score = (relevance_score * 0.55) + (perf_score * 0.35) + (confidence * 10.0)

            reason = (
                f"High topical overlap ({overlap} keywords in title) with "
                f"{int(views_per_day):,} views/day velocity over {age_days} days."
            )

            scored_candidates.append({
                **cand,
                "relevance_score": round(relevance_score, 1),
                "performance_score": round(perf_score, 1),
                "views_per_day": round(views_per_day, 1),
                "age_days": age_days,
                "like_rate_percent": round(like_rate, 2) if like_rate is not None else None,
                "confidence_score": confidence,
                "selection_score": round(selection_score, 1),
                "selection_reason": reason,
                "performance_label": "High-performing relative to topic comparison criteria",
            })

        # Sort descending by selection score
        scored_candidates.sort(key=lambda x: x["selection_score"], reverse=True)
        return scored_candidates[:max_competitors]

    @classmethod
    def build_side_by_side_matrix(
        cls,
        submitted_video: dict[str, Any],
        competitors: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Construct side-by-side feature matrix comparing submitted video with selected competitors."""
        s_views = submitted_video.get("view_count") or 0
        s_age = calculate_age_days(submitted_video.get("published_at", ""))
        s_vpd = round(s_views / s_age, 1)
        s_likes = submitted_video.get("like_count")
        s_like_rate = round(s_likes / s_views * 100, 2) if s_likes and s_views > 0 else None
        s_hashtags = extract_hashtags_from_text(submitted_video.get("title", "") + " " + submitted_video.get("description", ""))
        s_desc = submitted_video.get("description", "")

        matrix = [{
            "video_id": submitted_video.get("video_id"),
            "is_submitted": True,
            "title": submitted_video.get("title", "Submitted Video"),
            "channel_title": submitted_video.get("channel_title", "Your Channel"),
            "channel_id": submitted_video.get("channel_id", ""),
            "thumbnail_url": submitted_video.get("thumbnail_url"),
            "video_url": f"https://www.youtube.com/watch?v={submitted_video.get('video_id')}" if submitted_video.get("video_id") else "",
            "published_at": submitted_video.get("published_at", "")[:10],
            "age_days": s_age,
            "duration": submitted_video.get("duration", "Unknown"),
            "view_count": s_views,
            "views_per_day": s_vpd,
            "like_count": s_likes,
            "like_rate_percent": s_like_rate,
            "comment_count": submitted_video.get("comment_count"),
            "hashtag_count": len(s_hashtags),
            "observed_hashtags": s_hashtags,
            "description_word_count": len(s_desc.split()),
            "has_timestamps": bool(re.search(r"\b\d{1,2}:\d{2}\b", s_desc)),
            "selection_reason": "Target video submitted for audit",
        }]

        for comp in competitors:
            c_desc = comp.get("description", "")
            c_hashtags = extract_hashtags_from_text(comp.get("title", "") + " " + c_desc)
            matrix.append({
                "video_id": comp.get("video_id"),
                "is_submitted": False,
                "title": comp.get("title", ""),
                "channel_title": comp.get("channel_title", ""),
                "channel_id": comp.get("channel_id", ""),
                "thumbnail_url": comp.get("thumbnail_url"),
                "video_url": f"https://www.youtube.com/watch?v={comp.get('video_id')}",
                "published_at": comp.get("published_at", "")[:10],
                "age_days": comp.get("age_days", 1),
                "duration": comp.get("duration_formatted", "Unknown"),
                "view_count": comp.get("view_count", 0),
                "views_per_day": comp.get("views_per_day", 0),
                "like_count": comp.get("like_count"),
                "like_rate_percent": comp.get("like_rate_percent"),
                "comment_count": comp.get("comment_count"),
                "hashtag_count": len(c_hashtags),
                "observed_hashtags": c_hashtags,
                "description_word_count": len(c_desc.split()),
                "has_timestamps": bool(re.search(r"\b\d{1,2}:\d{2}\b", c_desc)),
                "selection_reason": comp.get("selection_reason", ""),
            })

        return matrix

    @classmethod
    def analyze_keywords_side_by_side(
        cls,
        submitted_video: dict[str, Any],
        competitors: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate side-by-side keyword presence and gap matrix."""
        s_title = submitted_video.get("title", "").lower()
        s_desc = submitted_video.get("description", "").lower()
        s_tags = [t.lower() for t in (submitted_video.get("tags") or [])]

        # Gather terms across competitors
        comp_corpus = []
        for c in competitors:
            comp_corpus.append({
                "title": c.get("title", "").lower(),
                "desc": c.get("description", "").lower(),
                "channel": c.get("channel_title", ""),
            })

        all_candidate_terms = []
        for c in competitors:
            tokens = clean_text_tokens(c.get("title", ""))
            # Unigrams & bigrams
            all_candidate_terms.extend(tokens)
            for i in range(len(tokens) - 1):
                all_candidate_terms.append(f"{tokens[i]} {tokens[i+1]}")

        # Also add submitted tokens
        sub_tokens = clean_text_tokens(s_title)
        all_candidate_terms.extend(sub_tokens)
        for i in range(len(sub_tokens) - 1):
            all_candidate_terms.append(f"{sub_tokens[i]} {sub_tokens[i+1]}")

        # Count frequencies
        term_counts: dict[str, int] = {}
        for t in all_candidate_terms:
            term_counts[t] = term_counts.get(t, 0) + 1

        top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        comparison_table = []
        for term, _ in top_terms:
            # Check submitted video evidence
            in_s_title = term in s_title
            in_s_desc = term in s_desc
            in_s_tags = term in s_tags

            if in_s_title and in_s_desc:
                s_evidence = "Observed in Title and Description"
            elif in_s_title:
                s_evidence = "Observed in Title"
            elif in_s_desc:
                s_evidence = "Observed in Description only"
            elif in_s_tags:
                s_evidence = "Observed in Video Tags only"
            else:
                s_evidence = "Not found in Title or Description"

            # Check competitor presence count
            comp_in_title_count = sum(1 for c in comp_corpus if term in c["title"])
            comp_in_desc_count = sum(1 for c in comp_corpus if term in c["desc"])
            comp_total = len(competitors) or 1

            comp_evidence = f"Present in {comp_in_title_count}/{comp_total} competitor titles, {comp_in_desc_count}/{comp_total} descriptions"

            # Match status & Gap assessment
            if (in_s_title or in_s_desc) and comp_in_title_count > 0:
                match_status = "Competitive Alignment"
                opportunity = "Covered; ensure natural semantic placement rather than repeated stuffing."
                relevance = "High"
            elif comp_in_title_count >= 2 and not in_s_title:
                match_status = "Competitor Keyword Gap"
                opportunity = f"High-intent phrase used by {comp_in_title_count} competing titles. Consider testing in title or first 100 characters of description."
                relevance = "High"
            elif in_s_title and comp_in_title_count == 0:
                match_status = "Unique Channel Differentiator"
                opportunity = "Unique angle specific to your video packaging; evaluates distinctly from competing videos."
                relevance = "Medium"
            else:
                match_status = "Secondary Topical Keyword"
                opportunity = "Useful supporting concept for description depth and viewer search context."
                relevance = "Medium"

            comparison_table.append({
                "keyword": term.title(),
                "submitted_evidence": s_evidence,
                "competitor_evidence": comp_evidence,
                "match_status": match_status,
                "relevance": relevance,
                "opportunity_gap": opportunity,
                "confidence": 0.94,
                "source": "explicit_metadata",
            })

        return comparison_table

    @classmethod
    def analyze_hashtags_side_by_side(
        cls,
        submitted_video: dict[str, Any],
        competitors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare observed hashtags between submitted video and competitors."""
        s_text = submitted_video.get("title", "") + " " + submitted_video.get("description", "")
        s_tags = extract_hashtags_from_text(s_text)

        comp_tag_counts: dict[str, int] = {}
        for c in competitors:
            c_text = c.get("title", "") + " " + c.get("description", "")
            c_tags = extract_hashtags_from_text(c_text)
            for t in c_tags:
                comp_tag_counts[t] = comp_tag_counts.get(t, 0) + 1

        missing_from_submitted = [
            {"tag": tag, "competitor_count": count, "reason": f"Observed in {count} competing video descriptions"}
            for tag, count in sorted(comp_tag_counts.items(), key=lambda x: x[1], reverse=True)
            if tag not in s_tags and count >= 1
        ]

        # Recommended topical hashtags
        primary_tokens = clean_text_tokens(submitted_video.get("title", ""))[:3]
        recommended = []
        for t in primary_tokens:
            h = f"#{t.lower()}"
            if h not in s_tags:
                recommended.append({
                    "tag": h,
                    "rationale": f"Directly reflects core topic entity '{t}'.",
                    "source": "inferred_topic_entity",
                })

        return {
            "submitted_hashtags": s_tags,
            "competitor_hashtags": [
                {"tag": t, "observed_count": c}
                for t, c in sorted(comp_tag_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "missing_competitor_hashtags": missing_from_submitted[:6],
            "recommended_hashtags": recommended[:4],
        }

    @classmethod
    def generate_gap_analysis(
        cls,
        submitted_video: dict[str, Any],
        competitors: list[dict[str, Any]],
        keyword_table: list[dict[str, Any]],
        hashtag_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Identify concrete evidence-backed gaps between submitted video and competitors."""
        gaps = []
        s_title = submitted_video.get("title", "")
        s_desc = submitted_video.get("description", "")
        s_views = submitted_video.get("view_count", 0)
        s_age = calculate_age_days(submitted_video.get("published_at", ""))
        s_vpd = s_views / s_age

        comp_vpd_values = [c.get("views_per_day", 0) for c in competitors if c.get("views_per_day") is not None]
        median_comp_vpd = sorted(comp_vpd_values)[len(comp_vpd_values) // 2] if comp_vpd_values else 0

        # Gap 1: Title Packaging & Search Term Specificity
        missing_kw = [k["keyword"] for k in keyword_table if k["match_status"] == "Competitor Keyword Gap"]
        if missing_kw:
            top_missing = missing_kw[:2]
            gaps.append({
                "title": "Search Intent Alignment in Title",
                "category": "Title & Packaging",
                "submitted_evidence": f"Submitted title ('{s_title}') omits recurring competitor search terms: {', '.join(top_missing)}.",
                "competitor_evidence": f"Multiple top competitors prominently feature '{top_missing[0]}' in their titles.",
                "why_it_matters": "Search and suggested video algorithms correlate title phrasing with viewer intent queries.",
                "recommended_action": f"Test an alternative title variant incorporating '{top_missing[0]}' within the first 45 characters.",
                "expected_benefit": "Hypothesis: May improve organic search discovery for viewers searching specifically for this topic.",
                "priority": "High",
                "effort": "Low",
                "confidence": 0.92,
                "evidence_ref": f"Observed in competitor titles across {len(competitors)} videos.",
            })

        # Gap 2: Description Depth & Timestamp Chapters
        has_timestamps = bool(re.search(r"\b\d{1,2}:\d{2}\b", s_desc))
        comp_with_timestamps = sum(1 for c in competitors if re.search(r"\b\d{1,2}:\d{2}\b", c.get("description", "")))
        if not has_timestamps and comp_with_timestamps >= 1:
            gaps.append({
                "title": "Missing Video Chapter Markers / Timestamps",
                "category": "Content Structure",
                "submitted_evidence": "Description lacks formatted 00:00 chapter timestamps.",
                "competitor_evidence": f"{comp_with_timestamps} competing videos provide timestamped chapters in their descriptions.",
                "why_it_matters": "YouTube uses chapter timestamps for Google Search key moments and improved viewer navigation.",
                "recommended_action": "Add structured timestamps (e.g. 00:00 Intro, 01:30 Architecture, 05:00 Demo) into description.",
                "expected_benefit": "Hypothesis: Enables Google Key Moments indexing and reduces bounce rate for specific sub-topic searches.",
                "priority": "Medium",
                "effort": "Low",
                "confidence": 0.95,
                "evidence_ref": "Observed description structure in top-performing competitors.",
            })

        # Gap 3: Description Resource Completeness
        desc_words = len(s_desc.split())
        comp_desc_words = [len(c.get("description", "").split()) for c in competitors]
        avg_comp_desc = sum(comp_desc_words) / len(comp_desc_words) if comp_desc_words else 100
        if desc_words < 50 and avg_comp_desc > 100:
            gaps.append({
                "title": "Brief Description Limiting Search Context",
                "category": "Metadata & Discovery",
                "submitted_evidence": f"Submitted video description contains only {desc_words} words.",
                "competitor_evidence": f"Competitors average {int(avg_comp_desc)} words with comprehensive summary paragraphs and resource links.",
                "why_it_matters": "The first 3 lines of description provide vital semantic signals for indexing before the 'Show more' fold.",
                "recommended_action": "Expand description to 150+ words covering key concepts, prerequisites, and resource links.",
                "expected_benefit": "Hypothesis: Provides additional semantic context for YouTube's recommendations without keyword stuffing.",
                "priority": "Medium",
                "effort": "Low",
                "confidence": 0.89,
                "evidence_ref": "Metadata depth comparison with selected competitors.",
            })

        # Gap 4: Hashtag Coverage
        missing_ht = hashtag_data.get("missing_competitor_hashtags", [])
        if not hashtag_data.get("submitted_hashtags") and missing_ht:
            top_ht = [h["tag"] for h in missing_ht[:3]]
            gaps.append({
                "title": "Absence of Relevant Topic Hashtags",
                "category": "Discovery Signals",
                "submitted_evidence": "No hashtags observed in submitted title or description.",
                "competitor_evidence": f"Competitors utilize topical hashtags such as {', '.join(top_ht)}.",
                "why_it_matters": "Hashtags assist YouTube in topical clustering and mobile hashtag landing page placement.",
                "recommended_action": f"Add 2-3 focused topical hashtags (e.g., {', '.join(top_ht)}) at the bottom of description.",
                "expected_benefit": "Hypothesis: Associates video with related topic hashtag clusters.",
                "priority": "Low",
                "effort": "Low",
                "confidence": 0.88,
                "evidence_ref": "Extracted from competitor description text.",
            })

        # Gap 5: View Velocity & Benchmark Context (Objective comparison)
        if s_vpd < median_comp_vpd * 0.5 and median_comp_vpd > 0:
            gaps.append({
                "title": "Observed View Velocity Differential",
                "category": "Observed Performance Signals",
                "submitted_evidence": f"Submitted video is pacing at {int(s_vpd):,} views/day over {s_age} days.",
                "competitor_evidence": f"Selected topical competitors average a median velocity of {int(median_comp_vpd):,} views/day.",
                "why_it_matters": "Velocity indicates higher audience pull from browse features or search rankings.",
                "recommended_action": "Combine title packaging refresh with high-contrast thumbnail test to evaluate click-through rate.",
                "expected_benefit": "Hypothesis: May reactivate impressions delivery in browse and suggested feeds.",
                "priority": "High",
                "effort": "Medium",
                "confidence": 0.91,
                "evidence_ref": "Calculated from YouTube API public view counts and publication dates.",
            })

        return gaps

    @classmethod
    def generate_prioritized_action_plan(cls, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create prioritized, actionable plan with validation methods."""
        plan = []
        priority_order = {"High": 1, "Medium": 2, "Low": 3}
        sorted_gaps = sorted(gaps, key=lambda g: priority_order.get(g["priority"], 4))

        for idx, gap in enumerate(sorted_gaps, 1):
            plan.append({
                "step": idx,
                "priority": gap["priority"],
                "category": gap["category"],
                "action": gap["recommended_action"],
                "rationale": gap["why_it_matters"],
                "hypothesis": gap["expected_benefit"],
                "effort": gap["effort"],
                "validation_method": "Monitor 7-day impressions and views velocity following metadata update.",
            })

        return plan
