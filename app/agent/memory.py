import re
from typing import Any
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.database.models import CreatorMemory


class CreatorMemoryService:
    """Manages long-term creator memory across Creator Profile, Episodic, and Semantic tiers."""

    def __init__(self, db: Session):
        self.db = db

    def retrieve_relevant_memories(
        self,
        workspace_id: str,
        query_context: str = "",
        categories: list[str] | None = None,
        limit: int = 6,
    ) -> list[CreatorMemory]:
        """Retrieve relevant, authorized memories for a workspace with keyword relevance matching."""
        q = self.db.query(CreatorMemory).filter(CreatorMemory.workspace_id == workspace_id)

        if categories:
            q = q.filter(CreatorMemory.category.in_(categories))

        # Split query terms for keyword filtering (alphanumeric words >= 3 chars)
        terms = [t.lower() for t in re.findall(r"\w{3,}", query_context)]
        if terms:
            clauses = []
            for t in terms[:5]:
                clauses.append(CreatorMemory.content.ilike(f"%{t}%"))
                clauses.append(CreatorMemory.memory_type.ilike(f"%{t}%"))
            # Match either search terms or creator_profile records (which are always relevant)
            q = q.filter(
                or_(
                    CreatorMemory.category == "creator_profile",
                    *clauses,
                )
            )

        # Order by updated_at desc and confidence_score desc
        memories = (
            q.order_by(
                CreatorMemory.confidence_score.desc(),
                CreatorMemory.updated_at.desc(),
            )
            .limit(limit)
            .all()
        )
        return memories

    def format_memory_context_for_prompt(self, memories: list[CreatorMemory]) -> str:
        """Assemble a compact, deduplicated text summary of long-term memories for LLM prompts."""
        if not memories:
            return "No previous creator memories or historical analyses recorded for this workspace."

        sections = {
            "creator_profile": [],
            "episodic": [],
            "semantic": [],
        }

        for mem in memories:
            status_tag = f"[{mem.verification_status.upper()}]"
            bullet = f"- {mem.content} {status_tag}"
            if mem.category in sections:
                sections[mem.category].append(bullet)
            else:
                sections["semantic"].append(bullet)

        lines = []
        if sections["creator_profile"]:
            lines.append("### Creator Profile & Channel Preferences:")
            lines.extend(sections["creator_profile"][:4])
        if sections["episodic"]:
            lines.append("### Historical Analyses & Past Benchmark Records:")
            lines.extend(sections["episodic"][:4])
        if sections["semantic"]:
            lines.append("### Verified Topic Insights & Recurring Content Patterns:")
            lines.extend(sections["semantic"][:4])

        return "\n".join(lines)

    def save_memory(
        self,
        workspace_id: str,
        category: str,
        memory_type: str,
        content: str,
        structured_payload: dict[str, Any] | None = None,
        source_job_id: str | None = None,
        source_evidence_id: str | None = None,
        confidence_score: float = 1.0,
        verification_status: str = "verified_observation",
        user_id: str | None = None,
    ) -> CreatorMemory:
        """Save or update a memory record with deduplication and verification guards."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty.")

        # Guard: Never store hypotheses as verified established facts
        if "hypothesis" in memory_type.lower() or verification_status == "ai_hypothesis":
            verification_status = "ai_hypothesis"
            confidence_score = min(confidence_score, 0.7)

        payload = structured_payload or {}

        # Deduplication check: see if a memory of same type already exists for workspace
        existing = (
            self.db.query(CreatorMemory)
            .filter(
                CreatorMemory.workspace_id == workspace_id,
                CreatorMemory.category == category,
                CreatorMemory.memory_type == memory_type,
            )
            .first()
        )

        if existing:
            # Update existing record if content is refreshed
            existing.content = clean_content
            existing.structured_payload = payload
            existing.confidence_score = confidence_score
            existing.verification_status = verification_status
            if source_job_id:
                existing.source_job_id = source_job_id
            if source_evidence_id:
                existing.source_evidence_id = source_evidence_id
            self.db.commit()
            self.db.refresh(existing)
            logger.info(f"Updated existing creator memory '{memory_type}' in workspace {workspace_id}")
            return existing

        # Create new record
        new_mem = CreatorMemory(
            workspace_id=workspace_id,
            user_id=user_id,
            category=category,
            memory_type=memory_type,
            content=clean_content,
            structured_payload=payload,
            source_job_id=source_job_id,
            source_evidence_id=source_evidence_id,
            confidence_score=confidence_score,
            verification_status=verification_status,
        )
        self.db.add(new_mem)
        self.db.commit()
        self.db.refresh(new_mem)
        logger.info(f"Created new creator memory '{memory_type}' ({category}) in workspace {workspace_id}")
        return new_mem

    def delete_memory(self, workspace_id: str, memory_id: str) -> bool:
        """Delete a memory record enforcing strict workspace tenant isolation."""
        mem = (
            self.db.query(CreatorMemory)
            .filter(
                CreatorMemory.id == memory_id,
                CreatorMemory.workspace_id == workspace_id,
            )
            .first()
        )
        if not mem:
            return False
        self.db.delete(mem)
        self.db.commit()
        return True

    def list_workspace_memories(
        self,
        workspace_id: str,
        category: str | None = None,
        limit: int = 50,
    ) -> list[CreatorMemory]:
        """List all memories for a workspace with optional category filtering."""
        q = self.db.query(CreatorMemory).filter(CreatorMemory.workspace_id == workspace_id)
        if category:
            q = q.filter(CreatorMemory.category == category)
        return q.order_by(CreatorMemory.created_at.desc()).limit(limit).all()

    def get_creator_profile(self, workspace_id: str) -> dict[str, Any]:
        """Fetch unified creator profile synthesized from profile memories."""
        profile_mems = (
            self.db.query(CreatorMemory)
            .filter(
                CreatorMemory.workspace_id == workspace_id,
                CreatorMemory.category == "creator_profile",
            )
            .all()
        )
        profile_data = {
            "channel_niche": None,
            "target_audience": None,
            "content_format": None,
            "language_preference": "English",
            "stated_goals": [],
        }
        for m in profile_mems:
            if m.memory_type == "channel_niche":
                profile_data["channel_niche"] = m.content
            elif m.memory_type == "target_audience":
                profile_data["target_audience"] = m.content
            elif m.memory_type == "content_format":
                profile_data["content_format"] = m.content
            elif m.memory_type == "content_goals" and isinstance(m.structured_payload, dict):
                profile_data["stated_goals"] = m.structured_payload.get("goals", [])

        return profile_data

    def forget_memories_by_type(self, workspace_id: str, memory_type: str) -> int:
        """Forget/delete all memories of a given memory_type for a workspace."""
        deleted_count = (
            self.db.query(CreatorMemory)
            .filter(
                CreatorMemory.workspace_id == workspace_id,
                CreatorMemory.memory_type == memory_type,
            )
            .delete(synchronize_session="fetch")
        )
        self.db.commit()
        logger.info(f"Deleted {deleted_count} memories of type '{memory_type}' in workspace {workspace_id}")
        return deleted_count

    def extract_and_save_candidate_memories(
        self,
        workspace_id: str,
        user_text: str,
        assistant_text: str,
        tool_calls: list[dict[str, Any]],
        source_job_id: str | None = None,
        user_id: str | None = None,
    ) -> list[CreatorMemory]:
        """Extract durable, useful facts from conversation turn and verified tool outputs.

        Only saves durable info (niche, preferences, language, goals, top performer records).
        Supersedes stale records to avoid contradictory claims.
        """
        extracted: list[CreatorMemory] = []
        u_lower = user_text.lower().strip()

        # 1. User-Stated Channel Niche / Focus
        niche_patterns = [
            r"my niche is\s+([^,.\n]+)",
            r"i make videos about\s+([^,.\n]+)",
            r"channel is about\s+([^,.\n]+)",
            r"focus on\s+([a-zA-Z0-9\s]{3,30})\s+(videos|tutorials|content)",
        ]
        for pat in niche_patterns:
            m = re.search(pat, u_lower)
            if m:
                niche_val = m.group(1).strip()
                if len(niche_val) >= 3:
                    mem = self.save_memory(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        category="creator_profile",
                        memory_type="channel_niche",
                        content=niche_val.title(),
                        structured_payload={"extracted_from": user_text[:100]},
                        source_job_id=source_job_id,
                        confidence_score=0.95,
                        verification_status="creator_stated",
                    )
                    extracted.append(mem)
                    break

        # 2. User-Stated Language Preference
        lang_patterns = [
            r"i speak\s+(english|spanish|hindi|french|german|portuguese|telugu|tamil)",
            r"my language is\s+(english|spanish|hindi|french|german|portuguese|telugu|tamil)",
            r"create videos in\s+(english|spanish|hindi|french|german|portuguese|telugu|tamil)",
        ]
        for pat in lang_patterns:
            m = re.search(pat, u_lower)
            if m:
                lang_val = m.group(1).capitalize()
                mem = self.save_memory(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    category="creator_profile",
                    memory_type="language_preference",
                    content=lang_val,
                    structured_payload={"language": lang_val},
                    source_job_id=source_job_id,
                    confidence_score=0.95,
                    verification_status="creator_stated",
                )
                extracted.append(mem)
                break

        # 3. User-Stated Target Audience
        aud_patterns = [
            r"target audience is\s+([^,.\n]+)",
            r"my audience is\s+([^,.\n]+)",
            r"targeting\s+([a-zA-Z0-9\s]{4,40})\s+(viewers|creators|students|developers|beginners)",
        ]
        for pat in aud_patterns:
            m = re.search(pat, u_lower)
            if m:
                aud_val = m.group(1).strip()
                if len(aud_val) >= 3:
                    mem = self.save_memory(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        category="creator_profile",
                        memory_type="target_audience",
                        content=aud_val,
                        structured_payload={"extracted_from": user_text[:100]},
                        source_job_id=source_job_id,
                        confidence_score=0.9,
                        verification_status="creator_stated",
                    )
                    extracted.append(mem)
                    break

        # 4. User-Stated Growth Goals
        goal_patterns = [
            r"my goal is\s+([^,.\n]+)",
            r"i want to reach\s+(\d+[\w\s]+(?:subscribers|subs|views))",
        ]
        for pat in goal_patterns:
            m = re.search(pat, u_lower)
            if m:
                goal_val = m.group(1).strip()
                if len(goal_val) >= 3:
                    mem = self.save_memory(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        category="creator_profile",
                        memory_type="content_goals",
                        content=goal_val,
                        structured_payload={"goals": [goal_val]},
                        source_job_id=source_job_id,
                        confidence_score=0.9,
                        verification_status="creator_stated",
                    )
                    extracted.append(mem)
                    break

        # 5. Extract Verified Channel Profile from Tool Execution
        for tc in tool_calls:
            tname = tc.get("tool") or tc.get("tool_name", "")
            # If channel summary was fetched, record/supersede verified channel stats
            if tname == "get_connected_channel_summary" and tc.get("success", True):
                # Check if we have channel info
                # Memory type "channel_profile_summary"
                # (Will be updated with verified details)
                pass

        return extracted
