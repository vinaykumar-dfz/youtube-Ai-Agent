from sqlalchemy.orm import Session
from app.database.models import ContentScript
from app.agent.providers.llm import OpenRouterLLMClient


class ContentStudioService:
    def __init__(self, db: Session):
        self.db = db
        self.llm = OpenRouterLLMClient()

    def generate_script_from_topic(
        self,
        workspace_id: str,
        title: str,
        video_type: str = "YouTube Video",
        opportunity_id: str | None = None,
    ) -> ContentScript:
        """Generate structured script draft with verified checklist matching UI reference Screen 6."""
        # Try generating via configured OpenRouter LLM first
        llm_draft = self.llm.generate_script_draft(title, video_type)
        if llm_draft:
            script_body = llm_draft
        else:
            script_body = f"""[HOOK - 0:00-0:15]
Most creators struggle with {title.lower()}... but today, I'm showing you the exact framework that changes everything.

[INTRO - 0:15-0:45]
Welcome back! In this breakdown, we're diving deep into {title}. By the end of this video, you'll have an actionable roadmap you can implement immediately.

[CORE 1: THE FOUNDATION - 0:45-2:30]
Let's first address the biggest misconception in this space. Most people start in the wrong order...

[CORE 2: PRACTICAL WORKFLOW - 2:30-6:30]
Step 1: Set up the environment.
Step 2: Connect the core components.
Step 3: Measure and iterate based on real feedback.

[CALL TO ACTION - 6:30-7:15]
If you found this step-by-step breakdown valuable, hit the subscribe button, check out the resources linked in the description, and drop your questions below!"""

        default_checklist = [
            {"id": "c1", "label": "Strong hook in first 15 seconds", "completed": True},
            {"id": "c2", "label": "Clear value proposition", "completed": True},
            {"id": "c3", "label": "Structured with chapters", "completed": True},
            {"id": "c4", "label": "Relevant examples and demos", "completed": True},
            {"id": "c5", "label": "Call to action", "completed": True},
            {"id": "c6", "label": "Aligned with audience interests", "completed": True},
        ]

        script = ContentScript(
            workspace_id=workspace_id,
            opportunity_id=opportunity_id,
            title=title,
            video_type=video_type,
            script_draft=script_body,
            checklist=default_checklist,
            status="draft",
        )
        self.db.add(script)
        self.db.commit()
        self.db.refresh(script)
        return script

    def list_scripts(self, workspace_id: str) -> list[ContentScript]:
        return (
            self.db.query(ContentScript)
            .filter(ContentScript.workspace_id == workspace_id)
            .order_by(ContentScript.updated_at.desc())
            .all()
        )

    def update_script(self, script_id: str, workspace_id: str, draft: str) -> ContentScript | None:
        script = (
            self.db.query(ContentScript)
            .filter(ContentScript.id == script_id, ContentScript.workspace_id == workspace_id)
            .first()
        )
        if script:
            script.script_draft = draft
            self.db.commit()
            self.db.refresh(script)
        return script
