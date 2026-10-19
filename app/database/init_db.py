from sqlalchemy.orm import Session
from app.core.logging import logger
from app.security import hash_password
from app.database.base import Base
from app.database.models import (
    Membership,
    Organization,
    User,
    Workspace,
    ChannelConnection,
    ContentOpportunity,
    ContentScript,
    ExperimentRecord,
    VideoObservation,
    NicheSearchCache,
    CreatorStruggleCache,
)
from app.database.session import SessionLocal, engine


from sqlalchemy import inspect, text


def ensure_schema_compatibility() -> None:
    """Ensure existing tables have all required columns without dropping or modifying data."""
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if "users" in tables:
            columns = [col["name"] for col in inspector.get_columns("users")]
            with engine.begin() as conn:
                if "google_id" not in columns:
                    logger.info("Ensuring schema compatibility: adding missing column users.google_id")
                    conn.execute(text("ALTER TABLE users ADD COLUMN google_id VARCHAR(100)"))
                    try:
                        conn.execute(text("CREATE UNIQUE INDEX ix_users_google_id ON users (google_id)"))
                    except Exception:
                        pass
                if "avatar_url" not in columns:
                    logger.info("Ensuring schema compatibility: adding missing column users.avatar_url")
                    conn.execute(text("ALTER TABLE users ADD COLUMN avatar_url VARCHAR(500)"))
                if "auth_provider" not in columns:
                    logger.info("Ensuring schema compatibility: adding missing column users.auth_provider")
                    conn.execute(text("ALTER TABLE users ADD COLUMN auth_provider VARCHAR(50) DEFAULT 'local' NOT NULL"))

        if "analysis_jobs" in tables:
            job_columns = [col["name"] for col in inspector.get_columns("analysis_jobs")]
            with engine.begin() as conn:
                if "checkpoint_data" not in job_columns:
                    logger.info("Ensuring schema compatibility: adding missing column analysis_jobs.checkpoint_data")
                    conn.execute(text("ALTER TABLE analysis_jobs ADD COLUMN checkpoint_data JSON"))
                if "retry_count" not in job_columns:
                    logger.info("Ensuring schema compatibility: adding missing column analysis_jobs.retry_count")
                    conn.execute(text("ALTER TABLE analysis_jobs ADD COLUMN retry_count INTEGER DEFAULT 0 NOT NULL"))

        if "conversation_messages" in tables:
            msg_columns = [col["name"] for col in inspector.get_columns("conversation_messages")]
            with engine.begin() as conn:
                if "extra_metadata" not in msg_columns:
                    logger.info("Ensuring schema compatibility: adding missing column conversation_messages.extra_metadata")
                    conn.execute(text("ALTER TABLE conversation_messages ADD COLUMN extra_metadata JSON"))

        if "conversations" in tables:
            conv_columns = [col["name"] for col in inspector.get_columns("conversations")]
            with engine.begin() as conn:
                if "channel_context" not in conv_columns:
                    logger.info("Ensuring schema compatibility: adding missing column conversations.channel_context")
                    conn.execute(text("ALTER TABLE conversations ADD COLUMN channel_context JSON"))
    except Exception as exc:
        logger.warning(f"Schema compatibility check warning: {exc}")


def init_database() -> None:
    """Create all tables in database and verify schema integrity."""
    try:
        Base.metadata.create_all(bind=engine)
        ensure_schema_compatibility()
        logger.info("Database tables verified and initialized successfully.")
    except Exception as exc:
        logger.error(f"SQLite database initialization error: {exc}")


def seed_default_data() -> None:
    """Seed initial demo organization, workspace, user, and sample channel data if empty."""
    try:
        db: Session = SessionLocal()
    except Exception as exc:
        logger.warning(f"Could not connect to database for seeding: {exc}")
        return

    try:
        # Check if users already exist
        if db.query(User).first() is not None:
            return

        # Create Default User
        demo_user = User(
            id="00000000-0000-0000-0000-000000000001",
            email="alex@growtube.dev",
            hashed_password=hash_password("growtube123"),
            full_name="Alex River",
            auth_provider="local",
            is_active=True,
        )
        db.add(demo_user)

        # Create Default Org
        demo_org = Organization(
            id="00000000-0000-0000-0000-000000000010",
            name="Alex Studio",
            slug="alex-studio",
        )
        db.add(demo_org)

        # Create Default Workspace
        demo_ws = Workspace(
            id="00000000-0000-0000-0000-000000000100",
            organization_id=demo_org.id,
            name="Creator Studio",
            slug="creator-studio",
        )
        db.add(demo_ws)

        # Create Membership
        demo_mem = Membership(
            id="00000000-0000-0000-0000-000000001000",
            user_id=demo_user.id,
            workspace_id=demo_ws.id,
            role="owner",
        )
        db.add(demo_mem)

        # Clean database initialization:
        # No prefilled ChannelConnection, ContentOpportunity, ContentScript, or ExperimentRecord.
        # Data will strictly populate when user uploads or analyzes a target.
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
