import threading
import time
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.database.models import AnalysisJob
from app.database.session import SessionLocal
from app.agent.orchestrator import AgentOrchestrator

_worker_thread: threading.Thread | None = None
_stop_signal = threading.Event()


def process_next_job() -> bool:
    """Find interrupted or oldest queued job and execute it with safe resumption. Returns True if a job was processed."""
    db: Session = SessionLocal()
    try:
        # 1. First recover any interrupted running or retry_pending jobs with checkpoints
        recoverable_job = (
            db.query(AnalysisJob)
            .filter(
                AnalysisJob.status.in_(["retry_pending", "running"]),
                AnalysisJob.retry_count < 2,
            )
            .order_by(AnalysisJob.updated_at.asc())
            .first()
        )
        if recoverable_job and recoverable_job.checkpoint_data:
            logger.info(f"Worker recovering interrupted job {recoverable_job.id} from checkpoint.")
            recoverable_job.retry_count += 1
            db.commit()
            orchestrator = AgentOrchestrator(db)
            orchestrator.execute_workflow(recoverable_job.id)
            return True

        # 2. Pick up oldest queued job
        job = (
            db.query(AnalysisJob)
            .filter(AnalysisJob.status == "queued")
            .order_by(AnalysisJob.created_at.asc())
            .first()
        )
        if not job:
            return False

        logger.info(f"Worker picked up job {job.id} for execution.")
        orchestrator = AgentOrchestrator(db)
        orchestrator.execute_workflow(job.id)
        return True
    except Exception as e:
        logger.error(f"Worker encountered error: {e}")
        return False
    finally:
        db.close()


def run_worker_loop():
    """Continuous polling loop for jobs."""
    logger.info("Background job worker loop started.")
    while not _stop_signal.is_set():
        processed = process_next_job()
        if not processed:
            time.sleep(settings.WORKER_POLL_INTERVAL_SECONDS)


def start_in_process_worker():
    """Start background worker thread within the FastAPI process."""
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _stop_signal.clear()
        _worker_thread = threading.Thread(target=run_worker_loop, daemon=True, name="JobWorkerThread")
        _worker_thread.start()
        logger.info("Embedded in-process background worker thread launched.")


def stop_in_process_worker():
    """Signal background worker thread to stop."""
    global _worker_thread
    _stop_signal.set()
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=2.0)
    _worker_thread = None


if __name__ == "__main__":
    from app.core.logging import setup_logging
    setup_logging()
    logger.info("Starting standalone GrowTube worker process...")
    try:
        run_worker_loop()
    except KeyboardInterrupt:
        logger.info("Worker process stopped by user.")
