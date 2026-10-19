import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.security import get_safe_redirect_url
from app.database.init_db import init_database, seed_default_data
from app.database.session import get_db
from app.security.service import IdentityService
from app.workers.worker import start_in_process_worker, stop_in_process_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info("Starting up GrowTube platform...")
    init_database()
    seed_default_data()
    start_in_process_worker()

    # Ollama Local Agent Check
    from app.agent.providers.llm import OllamaLLMClient
    ollama_client = OllamaLLMClient()
    ok, status_msg = ollama_client.check_model_status()
    if ok:
        logger.info(f"Ollama local model initialized: {status_msg}")
    else:
        logger.warning(f"Ollama local model warning: {status_msg}")

    yield
    # Shutdown
    logger.info("Shutting down GrowTube platform...")
    stop_in_process_worker()


app = FastAPI(
    title="GrowTube — YouTube Growth Agent",
    description="Production-grade, modular, open-source YouTube Growth Agent platform.",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS Middleware (Supporting secure HttpOnly credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1
app.include_router(api_router)


# -------------------------------------------------------------
# Frontend Static Files & Clean SPA Routes
# -------------------------------------------------------------
frontend_path = Path(__file__).resolve().parent.parent / "frontend"

if frontend_path.exists():
    # Mount css, js, assets directories
    app.mount("/css", StaticFiles(directory=frontend_path / "css"), name="css")
    app.mount("/js", StaticFiles(directory=frontend_path / "js"), name="js")
    app.mount("/assets", StaticFiles(directory=frontend_path / "assets"), name="assets")

    def _is_authenticated_request(request: Request, db: Session) -> bool:
        session_token = request.cookies.get(settings.SESSION_COOKIE_NAME)
        if not session_token:
            return False
        identity_svc = IdentityService(db)
        session = identity_svc.get_session(session_token)
        return bool(session and session.user and session.user.is_active)

    def _serve_protected_page(page_filename: str, request: Request, db: Session):
        if not _is_authenticated_request(request, db):
            current_path = request.url.path
            if request.url.query:
                current_path = f"{current_path}?{request.url.query}"
            safe_dest = get_safe_redirect_url(current_path, default="/dashboard")
            return RedirectResponse(
                url=f"/login?return_to={urllib.parse.quote(safe_dest)}",
                status_code=status.HTTP_302_FOUND,
            )
        return FileResponse(frontend_path / "pages" / page_filename)

    # Clean HTML Page Routes matching navigation
    @app.get("/", summary="Landing Page")
    async def serve_index():
        return FileResponse(frontend_path / "index.html")

    @app.get("/login", summary="Login Page")
    async def serve_login(request: Request, return_to: str | None = None, db: Session = Depends(get_db)):
        if _is_authenticated_request(request, db):
            dest = get_safe_redirect_url(return_to, default="/dashboard")
            return RedirectResponse(url=dest, status_code=status.HTTP_302_FOUND)
        return FileResponse(frontend_path / "pages" / "login.html")

    @app.get("/dashboard", summary="Dashboard Page")
    async def serve_dashboard(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("dashboard.html", request, db)

    @app.get("/video-analysis", summary="Video Analysis Dashboard Page")
    async def serve_video_analysis(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("video-analysis.html", request, db)

    @app.get("/analyze", summary="Analyze Page")
    async def serve_analyze(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("analyze.html", request, db)

    @app.get("/report", summary="Report Page")
    async def serve_report(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("report.html", request, db)

    @app.get("/opportunities", summary="Opportunities Page")
    async def serve_opportunities(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("opportunities.html", request, db)

    @app.get("/niche-radar", summary="Niche Radar Page")
    async def serve_niche_radar(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("niche-radar.html", request, db)

    @app.get("/creator-struggles", summary="Creator Struggles Page")
    async def serve_creator_struggles(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("creator-struggles.html", request, db)

    @app.get("/ask-ai", summary="Ask AI Page")
    async def serve_ask_ai(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("ask-ai.html", request, db)

    @app.get("/content-studio", summary="Content Studio Page")
    async def serve_content_studio(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("content-studio.html", request, db)

    @app.get("/experiments", summary="Experiments Page")
    async def serve_experiments(request: Request, db: Session = Depends(get_db)):
        return _serve_protected_page("experiments.html", request, db)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}
