import hashlib
import hmac
import secrets
import time
import urllib.parse
import httpx
from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_current_user, get_tenant_context
from app.core.config import settings
from app.core.logging import logger
from app.security import get_safe_redirect_url
from app.database.models import User
from app.database.session import get_db
from app.security.schemas import (
    GoogleConfigResponse,
    GoogleCredentialRequest,
    LogoutResponse,
    SessionResponse,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
    WorkspaceResponse,
)
from app.security.service import IdentityService

router = APIRouter(prefix="/auth", tags=["Authentication & Google Identity"])


def _generate_oauth_state(return_to: str | None = None) -> str:
    """Generate a cryptographically signed CSRF state token embedding timestamp and return destination."""
    raw = secrets.token_urlsafe(24)
    ts = str(int(time.time()))
    safe_dest = get_safe_redirect_url(return_to, default="/dashboard")
    dest_str = urllib.parse.quote(safe_dest, safe="")
    payload = f"{raw}.{ts}.{dest_str}"
    sig = hmac.new(settings.SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}.{sig}"


def _verify_oauth_state(state: str | None, oauth_state_cookie: str | None = None) -> tuple[bool, str]:
    """Validate CSRF state against cookie or cryptographic HMAC signature.
    
    Returns (is_valid, safe_return_destination).
    """
    if not state:
        return False, "/dashboard"
    
    dest = "/dashboard"
    try:
        parts = state.split(".")
        if len(parts) == 4:
            raw, ts_str, dest_str, sig = parts
            ts = int(ts_str)
            if abs(time.time() - ts) > 600:  # 10 minute validity
                return False, "/dashboard"
            payload = f"{raw}.{ts_str}.{dest_str}"
            expected_sig = hmac.new(settings.SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
            dest = get_safe_redirect_url(urllib.parse.unquote(dest_str), default="/dashboard")
            if secrets.compare_digest(sig, expected_sig):
                return True, dest
            if oauth_state_cookie and secrets.compare_digest(state, oauth_state_cookie):
                return True, dest
            return False, "/dashboard"
        elif len(parts) == 3:
            raw, ts_str, sig = parts
            ts = int(ts_str)
            if abs(time.time() - ts) > 600:
                return False, "/dashboard"
            payload = f"{raw}.{ts_str}"
            expected_sig = hmac.new(settings.SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
            if secrets.compare_digest(sig, expected_sig) or (oauth_state_cookie and secrets.compare_digest(state, oauth_state_cookie)):
                return True, "/dashboard"
            return False, "/dashboard"
    except Exception:
        return False, "/dashboard"
    return False, "/dashboard"


def set_auth_cookies(response: Response, session_token: str, csrf_token: str) -> None:
    """Set secure HttpOnly session cookie and client-accessible CSRF cookie."""
    max_age = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=session_token,
        max_age=max_age,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        path="/",
    )
    response.set_cookie(
        key=settings.CSRF_COOKIE_NAME,
        value=csrf_token,
        max_age=max_age,
        httponly=False,  # JavaScript reads this token to send in X-CSRF-Token headers
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        path="/",
    )


def clear_auth_cookies(response: Response) -> None:
    """Clear session and CSRF cookies on logout."""
    response.delete_cookie(key=settings.SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(key=settings.CSRF_COOKIE_NAME, path="/")


@router.get("/config", response_model=GoogleConfigResponse)
def get_auth_config():
    """Expose non-sensitive authentication configuration to frontend clients."""
    is_configured = bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)
    return GoogleConfigResponse(
        google_client_id=settings.GOOGLE_CLIENT_ID or "",
        is_configured=is_configured,
    )


# --------------------------------------------------------------------------
# Google OAuth 2.0 / OpenID Connect Endpoints
# --------------------------------------------------------------------------

@router.get("/google/login")
def google_oauth_login(return_to: str | None = None):
    """Initiate standard Google OAuth 2.0 authorization code flow."""
    is_placeholder = (
        not settings.GOOGLE_CLIENT_ID
        or not settings.GOOGLE_CLIENT_SECRET
        or "your-google-client-id" in (settings.GOOGLE_CLIENT_ID or "")
        or "your-google-client-secret" in (settings.GOOGLE_CLIENT_SECRET or "")
    )
    if is_placeholder:
        logger.warning("Google OAuth credentials not configured in environment (or placeholder values).")
        safe_ret = get_safe_redirect_url(return_to, default="/dashboard")
        return RedirectResponse(
            url=f"/login?error=google_not_configured&return_to={urllib.parse.quote(safe_ret)}",
            status_code=status.HTTP_302_FOUND,
        )

    # Cryptographically secure HMAC-signed state parameter for CSRF mitigation and destination preservation
    state = _generate_oauth_state(return_to=return_to)
    nonce = secrets.token_urlsafe(16)

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "access_type": "online",
        "prompt": "select_account",
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"

    response = RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)
    # Store signed state in temporary short-lived HttpOnly cookie for CSRF verification
    response.set_cookie(
        key="oauth_state",
        value=state,
        max_age=600,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    return response


@router.get("/google/callback")
async def google_oauth_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    oauth_state: str | None = Cookie(default=None),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """Handle Google OAuth 2.0 redirection, validate token, and issue authenticated session."""
    if error:
        logger.warning(f"Google OAuth reported error: {error}")
        return RedirectResponse(url=f"/login?error={urllib.parse.quote(error)}")

    if not code or not state:
        return RedirectResponse(url="/login?error=missing_oauth_parameters")

    # Verify CSRF state parameter matches cookie or cryptographic HMAC signature
    is_valid, return_dest = _verify_oauth_state(state, oauth_state)
    if not is_valid:
        logger.error("OAuth state mismatch; potential CSRF attempt.")
        return RedirectResponse(url="/login?error=invalid_csrf_state")

    # Exchange authorization code with Google token endpoint
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "code": code,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_resp = await client.post(token_url, data=token_data)
            if token_resp.status_code != 200:
                logger.error(f"Google token exchange failed: {token_resp.text}")
                return RedirectResponse(url="/login?error=token_exchange_failed")

            tokens = token_resp.json()
            access_token = tokens.get("access_token")

            # Fetch verified OpenID profile data from Google UserInfo endpoint
            userinfo_resp = await client.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if userinfo_resp.status_code != 200:
                logger.error(f"Google userinfo query failed: {userinfo_resp.text}")
                return RedirectResponse(url="/login?error=userinfo_fetch_failed")

            user_info = userinfo_resp.json()

    except Exception as e:
        logger.exception(f"Unexpected error during Google OAuth exchange: {e}")
        return RedirectResponse(url="/login?error=oauth_internal_error")

    # Register or locate user and create server-side session
    ip = request.client.host if request and request.client else None
    ua = request.headers.get("user-agent") if request else None

    try:
        service = IdentityService(db)
        user, session, workspace, role = service.create_or_get_google_user(
            user_info,
            ip_address=ip,
            user_agent=ua,
        )
    except Exception as e:
        logger.exception(f"Failed to provision Google user: {e}")
        return RedirectResponse(url=f"/login?error={urllib.parse.quote(str(e))}")

    # Set secure session cookies and redirect into application destination
    redirect = RedirectResponse(url=return_dest, status_code=status.HTTP_302_FOUND)
    set_auth_cookies(redirect, session.session_token, session.csrf_token)
    redirect.delete_cookie("oauth_state", path="/")
    return redirect


@router.post("/google/credential", response_model=SessionResponse)
async def google_credential_callback(
    payload: GoogleCredentialRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Authenticate client-side Google Identity Services credential (One Tap / Sign-In Button)."""
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google Client ID is not configured on the server.",
        )

    # Server-side validation of ID token with Google's tokeninfo endpoint
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            tokeninfo_resp = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": payload.credential},
            )
            if tokeninfo_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Google ID Token credential.",
                )
            claims = tokeninfo_resp.json()

            # Verify audience matches client ID
            if claims.get("aud") != settings.GOOGLE_CLIENT_ID:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Google ID token audience mismatch.",
                )

            if not claims.get("email_verified") in (True, "true"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Google account email is not verified.",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error validating Google ID token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate Google credential.",
        )

    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")

    service = IdentityService(db)
    user, session, workspace, role = service.create_or_get_google_user(
        claims,
        ip_address=ip,
        user_agent=ua,
    )

    set_auth_cookies(response, session.session_token, session.csrf_token)

    ws_resp = WorkspaceResponse(
        id=workspace.id,
        organization_id=workspace.organization_id,
        name=workspace.name,
        slug=workspace.slug,
        role=role,
    )
    user_resp = UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
        auth_provider=user.auth_provider,
        is_active=user.is_active,
        workspaces=[ws_resp],
    )
    return SessionResponse(
        user=user_resp,
        active_workspace=ws_resp,
        csrf_token=session.csrf_token,
    )


# --------------------------------------------------------------------------
# Password Registration, Login, Logout & Session Management
# --------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: UserCreate,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        service = IdentityService(db)
        ip = request.client.host if request.client else None
        ua = request.headers.get("user-agent")
        token_resp, session = service.register_user(payload, ip_address=ip, user_agent=ua)
        set_auth_cookies(response, session.session_token, session.csrf_token)
        return token_resp
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenResponse)
def login(
    payload: UserLogin,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        service = IdentityService(db)
        ip = request.client.host if request.client else None
        ua = request.headers.get("user-agent")
        token_resp, session = service.authenticate_user(payload, ip_address=ip, user_agent=ua)
        set_auth_cookies(response, session.session_token, session.csrf_token)
        return token_resp
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/logout", response_model=LogoutResponse)
def logout(
    response: Response,
    request: Request,
    growtube_session: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """Revoke user session server-side and clear client cookies."""
    session_token = growtube_session or request.cookies.get(settings.SESSION_COOKIE_NAME)
    if session_token:
        service = IdentityService(db)
        service.revoke_session(session_token)

    clear_auth_cookies(response)
    return LogoutResponse(message="Successfully signed out of GrowTube.", status="success")


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(
    user: User = Depends(get_current_user),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """Retrieve verified profile and active workspaces for the current authenticated user."""
    workspaces = [
        WorkspaceResponse(
            id=m.workspace.id,
            organization_id=m.workspace.organization_id,
            name=m.workspace.name,
            slug=m.workspace.slug,
            role=m.role,
        )
        for m in user.memberships
        if m.workspace
    ]

    csrf = tenant.session.csrf_token if tenant.session else None

    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
        auth_provider=user.auth_provider,
        is_active=user.is_active,
        workspaces=workspaces,
        csrf_token=csrf,
    )
