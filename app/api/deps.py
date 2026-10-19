from typing import Annotated, Callable
from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.security import decode_access_token
from app.database.models import User, UserSession
from app.database.session import get_db
from app.security.service import IdentityService


class TenantContext:
    def __init__(self, user: User, workspace_id: str, role: str, session: UserSession | None = None):
        self.user = user
        self.workspace_id = workspace_id
        self.role = role
        self.session = session


def get_current_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    growtube_session: Annotated[str | None, Cookie(alias="growtube_session")] = None,
    x_csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
    db: Session = Depends(get_db),
) -> User:
    """Validate HttpOnly session cookie or Bearer token with server-side revocation and CSRF protection."""
    identity_svc = IdentityService(db)

    # 1. Check HttpOnly session cookie first
    session_token = growtube_session or request.cookies.get(settings.SESSION_COOKIE_NAME)
    if session_token:
        session = identity_svc.get_session(session_token)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session has expired or was revoked. Please log in again.",
            )

        # CSRF Protection on mutating HTTP requests when using cookie-based auth
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            # Exclude auth endpoints that establish initial session or handle OAuth callbacks
            path = request.url.path
            exempt_paths = (
                "/api/v1/auth/login",
                "/api/v1/auth/register",
                "/api/v1/auth/google/callback",
                "/api/v1/auth/google/credential",
            )
            if not any(path.startswith(ep) for ep in exempt_paths):
                if not x_csrf_token or x_csrf_token != session.csrf_token:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="CSRF validation failed: missing or invalid X-CSRF-Token header.",
                    )

        request.state.session = session
        user = session.user
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive or not found.",
            )
        return user

    # 2. Check Authorization: Bearer token (for API clients)
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        # Check if Bearer token is a session token
        session = identity_svc.get_session(token)
        if session and session.user and session.user.is_active:
            request.state.session = session
            return session.user

        payload = decode_access_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired Bearer token.",
            )

        user_id = payload.get("sub")
        user = identity_svc.get_user_by_id(user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive or not found.",
            )
        return user

    # 3. Reject unauthenticated access strictly
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please sign in with Google or your credentials.",
    )


def get_tenant_context(
    request: Request,
    user: User = Depends(get_current_user),
    x_workspace_id: Annotated[str | None, Header(alias="X-Workspace-Id")] = None,
    db: Session = Depends(get_db),
) -> TenantContext:
    """Ensure user has validated membership in the targeted workspace, protecting against cross-tenant access."""
    identity_svc = IdentityService(db)
    active_session = getattr(request.state, "session", None)

    # If specific workspace header is provided, enforce explicit role membership
    if x_workspace_id:
        role = identity_svc.verify_workspace_access(user.id, x_workspace_id)
        if not role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Cross-tenant access prohibited: no authorized membership in workspace '{x_workspace_id}'.",
            )
        return TenantContext(user=user, workspace_id=x_workspace_id, role=role, session=active_session)

    # Default to user's first active workspace membership
    if not user.memberships:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is not assigned to any workspace.",
        )

    primary_membership = user.memberships[0]
    return TenantContext(
        user=user,
        workspace_id=primary_membership.workspace_id,
        role=primary_membership.role,
        session=active_session,
    )


def get_optional_tenant_context(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    growtube_session: Annotated[str | None, Cookie(alias="growtube_session")] = None,
    x_workspace_id: Annotated[str | None, Header(alias="X-Workspace-Id")] = None,
    db: Session = Depends(get_db),
) -> TenantContext | None:
    """Optional tenant context that yields None rather than 401/403 if unauthenticated."""
    try:
        user = get_current_user(
            request=request,
            authorization=authorization,
            growtube_session=growtube_session,
            db=db,
        )
        return get_tenant_context(request=request, user=user, x_workspace_id=x_workspace_id, db=db)
    except Exception:
        return None


def require_role(allowed_roles: list[str]) -> Callable:
    """Role-based authorization dependency factory (e.g. ['owner', 'admin'])."""
    def role_checker(tenant: TenantContext = Depends(get_tenant_context)) -> TenantContext:
        if tenant.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of roles {allowed_roles}, but current role is '{tenant.role}'.",
            )
        return tenant
    return role_checker
