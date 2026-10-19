import re
import secrets
from datetime import UTC, datetime, timedelta
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import Membership, Organization, User, UserSession, Workspace
from app.security.crypto import create_access_token, decode_access_token, hash_password, verify_password
from app.security.schemas import (
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
    WorkspaceResponse,
)


class IdentityService:
    def __init__(self, db: Session):
        self.db = db

    def register_user(
        self,
        payload: UserCreate,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> tuple[TokenResponse, UserSession]:
        norm_email = payload.email.strip().lower()
        existing = self.db.query(User).filter(User.email == norm_email).first()
        if existing:
            raise ValueError("Email already registered")

        user = User(
            email=norm_email,
            hashed_password=hash_password(payload.password),
            full_name=payload.full_name.strip(),
            auth_provider="local",
            is_active=True,
        )
        self.db.add(user)
        self.db.flush()

        # Create default organization & workspace for the user
        clean_name = payload.full_name.strip()
        slug_base = re.sub(r"[^a-zA-Z0-9]+", "-", clean_name.lower()).strip("-") or "user"
        org = Organization(name=f"{clean_name}'s Studio", slug=f"{slug_base}-org-{secrets.token_hex(3)}")
        self.db.add(org)
        self.db.flush()

        workspace = Workspace(
            organization_id=org.id,
            name=f"{clean_name}'s Channel",
            slug=f"{slug_base}-channel-{secrets.token_hex(3)}",
        )
        self.db.add(workspace)
        self.db.flush()

        membership = Membership(
            user_id=user.id,
            workspace_id=workspace.id,
            role="owner",
        )
        self.db.add(membership)
        self.db.commit()

        session = self.create_session(user.id, ip_address=ip_address, user_agent=user_agent)
        token_resp = self._build_token_response(user, workspace, "owner", csrf_token=session.csrf_token)
        return token_resp, session

    def authenticate_user(
        self,
        payload: UserLogin,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> tuple[TokenResponse, UserSession]:
        norm_email = payload.email.strip().lower()
        user = self.db.query(User).filter(User.email == norm_email).first()
        if not user or not user.hashed_password or not verify_password(payload.password, user.hashed_password):
            raise ValueError("Invalid email or password")

        if not user.is_active:
            raise ValueError("Account is deactivated")

        # Fetch primary or first active workspace
        membership = self.db.query(Membership).filter(Membership.user_id == user.id).first()
        if not membership:
            raise ValueError("User has no active workspace")

        workspace = self.db.query(Workspace).filter(Workspace.id == membership.workspace_id).first()
        session = self.create_session(user.id, ip_address=ip_address, user_agent=user_agent)
        token_resp = self._build_token_response(user, workspace, membership.role, csrf_token=session.csrf_token)
        return token_resp, session

    def create_or_get_google_user(
        self,
        google_info: dict,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> tuple[User, UserSession, Workspace, str]:
        """Authenticate or register a user through verified Google OpenID Connect claims."""
        google_sub = str(google_info.get("sub") or "").strip()
        email = str(google_info.get("email") or "").strip().lower()
        if not email or not google_sub:
            raise ValueError("Incomplete Google profile information: missing email or sub")

        # Verify Google verified email claim
        email_verified = google_info.get("email_verified")
        if email_verified not in (True, "true", "True", 1):
            raise ValueError("Google account email is not verified by Google.")

        name = str(google_info.get("name") or email.split("@")[0]).strip()
        picture = google_info.get("picture")

        # 1. Look up existing user strictly by Google subject identifier
        user = self.db.query(User).filter(User.google_id == google_sub).first()

        if user:
            # Existing linked Google user
            if picture and not user.avatar_url:
                user.avatar_url = picture
            if not user.full_name or user.full_name == email:
                user.full_name = name
            self.db.commit()
        else:
            # 2. Check if user already exists with this email
            existing_email_user = self.db.query(User).filter(User.email == email).first()
            if existing_email_user:
                if existing_email_user.auth_provider == "local":
                    # Safe collision protection: do not silently hijack a password account
                    raise ValueError(
                        f"An account with {email} already exists using password sign-in. "
                        "Please sign in with your email and password."
                    )
                elif existing_email_user.google_id and existing_email_user.google_id != google_sub:
                    raise ValueError("This email is already associated with a different Google account.")
                else:
                    # Link account if provider is google and sub wasn't saved yet
                    existing_email_user.google_id = google_sub
                    if picture and not existing_email_user.avatar_url:
                        existing_email_user.avatar_url = picture
                    self.db.commit()
                    user = existing_email_user
            else:
                # 3. Create brand new Google user
                user = User(
                    email=email,
                    full_name=name,
                    google_id=google_sub,
                    avatar_url=picture,
                    auth_provider="google",
                    hashed_password=None,
                    is_active=True,
                )
                self.db.add(user)
                self.db.flush()

                # Create personal Org & Workspace
                slug_base = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-") or "creator"
                org = Organization(name=f"{name}'s Studio", slug=f"{slug_base}-org-{secrets.token_hex(3)}")
                self.db.add(org)
                self.db.flush()

                workspace = Workspace(
                    organization_id=org.id,
                    name=f"{name}'s Channel",
                    slug=f"{slug_base}-channel-{secrets.token_hex(3)}",
                )
                self.db.add(workspace)
                self.db.flush()

                membership = Membership(
                    user_id=user.id,
                    workspace_id=workspace.id,
                    role="owner",
                )
                self.db.add(membership)
                self.db.commit()

        # Retrieve user's primary workspace and role
        membership = self.db.query(Membership).filter(Membership.user_id == user.id).first()
        if not membership:
            # Auto-repair workspace if user had none
            slug_base = re.sub(r"[^a-zA-Z0-9]+", "-", user.full_name.lower()).strip("-") or "creator"
            org = Organization(name=f"{user.full_name}'s Studio", slug=f"{slug_base}-org-{secrets.token_hex(3)}")
            self.db.add(org)
            self.db.flush()
            workspace = Workspace(organization_id=org.id, name=f"{user.full_name}'s Channel", slug=f"{slug_base}-channel-{secrets.token_hex(3)}")
            self.db.add(workspace)
            self.db.flush()
            membership = Membership(user_id=user.id, workspace_id=workspace.id, role="owner")
            self.db.add(membership)
            self.db.commit()
        else:
            workspace = self.db.query(Workspace).filter(Workspace.id == membership.workspace_id).first()

        session = self.create_session(user.id, ip_address=ip_address, user_agent=user_agent)
        return user, session, workspace, membership.role

    def create_session(
        self,
        user_id: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> UserSession:
        """Create a server-persisted session with CSRF protection."""
        session_token = secrets.token_urlsafe(48)
        csrf_token = secrets.token_hex(32)
        expires_at = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            csrf_token=csrf_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            is_revoked=False,
        )
        self.db.add(session)
        self.db.commit()
        return session

    def get_session(self, session_token: str) -> UserSession | None:
        """Retrieve and validate session. Auto-revokes if expired."""
        session = (
            self.db.query(UserSession)
            .filter(UserSession.session_token == session_token)
            .first()
        )
        if not session and "." in session_token:
            # Fallback for JWT access token format
            payload = decode_access_token(session_token)
            if payload and payload.get("sub"):
                active = (
                    self.db.query(UserSession)
                    .filter(UserSession.user_id == payload["sub"], UserSession.is_revoked == False)
                    .order_by(UserSession.created_at.desc())
                    .first()
                )
                if active:
                    session = active

        if not session or session.is_revoked:
            return None

        # Check expiration (aware comparison)
        now = datetime.now(UTC)
        session_exp = session.expires_at
        if session_exp.tzinfo is None:
            session_exp = session_exp.replace(tzinfo=UTC)

        if session_exp < now:
            session.is_revoked = True
            self.db.commit()
            return None

        return session

    def revoke_session(self, session_token: str) -> bool:
        """Revoke a session in the database."""
        session = (
            self.db.query(UserSession)
            .filter(UserSession.session_token == session_token)
            .first()
        )
        if session and not session.is_revoked:
            session.is_revoked = True
            self.db.commit()
            return True
        return False

    def get_user_by_id(self, user_id: str) -> User | None:
        return self.db.query(User).filter(User.id == user_id).first()

    def verify_workspace_access(self, user_id: str, workspace_id: str) -> str | None:
        """Returns the user's role in the workspace, or None if no access."""
        membership = (
            self.db.query(Membership)
            .filter(Membership.user_id == user_id, Membership.workspace_id == workspace_id)
            .first()
        )
        return membership.role if membership else None

    def _build_token_response(
        self,
        user: User,
        workspace: Workspace,
        role: str,
        csrf_token: str | None = None,
    ) -> TokenResponse:
        token_data = {
            "sub": user.id,
            "email": user.email,
            "workspace_id": workspace.id,
            "role": role,
        }
        token = create_access_token(token_data)

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

        return TokenResponse(
            access_token=token,
            csrf_token=csrf_token,
            user=user_resp,
            active_workspace=ws_resp,
        )
