from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=2)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class WorkspaceResponse(BaseModel):
    id: str
    organization_id: str
    name: str
    slug: str
    role: str

    model_config = {"from_attributes": True}


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    avatar_url: str | None = None
    auth_provider: str = "local"
    is_active: bool
    workspaces: list[WorkspaceResponse] = []
    csrf_token: str | None = None

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    csrf_token: str | None = None
    user: UserResponse
    active_workspace: WorkspaceResponse


class GoogleCredentialRequest(BaseModel):
    credential: str = Field(description="Google Identity Services ID Token (JWT)")


class SessionResponse(BaseModel):
    user: UserResponse
    active_workspace: WorkspaceResponse
    csrf_token: str


class LogoutResponse(BaseModel):
    message: str = "Successfully logged out"
    status: str = "success"


class GoogleConfigResponse(BaseModel):
    google_client_id: str
    is_configured: bool
