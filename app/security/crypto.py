import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import settings


def hash_password(password: str, salt: str | None = None) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with a unique salt."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    ).hex()
    return f"{salt}${hashed}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """Verify a plain password against the stored salt$hash."""
    try:
        salt, hashed = stored_hash.split("$", 1)
        expected = hash_password(plain_password, salt=salt)
        return hmac.compare_digest(stored_hash, expected)
    except Exception:
        return False


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a URL-safe signed session token (Header.Payload.Signature)."""
    to_encode = data.copy()
    now = datetime.now(UTC)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": int(expire.timestamp()), "iat": int(now.timestamp())})

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(to_encode).encode()).decode().rstrip("=")

    message = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(settings.SECRET_KEY.encode(), message, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and verify a signed session token. Returns payload or None if invalid/expired."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(settings.SECRET_KEY.encode(), message, hashlib.sha256).digest()
        actual_sig = base64.urlsafe_b64decode(sig_b64 + "=="[: (4 - len(sig_b64) % 4) % 4])

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        # Decode payload
        payload_json = base64.urlsafe_b64decode(
            payload_b64 + "=="[: (4 - len(payload_b64) % 4) % 4]
        ).decode()
        payload = json.loads(payload_json)

        # Check expiration
        exp = payload.get("exp")
        if exp and exp < int(time.time()):
            return None

        return payload
    except Exception:
        return None


def get_safe_redirect_url(target: str | None, default: str = "/dashboard") -> str:
    """Validate and sanitize a redirect URL to prevent open redirect vulnerabilities."""
    if not target or not isinstance(target, str):
        return default

    clean = target.strip()
    try:
        import urllib.parse
        clean = urllib.parse.unquote(clean)
    except Exception:
        return default

    # Must start with single '/', no protocol-relative '//', no backslashes
    if not clean.startswith("/") or clean.startswith("//") or clean.startswith("/\\"):
        return default

    if "\\" in clean or any(ord(c) < 32 or ord(c) == 127 for c in clean):
        return default

    try:
        import urllib.parse
        parsed = urllib.parse.urlsplit(clean)
        if parsed.scheme or parsed.netloc:
            return default
    except Exception:
        return default

    return clean

