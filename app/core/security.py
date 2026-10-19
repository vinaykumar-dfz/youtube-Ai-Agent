"""Compatibility shim pointing to app.security.crypto."""
from app.security.crypto import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
    get_safe_redirect_url,
)

__all__ = [
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
    "get_safe_redirect_url",
]
