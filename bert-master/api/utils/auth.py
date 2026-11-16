"""JWT 认证工具 — 纯标准库实现，无需 PyJWT 依赖"""

import os
import hashlib
import hmac
import time
import json
import base64
import logging
from typing import Optional, Dict, Any

from fastapi import Request

logger = logging.getLogger(__name__)

_JWT_SECRET: Optional[str] = None


def _get_secret() -> str:
    global _JWT_SECRET
    if _JWT_SECRET is None:
        _JWT_SECRET = os.getenv("JWT_SECRET", "campusmind-default-secret-change-me")
    return _JWT_SECRET


def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return salt + ":" + hashed


def verify_password(password: str, stored: str) -> bool:
    parts = stored.split(":", 1)
    if len(parts) != 2:
        return False
    salt, expected = parts
    return hmac.compare_digest(
        hashlib.sha256((salt + password).encode()).hexdigest(), expected
    )


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def create_token(user_id: int, role: str, school_id: str = "",
                 expire_hours: int = 72) -> str:
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload_dict = {
        "uid": user_id,
        "role": role,
        "school": school_id or "",
        "exp": int(time.time()) + expire_hours * 3600,
    }
    payload = _b64url_encode(json.dumps(payload_dict).encode())
    sig_bytes = hmac.new(
        _get_secret().encode(), (header + "." + payload).encode(), hashlib.sha256
    ).digest()
    sig = _b64url_encode(sig_bytes)
    return f"{header}.{payload}.{sig}"


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, payload, sig = parts
        expected = hmac.new(
            _get_secret().encode(), (header + "." + payload).encode(), hashlib.sha256
        ).digest()
        expected_sig = _b64url_encode(expected)
        if not hmac.compare_digest(sig, expected_sig):
            return None
        data = json.loads(_b64url_decode(payload))
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """从 Authorization 头提取用户信息；未登录返回 None（允许匿名）"""
    auth = request.headers.get("Authorization", "")
    if not auth:
        return None
    # Python 3.8 没有 str.removeprefix
    token = auth.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return verify_token(token)
