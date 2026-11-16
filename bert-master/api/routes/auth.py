"""用户注册 / 登录 / 信息查询"""

import os
import hmac

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from ..utils.db import mysql_conn
from ..utils.auth import (
    hash_password, verify_password, create_token, get_current_user,
)
from ..utils.rbac import normalize_role, is_reserved_conversation_school_id
from ..utils.admin_permissions import parse_permissions_blob

router = APIRouter(prefix="/auth", tags=["auth"])

# 注册角色：student | school_admin | super_admin
# 兼容旧前端传 admin → 视为 school_admin


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "student"
    display_name: str = ""
    school_id: str = ""
    identity_tag: str = ""
    admin_register_secret: Optional[str] = None
    super_admin_register_secret: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    token: str
    user: dict


def _normalize_register_role(role: str) -> str:
    r = (role or "student").strip()
    if r == "admin":
        return "school_admin"
    if r in ("student", "school_admin", "super_admin"):
        return r
    raise HTTPException(400, "角色只能是 student、school_admin 或 super_admin")


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    if not req.username or not req.password:
        raise HTTPException(400, "用户名和密码不能为空")
    db_role = _normalize_register_role(req.role)
    if len(req.password) < 4:
        raise HTTPException(400, "密码长度不能少于 4 位")

    if db_role == "school_admin":
        expected = (os.getenv("ADMIN_REGISTER_SECRET") or "").strip()
        if not expected:
            raise HTTPException(
                403,
                "校级管理员注册未开放：请在服务器 .env 配置 ADMIN_REGISTER_SECRET",
            )
        provided = req.admin_register_secret or ""
        if len(provided) != len(expected) or not hmac.compare_digest(
            provided.encode("utf-8"), expected.encode("utf-8")
        ):
            raise HTTPException(403, "校级管理员注册密钥错误")
        if not (req.school_id or "").strip():
            raise HTTPException(400, "校级管理员必须填写学校标识 school_id")

    if db_role == "super_admin":
        expected = (os.getenv("SUPER_ADMIN_REGISTER_SECRET") or "").strip()
        if not expected:
            raise HTTPException(
                403,
                "超级管理员注册未开放：请在服务器 .env 配置 SUPER_ADMIN_REGISTER_SECRET",
            )
        provided = req.super_admin_register_secret or ""
        if len(provided) != len(expected) or not hmac.compare_digest(
            provided.encode("utf-8"), expected.encode("utf-8")
        ):
            raise HTTPException(403, "超级管理员注册密钥错误")

    pw_hash = hash_password(req.password)
    school_to_store = (req.school_id or "").strip()
    if is_reserved_conversation_school_id(school_to_store):
        raise HTTPException(400, "school_id 为系统保留值，请更换为其他学校标识")
    if db_role == "super_admin":
        school_to_store = school_to_store or ""

    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库未配置或不可用")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE username=%s", (req.username,))
                if cur.fetchone():
                    raise HTTPException(409, "该用户名已被注册")
                cur.execute(
                    """INSERT INTO users
                       (username, password_hash, role, display_name, school_id, identity_tag)
                       VALUES (%s,%s,%s,%s,%s,%s)""",
                    (req.username, pw_hash, db_role,
                     req.display_name, school_to_store, req.identity_tag),
                )
            conn.commit()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, role, display_name, school_id, identity_tag, admin_permissions FROM users WHERE username=%s",
                    (req.username,),
                )
                user = cur.fetchone()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"注册失败: {e}")

    token = create_token(user["id"], user["role"], user.get("school_id", ""))
    user = dict(user)
    user["admin_permissions"] = sorted(list(parse_permissions_blob(user.get("admin_permissions"))))
    return TokenResponse(token=token, user=user)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    if not req.username or not req.password:
        raise HTTPException(400, "用户名和密码不能为空")
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库未配置或不可用")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, password_hash, role, display_name, school_id, identity_tag, admin_permissions FROM users WHERE username=%s",
                    (req.username,),
                )
                user = cur.fetchone()
        except Exception as e:
            raise HTTPException(500, f"查询失败: {e}")

    if not user:
        raise HTTPException(401, "用户名或密码错误")
    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")

    # 历史 admin → 视为 school_admin 签发 token
    r = user.get("role")
    if r == "admin":
        r = "school_admin"
    user = dict(user)
    user["role"] = normalize_role(r)

    token = create_token(user["id"], user["role"], user.get("school_id", ""))
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    safe_user["admin_permissions"] = sorted(
        list(parse_permissions_blob(safe_user.get("admin_permissions")))
    )
    return TokenResponse(token=token, user=safe_user)


@router.get("/me")
def me(request: Request):
    payload = get_current_user(request)
    if not payload:
        raise HTTPException(401, "未登录或 token 已过期")
    uid = payload.get("uid")
    with mysql_conn() as conn:
        if conn is None:
            return {"uid": uid, "role": payload.get("role")}
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, role, display_name, school_id, identity_tag, admin_permissions FROM users WHERE id=%s",
                    (uid,),
                )
                user = cur.fetchone()
        except Exception:
            return {"uid": uid, "role": payload.get("role")}
    if not user:
        raise HTTPException(401, "用户不存在")
    if user.get("role") == "admin":
        user = dict(user)
        user["role"] = "school_admin"
    user = dict(user)
    user["admin_permissions"] = sorted(list(parse_permissions_blob(user.get("admin_permissions"))))
    return user
