"""超级管理员：管理校级管理员账号（各校管各校的上级能力）"""

import re
import json
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..utils.db import mysql_conn
from ..utils.auth import get_current_user, hash_password
from ..utils.rbac import (
    is_super_admin,
    is_school_admin,
    normalize_role,
    is_any_admin,
    is_reserved_conversation_school_id,
)
from ..utils.admin_permissions import (
    PERM_RISK_VIEW,
    PERM_RISK_EDIT,
    PERM_RISK_MANAGE,
    PERM_STUDENT_PROFILE_VIEW,
    ALL_PERMISSIONS,
    parse_permissions_blob,
    serialize_permissions,
    has_permission,
)

router = APIRouter(prefix="/admin", tags=["admin-mgmt"])

RISK_LEXICON_CATEGORIES = frozenset(
    {"crisis_zh", "crisis_en", "crisis_regex", "negative", "metaphor", "homophone"}
)


class CreateSchoolAdminBody(BaseModel):
    username: str
    password: str
    display_name: str = ""
    school_id: str
    identity_tag: str = "教师"


class UpdateUserBody(BaseModel):
    display_name: Optional[str] = None
    school_id: Optional[str] = None
    password: Optional[str] = None


class UpdateAdminPermissionsBody(BaseModel):
    permissions: List[str] = []


class RiskLexiconCreate(BaseModel):
    phrase: str = Field(..., min_length=1, max_length=512)
    category: str
    is_active: bool = True
    sort_order: int = 0


class RiskLexiconUpdate(BaseModel):
    phrase: Optional[str] = Field(None, min_length=1, max_length=512)
    category: Optional[str] = None
    is_active: Optional[bool] = None
    sort_order: Optional[int] = None


class RiskConfigUpdate(BaseModel):
    attention_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    intervention_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


def _require_admin_with_permission(request: Request, wanted: str):
    user = get_current_user(request)
    if not user or not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    uid = user.get("uid")
    perms = set()
    with mysql_conn() as conn:
        if conn is not None and uid:
            with conn.cursor() as cur:
                cur.execute("SELECT admin_permissions FROM users WHERE id=%s", (uid,))
                row = cur.fetchone()
                perms = parse_permissions_blob((row or {}).get("admin_permissions"))
    if not has_permission(user.get("role"), perms, wanted):
        raise HTTPException(403, "当前账号暂无该操作权限")
    return user


def _normalize_lexicon_category(raw: str) -> str:
    c = (raw or "").strip().lower()
    if c not in RISK_LEXICON_CATEGORIES:
        raise HTTPException(
            400,
            f"category 必须是: {', '.join(sorted(RISK_LEXICON_CATEGORIES))}",
        )
    return c


def _validate_phrase_for_category(phrase: str, category: str) -> str:
    p = (phrase or "").strip()
    if not p:
        raise HTTPException(400, "phrase 不能为空")
    if category == "crisis_regex":
        try:
            re.compile(p)
        except re.error as e:
            raise HTTPException(400, f"正则无效: {e}") from e
    return p


def _risk_reload_after_mutation():
    from ..utils.risk_lexicon import reset_pipelines_after_lexicon_change

    reset_pipelines_after_lexicon_change()


@router.get("/users")
def list_managed_users(request: Request, role_filter: Optional[str] = None):
    """
    超级管理员：列出用户（可筛 role）。
    校级管理员：仅可查看本校学生列表。
    """
    user = get_current_user(request)
    if not user or not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    vr = normalize_role(user.get("role"))
    my_school = (user.get("school") or "").strip()
    with mysql_conn() as conn:
        if conn is None:
            return {"items": []}
        with conn.cursor() as cur:
            if is_super_admin(vr):
                if role_filter:
                    cur.execute(
                        """SELECT id, username, role, display_name, school_id, identity_tag, admin_permissions, created_at
                           FROM users WHERE role=%s ORDER BY id DESC LIMIT 500""",
                        (role_filter,),
                    )
                else:
                    cur.execute(
                        """SELECT id, username, role, display_name, school_id, identity_tag, admin_permissions, created_at
                           FROM users ORDER BY id DESC LIMIT 500""",
                    )
            else:
                if not my_school:
                    return {"items": []}
                if role_filter and role_filter not in ("student", "school_admin"):
                    return {"items": []}
                if role_filter == "school_admin":
                    cur.execute(
                        """SELECT id, username, role, display_name, school_id, identity_tag, admin_permissions, created_at
                           FROM users
                           WHERE role='school_admin' AND school_id=%s
                           ORDER BY id DESC LIMIT 500""",
                        (my_school,),
                    )
                else:
                    cur.execute(
                        """SELECT id, username, role, display_name, school_id, identity_tag, admin_permissions, created_at
                           FROM users
                           WHERE role='student' AND school_id=%s
                           ORDER BY id DESC LIMIT 500""",
                        (my_school,),
                    )
            items = cur.fetchall() or []
            for row in items:
                row["admin_permissions"] = sorted(list(parse_permissions_blob(row.get("admin_permissions"))))
            return {"items": items}


@router.post("/school-admins")
def create_school_admin(req: CreateSchoolAdminBody, request: Request):
    """仅超级管理员：创建校级管理员账号。"""
    actor = get_current_user(request)
    if not actor or not is_super_admin(actor.get("role")):
        raise HTTPException(403, "仅超级管理员可创建校级管理员")
    if not req.username or not req.password or len(req.password) < 4:
        raise HTTPException(400, "用户名与密码不合法")
    sid = (req.school_id or "").strip()
    if not sid:
        raise HTTPException(400, "school_id 不能为空")
    if is_reserved_conversation_school_id(sid):
        raise HTTPException(400, "school_id 为系统保留值，请更换")
    pw = hash_password(req.password)
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username=%s", (req.username,))
            if cur.fetchone():
                raise HTTPException(409, "用户名已存在")
            cur.execute(
                """INSERT INTO users
                   (username, password_hash, role, display_name, school_id, identity_tag, admin_permissions)
                   VALUES (%s,%s,'school_admin',%s,%s,%s,%s)""",
                (
                    req.username, pw, req.display_name, sid, req.identity_tag or "教师",
                    json.dumps([], ensure_ascii=False),
                ),
            )
        conn.commit()
    return {"message": "校级管理员已创建", "username": req.username, "school_id": sid}


@router.put("/users/{uid}")
def update_user(uid: int, req: UpdateUserBody, request: Request):
    """超级管理员：修改任意用户；校级管理员不可调用本接口（后续可扩展）。"""
    actor = get_current_user(request)
    if not actor or not is_super_admin(actor.get("role")):
        raise HTTPException(403, "仅超级管理员可修改用户")
    fields = []
    params = []
    if req.display_name is not None:
        fields.append("display_name=%s")
        params.append(req.display_name)
    if req.school_id is not None:
        ns = req.school_id.strip()
        if is_reserved_conversation_school_id(ns):
            raise HTTPException(400, "school_id 为系统保留值")
        fields.append("school_id=%s")
        params.append(ns)
    if req.password:
        if len(req.password) < 4:
            raise HTTPException(400, "密码过短")
        fields.append("password_hash=%s")
        params.append(hash_password(req.password))
    if not fields:
        raise HTTPException(400, "无更新字段")
    params.append(uid)
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE id=%s", tuple(params))
            if cur.rowcount == 0:
                raise HTTPException(404, "用户不存在")
        conn.commit()
    return {"message": "已更新", "id": uid}


@router.put("/users/{uid}/permissions")
def update_admin_permissions(uid: int, req: UpdateAdminPermissionsBody, request: Request):
    """给校级管理员授予细粒度权限（超管全局；校管仅同校且仅 profile_view）。"""
    actor = get_current_user(request)
    if not actor or not is_any_admin(actor.get("role")):
        raise HTTPException(403, "仅管理员可授权")
    actor_role = normalize_role(actor.get("role"))
    actor_school = (actor.get("school") or "").strip()
    perms = sorted(list(set([p for p in req.permissions if p in ALL_PERMISSIONS])))
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute("SELECT id, role, school_id FROM users WHERE id=%s", (uid,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "用户不存在")
            if not is_school_admin(row.get("role")):
                raise HTTPException(400, "仅可给校级管理员授权")
            if not is_super_admin(actor_role):
                if not is_school_admin(actor_role):
                    raise HTTPException(403, "仅管理员可授权")
                if not actor_school or actor_school != (row.get("school_id") or "").strip():
                    raise HTTPException(403, "仅可给同校管理员授权")
                # 校级管理员只能授予/回收学生画像查看权限
                invalid = [p for p in perms if p != PERM_STUDENT_PROFILE_VIEW]
                if invalid:
                    raise HTTPException(403, "校级管理员仅可配置 student_profile_view 权限")
            cur.execute(
                "UPDATE users SET admin_permissions=%s WHERE id=%s",
                (serialize_permissions(perms), uid),
            )
        conn.commit()
    return {"message": "权限已更新", "id": uid, "permissions": perms}


def _risk_to_score(level: str) -> int:
    lv = (level or "").strip().lower()
    if lv == "crisis":
        return 3
    if lv == "intervention":
        return 2
    if lv == "attention":
        return 1
    return 0


@router.get("/users/{uid}/profile")
def get_user_profile(uid: int, request: Request, days: int = 30, recent_limit: int = 20):
    """个人画像：最近聊天、情绪占比、风险占比与按日曲线。"""
    viewer = get_current_user(request)
    if not viewer or not is_any_admin(viewer.get("role")):
        raise HTTPException(403, "需要管理员权限")
    vr = normalize_role(viewer.get("role"))
    viewer_school = (viewer.get("school") or "").strip()
    d = max(7, min(int(days or 30), 180))
    rlim = max(5, min(int(recent_limit or 20), 100))
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, role, display_name, school_id, identity_tag FROM users WHERE id=%s",
                (uid,),
            )
            target = cur.fetchone()
            if not target:
                raise HTTPException(404, "用户不存在")
            if normalize_role(target.get("role")) != "student":
                raise HTTPException(400, "仅支持查看学生画像")
            target_school = (target.get("school_id") or "").strip()
            if not is_super_admin(vr):
                if not is_school_admin(vr):
                    raise HTTPException(403, "无权限")
                my_perms = set()
                cur.execute("SELECT admin_permissions FROM users WHERE id=%s", (viewer.get("uid"),))
                viewer_row = cur.fetchone() or {}
                my_perms = parse_permissions_blob(viewer_row.get("admin_permissions"))
                if PERM_STUDENT_PROFILE_VIEW not in my_perms:
                    raise HTTPException(403, "缺少 student_profile_view 权限")
                if not viewer_school or viewer_school != target_school:
                    raise HTTPException(403, "仅可查看本校学生")

            cur.execute(
                """
                SELECT id, created_at, user_text, mood_state, top_emotion, probability, risk_level, intervention
                FROM conversations
                WHERE user_id=%s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (uid, rlim),
            )
            recent_items = cur.fetchall() or []

            cur.execute(
                """
                SELECT LOWER(COALESCE(top_emotion,'')) AS emotion, COUNT(*) AS c
                FROM conversations
                WHERE user_id=%s
                GROUP BY emotion
                ORDER BY c DESC
                LIMIT 8
                """,
                (uid,),
            )
            emo_rows = cur.fetchall() or []
            emo_total = sum(int(x.get("c") or 0) for x in emo_rows)
            emotion_ratio = [
                {
                    "emotion": (x.get("emotion") or "unknown"),
                    "count": int(x.get("c") or 0),
                    "ratio": round((int(x.get("c") or 0) / emo_total), 4) if emo_total else 0.0,
                }
                for x in emo_rows
            ]

            cur.execute(
                """
                SELECT LOWER(COALESCE(risk_level,'normal')) AS risk_level, COUNT(*) AS c
                FROM conversations
                WHERE user_id=%s
                GROUP BY risk_level
                """,
                (uid,),
            )
            risk_rows = cur.fetchall() or []
            risk_total = sum(int(x.get("c") or 0) for x in risk_rows)
            risk_ratio = [
                {
                    "risk_level": (x.get("risk_level") or "normal"),
                    "count": int(x.get("c") or 0),
                    "ratio": round((int(x.get("c") or 0) / risk_total), 4) if risk_total else 0.0,
                }
                for x in risk_rows
            ]

            cur.execute(
                """
                SELECT DATE(created_at) AS d,
                       AVG(probability) AS avg_prob,
                       AVG(
                         CASE
                           WHEN LOWER(COALESCE(risk_level,''))='crisis' THEN 3
                           WHEN LOWER(COALESCE(risk_level,''))='intervention' THEN 2
                           WHEN LOWER(COALESCE(risk_level,''))='attention' THEN 1
                           ELSE 0
                         END
                       ) AS avg_risk_score
                FROM conversations
                WHERE user_id=%s
                  AND created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL %s DAY)
                GROUP BY DATE(created_at)
                ORDER BY d
                """,
                (uid, d),
            )
            trend_rows = cur.fetchall() or []

    trend = {
        "labels": [str(x.get("d")) for x in trend_rows],
        "emotion_prob": [round(float(x.get("avg_prob") or 0.0), 4) for x in trend_rows],
        "intervention_score": [round(float(x.get("avg_risk_score") or 0.0), 4) for x in trend_rows],
    }

    for row in recent_items:
        row["risk_score"] = _risk_to_score(row.get("risk_level"))

    return {
        "user": target,
        "recent_conversations": recent_items,
        "emotion_ratio": emotion_ratio,
        "risk_ratio": risk_ratio,
        "trend": trend,
        "days": d,
    }


@router.get("/risk-lexicon")
def get_risk_lexicon(request: Request, limit: int = 800):
    """查看当前库中的风险词表与阈值。"""
    _require_admin_with_permission(request, PERM_RISK_VIEW)
    lim = max(1, min(int(limit), 2000))
    with mysql_conn() as conn:
        if conn is None:
            return {
                "db_available": False,
                "items": [],
                "config": {},
                "hint": "未配置 MySQL 或未安装 pymysql，系统使用代码内置词表。",
            }
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, phrase, category, is_active, sort_order, created_at
                    FROM risk_lexicon
                    ORDER BY category, sort_order, id
                    LIMIT %s
                    """,
                    (lim,),
                )
                items = cur.fetchall() or []
                cur.execute("SELECT config_key, config_value FROM risk_config")
                cfg_rows = cur.fetchall() or []
        except Exception as e:
            raise HTTPException(503, f"读取风险词表失败: {e}") from e
    cfg = {r.get("config_key"): r.get("config_value") for r in (cfg_rows or [])}
    return {"db_available": True, "items": items, "config": cfg, "hint": None}


@router.post("/risk-lexicon")
def create_risk_lexicon_row(req: RiskLexiconCreate, request: Request):
    """新增风险词表条目。保存后自动热更新管线。"""
    _require_admin_with_permission(request, PERM_RISK_EDIT)
    cat = _normalize_lexicon_category(req.category)
    phrase = _validate_phrase_for_category(req.phrase, cat)
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO risk_lexicon (phrase, category, is_active, sort_order)
                VALUES (%s,%s,%s,%s)
                """,
                (phrase, cat, bool(req.is_active), int(req.sort_order)),
            )
            new_id = cur.lastrowid
        conn.commit()
    _risk_reload_after_mutation()
    return {"message": "已新增", "id": new_id}


@router.put("/risk-lexicon/{row_id}")
def update_risk_lexicon_row(row_id: int, req: RiskLexiconUpdate, request: Request):
    """更新风险词表条目。保存后自动热更新管线。"""
    _require_admin_with_permission(request, PERM_RISK_EDIT)
    fields, params = [], []
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT phrase, category FROM risk_lexicon WHERE id=%s",
                (row_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "记录不存在")
            eff_cat = _normalize_lexicon_category(
                req.category if req.category is not None else (row.get("category") or "")
            )
            phrase_src = (
                req.phrase if req.phrase is not None else (row.get("phrase") or "")
            )
            phrase_ok = _validate_phrase_for_category(phrase_src, eff_cat)
            if req.category is not None:
                fields.append("category=%s")
                params.append(eff_cat)
            if req.phrase is not None:
                fields.append("phrase=%s")
                params.append(phrase_ok)
            if req.is_active is not None:
                fields.append("is_active=%s")
                params.append(bool(req.is_active))
            if req.sort_order is not None:
                fields.append("sort_order=%s")
                params.append(int(req.sort_order))
            if not fields:
                raise HTTPException(400, "无更新字段")
            params.append(row_id)
            cur.execute(
                f"UPDATE risk_lexicon SET {', '.join(fields)} WHERE id=%s",
                tuple(params),
            )
        conn.commit()
    _risk_reload_after_mutation()
    return {"message": "已更新", "id": row_id}


@router.delete("/risk-lexicon/{row_id}")
def delete_risk_lexicon_row(row_id: int, request: Request):
    """删除风险词表条目。删除后自动热更新管线。"""
    _require_admin_with_permission(request, PERM_RISK_MANAGE)
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM risk_lexicon WHERE id=%s", (row_id,))
            if cur.rowcount == 0:
                raise HTTPException(404, "记录不存在")
        conn.commit()
    _risk_reload_after_mutation()
    return {"message": "已删除", "id": row_id}


@router.put("/risk-config")
def update_risk_config(req: RiskConfigUpdate, request: Request):
    """更新 risk_config 中的阈值（关注级 / 干预级）。保存后自动热更新。"""
    _require_admin_with_permission(request, PERM_RISK_MANAGE)
    updates = (
        req.model_dump(exclude_none=True)
        if hasattr(req, "model_dump")
        else req.dict(exclude_none=True)
    )
    if not updates:
        raise HTTPException(400, "无更新字段")
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            for key, val in updates.items():
                v = f"{float(val):.4f}".rstrip("0").rstrip(".")
                cur.execute(
                    """
                    INSERT INTO risk_config (config_key, config_value)
                    VALUES (%s,%s)
                    ON DUPLICATE KEY UPDATE config_value=VALUES(config_value)
                    """,
                    (key, v),
                )
        conn.commit()
    _risk_reload_after_mutation()
    return {"message": "阈值已保存", "config": updates}


@router.post("/risk-lexicon/reload")
def reload_risk_lexicon(request: Request):
    """
    手动刷新缓存（一般增删改已自动刷新）。仅超级管理员。
    """
    _require_admin_with_permission(request, PERM_RISK_MANAGE)
    from ..utils.risk_lexicon import reset_pipelines_after_lexicon_change

    reset_pipelines_after_lexicon_change()
    return {
        "message": "已清空缓存并重置分析管线，后续请求将使用数据库中的 risk_lexicon / risk_config",
    }
