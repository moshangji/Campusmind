"""管理员细粒度权限工具。"""

import json
from typing import Iterable, Set, Optional

from .db import mysql_conn
from .rbac import normalize_role, is_super_admin, is_school_admin

PERM_SURVEY_MANAGE = "survey_manage"
PERM_RISK_VIEW = "risk_lexicon_view"
PERM_RISK_EDIT = "risk_lexicon_edit"
PERM_RISK_MANAGE = "risk_lexicon_manage"
PERM_STUDENT_PROFILE_VIEW = "student_profile_view"

ALL_PERMISSIONS = {
    PERM_SURVEY_MANAGE,
    PERM_RISK_VIEW,
    PERM_RISK_EDIT,
    PERM_RISK_MANAGE,
    PERM_STUDENT_PROFILE_VIEW,
}


def normalize_permissions(perms: Iterable[str]) -> Set[str]:
    out = set()
    for p in perms or []:
        k = str(p or "").strip().lower()
        if k in ALL_PERMISSIONS:
            out.add(k)
    return out


def parse_permissions_blob(raw) -> Set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return normalize_permissions(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return set()
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return normalize_permissions(obj)
        except Exception:
            pass
        return normalize_permissions(s.split(","))
    return set()


def serialize_permissions(perms: Iterable[str]) -> str:
    return json.dumps(sorted(list(normalize_permissions(perms))), ensure_ascii=False)


def load_user_permissions(uid: Optional[int]) -> Set[str]:
    if not uid:
        return set()
    with mysql_conn() as conn:
        if conn is None:
            return set()
        with conn.cursor() as cur:
            cur.execute("SELECT admin_permissions FROM users WHERE id=%s", (uid,))
            row = cur.fetchone()
            return parse_permissions_blob((row or {}).get("admin_permissions"))


def has_permission(role: str, perms: Set[str], wanted: str) -> bool:
    r = normalize_role(role)
    if is_super_admin(r):
        return True
    if not is_school_admin(r):
        return False
    return wanted in normalize_permissions(perms)
