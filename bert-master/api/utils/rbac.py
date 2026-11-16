"""角色与多租户权限（各校管各校 + 超级管理员全量 + 平台内部对话桶）"""

from typing import Optional, Tuple

# 数据库存 role；兼容历史值 admin
_ROLE_ALIASES = {"admin": "school_admin"}

# 超管 / 未绑定学校的管理员对话写入此 school_id，仅 super_admin 可查，校级管理员不可见
PLATFORM_CONVERSATION_SCHOOL_ID = "__CM_PLATFORM__"


def normalize_role(role: Optional[str]) -> str:
    if not role:
        return "student"
    r = str(role).strip()
    return _ROLE_ALIASES.get(r, r)


def is_student(role: Optional[str]) -> bool:
    return normalize_role(role) == "student"


def is_school_admin(role: Optional[str]) -> bool:
    return normalize_role(role) == "school_admin"


def is_super_admin(role: Optional[str]) -> bool:
    return normalize_role(role) == "super_admin"


def is_any_admin(role: Optional[str]) -> bool:
    return is_school_admin(role) or is_super_admin(role)


def tenant_filter_school_id(viewer_role: str, viewer_school_id: str) -> Tuple[str, tuple]:
    """
    返回 SQL WHERE 片段与参数（对话 / 风险统计等）。
    - super_admin: 无租户限制（全量）
    - school_admin: 仅 school_id 与本校完全一致（不含公共池、不含平台桶、不含他校）
    - 校级未绑定 school：不可见任何行（避免误放开）
    """
    r = normalize_role(viewer_role)
    if is_super_admin(r):
        return "", ()
    if not is_school_admin(r):
        return ("1=0", ())
    sid = (viewer_school_id or "").strip()
    if not sid:
        return ("1=0", ())
    if sid == PLATFORM_CONVERSATION_SCHOOL_ID:
        return ("1=0", ())
    return ("school_id = %s", (sid,))


def is_reserved_conversation_school_id(school_id: Optional[str]) -> bool:
    return (school_id or "").strip() == PLATFORM_CONVERSATION_SCHOOL_ID


def questionnaire_manage_allowed(
    viewer_role: str,
    viewer_school_id: str,
    questionnaire_school_id: Optional[str],
    viewer_permissions=None,
) -> bool:
    """是否允许启停/编辑/删除该问卷模板。"""
    r = normalize_role(viewer_role)
    if is_super_admin(r):
        return True
    if not is_school_admin(r):
        return False
    perms = set(viewer_permissions or [])
    if "survey_manage" not in perms:
        return False
    qs = (questionnaire_school_id or "").strip()
    my = (viewer_school_id or "").strip()
    # 校级管理员不可改全局公共模板
    if qs == "":
        return False
    return qs == my
