"""
对话存储与查询工具（含 school_id 多租户 + 公共池）。
"""

import datetime as _dt
import logging
from typing import List, Dict, Any, Optional

from .db import mysql_conn
from .rbac import normalize_role, is_super_admin, tenant_filter_school_id

logger = logging.getLogger(__name__)


def save_conversation(
    identity: str,
    user_text: str,
    mood_state: str,
    top_emotion: str,
    probability: float,
    risk_level: str,
    intervention: str,
    user_id: int = None,
    school_id: str = "",
) -> None:
    """保存一条对话；匿名用户 school_id 为空字符串表示公共池。"""
    with mysql_conn() as conn:
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations
                      (user_id, user_identity, school_id, created_at, user_text, mood_state,
                       top_emotion, probability, risk_level, intervention)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        identity or None,
                        (school_id or "").strip() or "",
                        _dt.datetime.utcnow(),
                        user_text,
                        (mood_state or "").strip()[:64] or None,
                        top_emotion,
                        probability,
                        risk_level,
                        intervention,
                    ),
                )
            conn.commit()
        except Exception as e:
            logger.warning(f"保存对话记录失败: {e}")


def get_chat_context_messages(
    user_id: int,
    school_id: str,
    *,
    max_turns: int = 6,
) -> List[Dict[str, str]]:
    """
    为已登录用户拉取近期对话，供大模型多轮上下文使用。

    每条库记录含 user_text 与 intervention，展开为 user → assistant 交替。
    按时间从早到晚排列；不含尚未写入的当前轮。

    max_turns: 最多取最近多少条 conversations 行（非 token 上限）。
    """
    if user_id is None or int(user_id) <= 0:
        return []
    try:
        max_turns = max(1, min(int(max_turns), 20))
    except (TypeError, ValueError):
        max_turns = 6
    sid = (school_id or "").strip()
    rows: List[Dict[str, Any]] = []
    with mysql_conn() as conn:
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_text, intervention
                    FROM conversations
                    WHERE user_id = %s AND school_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (int(user_id), sid, int(max_turns)),
                )
                rows = list(reversed(cur.fetchall() or []))
        except Exception as e:
            logger.warning(f"读取对话上下文失败: {e}")
            return []
    out: List[Dict[str, str]] = []
    for row in rows:
        ut = (row.get("user_text") or "").strip()
        bot = (row.get("intervention") or "").strip()
        if ut:
            out.append({"role": "user", "content": ut})
        if bot:
            out.append({"role": "assistant", "content": bot})
    return out


def get_user_conversation_history_rows(
    user_id: int,
    school_id: str,
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    已登录用户 + 学校租户：按时间正序返回最近若干条对话行，供前端展示。

    limit：最多取最近多少条 conversations 行（与多轮上下文的「行数」口径一致），1～20。
    """
    if user_id is None or int(user_id) <= 0:
        return []
    try:
        limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        limit = 10
    sid = (school_id or "").strip()
    rows: List[Dict[str, Any]] = []
    with mysql_conn() as conn:
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_text, intervention, top_emotion, risk_level,
                           user_identity, created_at
                    FROM conversations
                    WHERE user_id = %s AND school_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (int(user_id), sid, int(limit)),
                )
                raw = list(reversed(cur.fetchall() or []))
        except Exception as e:
            logger.warning(f"读取用户对话展示列表失败: {e}")
            return []
    for row in raw:
        ca = row.get("created_at")
        if ca is not None and hasattr(ca, "isoformat"):
            row = dict(row)
            row["created_at"] = ca.isoformat() + ("Z" if ca.tzinfo is None else "")
        rows.append(row)
    return rows


def get_recent_conversations(limit: int = 50) -> List[Dict[str, Any]]:
    """兼容旧接口：无鉴权场景返回空或慎用。"""
    return get_recent_conversations_filtered(
        limit=limit, viewer_role="super_admin", viewer_school_id=""
    )


def get_recent_conversations_filtered(
    limit: int,
    viewer_role: str,
    viewer_school_id: str,
) -> List[Dict[str, Any]]:
    """按管理员租户过滤：校级=本校+公共池；超级=全部。"""
    vr = normalize_role(viewer_role)
    with mysql_conn() as conn:
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                if is_super_admin(vr):
                    cur.execute(
                        """
                        SELECT c.id, c.user_identity, c.school_id, c.created_at, c.user_text,
                               c.mood_state, c.top_emotion, c.probability, c.risk_level,
                               u.username, u.display_name
                        FROM conversations c
                        LEFT JOIN users u ON c.user_id=u.id
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (int(limit),),
                    )
                else:
                    clause, params = tenant_filter_school_id(vr, viewer_school_id)
                    cur.execute(
                        f"""
                        SELECT c.id, c.user_identity, c.school_id, c.created_at, c.user_text,
                               c.mood_state, c.top_emotion, c.probability, c.risk_level,
                               u.username, u.display_name
                        FROM conversations c
                        LEFT JOIN users u ON c.user_id=u.id
                        WHERE {clause.replace('school_id', 'c.school_id')}
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        params + (int(limit),),
                    )
                return cur.fetchall() or []
        except Exception as e:
            logger.warning(f"查询对话记录失败: {e}")
            return []


def get_risk_stats(viewer_role: str, viewer_school_id: str) -> Dict[str, Any]:
    """按租户统计风险等级数量（用于管理端预警看板）。"""
    vr = normalize_role(viewer_role)
    with mysql_conn() as conn:
        if conn is None:
            return {
                "normal": 0,
                "warning": 0,
                "crisis": 0,
                "attention": 0,
                "intervention": 0,
                "total": 0,
            }
        try:
            with conn.cursor() as cur:
                if is_super_admin(vr):
                    cur.execute(
                        """
                        SELECT LOWER(COALESCE(risk_level,'normal')) AS rl, COUNT(*) AS c
                        FROM conversations
                        GROUP BY rl
                        """
                    )
                else:
                    clause, params = tenant_filter_school_id(vr, viewer_school_id)
                    cur.execute(
                        f"""
                        SELECT LOWER(COALESCE(risk_level,'normal')) AS rl, COUNT(*) AS c
                        FROM conversations
                        WHERE {clause}
                        GROUP BY rl
                        """,
                        params,
                    )
                rows = cur.fetchall() or []
            out = {
                "normal": 0,
                "warning": 0,
                "crisis": 0,
                "attention": 0,
                "intervention": 0,
                "total": 0,
            }
            for row in rows:
                rl = (row.get("rl") or "normal").lower()
                c = int(row.get("c") or 0)
                out["total"] += c
                if rl == "normal":
                    out["normal"] += c
                elif rl == "crisis":
                    out["crisis"] += c
                elif rl == "attention":
                    out["attention"] += c
                    out["warning"] += c
                elif rl == "intervention":
                    out["intervention"] += c
                    out["warning"] += c
                elif rl == "warning":
                    out["warning"] += c
                else:
                    out["warning"] += c
            return out
        except Exception as e:
            logger.warning(f"风险统计失败: {e}")
            return {
                "normal": 0,
                "warning": 0,
                "crisis": 0,
                "attention": 0,
                "intervention": 0,
                "total": 0,
            }
