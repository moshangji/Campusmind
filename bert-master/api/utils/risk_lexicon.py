"""
风险词表与阈值：优先从 MySQL（risk_lexicon / risk_config）加载，供 SensitiveWordDetector / RiskAssessor 使用。
修改库表后可调用 POST /api/v1/admin/risk-lexicon/reload（仅超管）或重启进程。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .db import mysql_conn

logger = logging.getLogger(__name__)

_SNAPSHOT_CACHE: Optional["RiskLexiconSnapshot"] = None


@dataclass
class RiskLexiconSnapshot:
    crisis_phrases_zh: List[str] = field(default_factory=list)
    crisis_phrases_en: List[str] = field(default_factory=list)
    crisis_regex_patterns: List[str] = field(default_factory=list)
    words_negative: List[str] = field(default_factory=list)
    words_metaphor: List[str] = field(default_factory=list)
    words_homophone: List[str] = field(default_factory=list)
    attention_threshold: float = 0.35
    intervention_threshold: float = 0.52


def clear_risk_lexicon_cache() -> None:
    global _SNAPSHOT_CACHE
    _SNAPSHOT_CACHE = None


def _fetch_snapshot_from_db() -> Optional[RiskLexiconSnapshot]:
    with mysql_conn() as conn:
        if conn is None:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS c FROM risk_lexicon WHERE is_active=TRUE"
                )
                cnt_row = cur.fetchone() or {}
                if int(cnt_row.get("c") or 0) <= 0:
                    return None
                cur.execute(
                    """
                    SELECT phrase, category FROM risk_lexicon
                    WHERE is_active=TRUE
                    ORDER BY category, sort_order, id
                    """
                )
                rows = cur.fetchall() or []
                cur.execute("SELECT config_key, config_value FROM risk_config")
                cfg_rows = cur.fetchall() or []
        except Exception as e:
            logger.warning(f"读取 risk_lexicon / risk_config 失败，将使用代码内置词表: {e}")
            return None

    cfg: Dict[str, str] = {
        (r.get("config_key") or ""): (r.get("config_value") or "")
        for r in cfg_rows
    }
    snap = RiskLexiconSnapshot()
    try:
        snap.attention_threshold = float(
            cfg.get("attention_threshold", snap.attention_threshold)
        )
        snap.intervention_threshold = float(
            cfg.get("intervention_threshold", snap.intervention_threshold)
        )
    except (TypeError, ValueError):
        pass

    for r in rows:
        phrase = (r.get("phrase") or "").strip()
        cat = (r.get("category") or "").strip().lower()
        if not phrase:
            continue
        if cat == "crisis_zh":
            snap.crisis_phrases_zh.append(phrase)
        elif cat == "crisis_en":
            snap.crisis_phrases_en.append(phrase.lower())
        elif cat == "crisis_regex":
            snap.crisis_regex_patterns.append(phrase)
        elif cat == "negative":
            snap.words_negative.append(phrase)
        elif cat == "metaphor":
            snap.words_metaphor.append(phrase)
        elif cat == "homophone":
            snap.words_homophone.append(phrase)
        else:
            logger.warning("未知 risk_lexicon.category=%s，已跳过", cat)

    logger.info(
        "已从数据库加载风险词表: crisis_zh=%s crisis_regex=%s negative=%s",
        len(snap.crisis_phrases_zh),
        len(snap.crisis_regex_patterns),
        len(snap.words_negative),
    )
    return snap


def load_risk_lexicon_snapshot(*, force_reload: bool = False) -> Optional[RiskLexiconSnapshot]:
    """
    若库中存在启用的 risk_lexicon 行，则返回快照；否则返回 None（调用方使用代码内置默认）。
    """
    global _SNAPSHOT_CACHE
    if _SNAPSHOT_CACHE is not None and not force_reload:
        return _SNAPSHOT_CACHE
    snap = _fetch_snapshot_from_db()
    if snap is not None:
        _SNAPSHOT_CACHE = snap
    return snap


def reset_pipelines_after_lexicon_change() -> None:
    """清空缓存并让聊天/情感管线下次重建（加载新词表）。"""
    clear_risk_lexicon_cache()
    try:
        from ..routes import chat as chat_mod
        from ..routes import emotion as emotion_mod

        chat_mod.reset_chat_pipeline()
        emotion_mod.reset_emotion_pipeline()
    except Exception as e:
        logger.warning("重置 ML 管线失败，请重启进程: %s", e)
