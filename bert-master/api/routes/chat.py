"""AI 对话/干预接口"""

import sys
import os

_DEFAULT_CTX_TURNS = 6
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from ..models.schemas import ChatRequest, ChatResponse
from ..utils.conversations import (
    save_conversation,
    get_chat_context_messages,
    get_user_conversation_history_rows,
)
from ..utils.auth import get_current_user, verify_token
from ..utils.rbac import (
    normalize_role,
    is_super_admin,
    is_school_admin,
    PLATFORM_CONVERSATION_SCHOOL_ID,
)
from ..utils.risk_lexicon import load_risk_lexicon_snapshot

router = APIRouter(prefix="/chat", tags=["chat"])

_pipeline = None


def reset_chat_pipeline() -> None:
    global _pipeline
    _pipeline = None


def _use_chinese_emotion():
    """是否启用中文情感模型（默认否）"""
    v = os.environ.get("USE_CHINESE_EMOTION", "0").strip().lower()
    return v in ("1", "true", "yes")


def _find_chinese_model_path():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for path in [
        os.path.join(base, "output", "chinese_emotion", "final_model"),
        os.path.join(base, "..", "output", "chinese_emotion", "final_model"),
        "/root/autodl-tmp/Campusmind/output/chinese_emotion/final_model",
    ]:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and os.path.exists(os.path.join(abs_path, "config.json")):
            return abs_path
    return None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from chinese_emotion import SensitiveWordDetector, RiskAssessor
        from baichuan import InterventionGenerator

        snap = load_risk_lexicon_snapshot()
        detector = (
            SensitiveWordDetector(snapshot=snap)
            if snap is not None
            else SensitiveWordDetector()
        )
        assessor = (
            RiskAssessor(
                attention_threshold=snap.attention_threshold,
                intervention_threshold=snap.intervention_threshold,
            )
            if snap is not None
            else RiskAssessor()
        )

        chinese_path = _find_chinese_model_path() if _use_chinese_emotion() else None
        if chinese_path:
            from chinese_emotion import ChineseEmotionClassifier
            _pipeline = {
                "use_chinese": True,
                "classifier": ChineseEmotionClassifier(model_path=chinese_path),
                "detector": detector,
                "assessor": assessor,
                "intervention": InterventionGenerator(),
            }
        else:
            from test_goemotions_model import find_model_path, load_model, predict_emotion
            from translation import translate_zh_to_en
            path = find_model_path()
            if not path:
                raise HTTPException(status_code=503, detail="未找到情感模型")
            model, tokenizer = load_model(path)
            _pipeline = {
                "use_chinese": False,
                "model": model,
                "tokenizer": tokenizer,
                "translator": translate_zh_to_en,
                "detector": detector,
                "assessor": assessor,
                "intervention": InterventionGenerator(),
            }
    return _pipeline


def _school_id_for_conversation(payload: Optional[Dict[str, Any]]) -> str:
    """与写入 conversations 时一致的 school_id 解析（多租户 + 平台桶）。"""
    if not payload:
        return ""
    role = normalize_role(payload.get("role"))
    token_school = (payload.get("school") or "").strip()
    if is_super_admin(role):
        return PLATFORM_CONVERSATION_SCHOOL_ID
    if is_school_admin(role):
        return token_school or PLATFORM_CONVERSATION_SCHOOL_ID
    return token_school


@router.get("/history")
def chat_history(request: Request, limit: int = 10):
    """
    已登录用户：按 user_id + 学校租户返回近期对话行，供前端展示（与 MySQL 存档一致，避免串号）。

    情感/风险评估仍只在发消息接口对「当前一句」执行；本接口只读库。
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="需要登录")
    uid = user.get("uid")
    if uid is None:
        raise HTTPException(status_code=401, detail="无效令牌")
    try:
        lim = int(limit)
    except (TypeError, ValueError):
        lim = 10
    lim = max(1, min(lim, 20))
    school_id_save = _school_id_for_conversation(user)
    try:
        n = int(os.environ.get("CHAT_CONTEXT_MAX_TURNS", str(_DEFAULT_CTX_TURNS)))
    except ValueError:
        n = _DEFAULT_CTX_TURNS
    n = max(1, min(n, 20))
    items = get_user_conversation_history_rows(int(uid), school_id_save, limit=lim)
    return {
        "items": items,
        "context_max_turns": n,
        "display_limit": lim,
    }


@router.post("/message", response_model=ChatResponse)
def chat_message(req: ChatRequest):
    """完整流程：情感识别 + 风险评估 + 大模型干预回复（默认翻译+GoEmotions）"""
    p = _get_pipeline()
    if p["use_chinese"]:
        top = p["classifier"].predict(req.message, top_k=5)
    else:
        try:
            en_text = p["translator"](req.message)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"翻译失败: {e}")
        from test_goemotions_model import predict_emotion
        device = next(p["model"].parameters()).device
        results = predict_emotion(p["model"], p["tokenizer"], en_text, top_k=5, device=device)
        top = [{"label": r["emotion"], "probability": r["probability"]} for r in results]
    sensitive = p["detector"].detect(req.message)
    risk = p["assessor"].assess(top, sensitive_detected=sensitive, has_crisis_word=p["detector"].has_risk(req.message))
    user_id = None
    school_id_save = ""
    payload = None
    if req.token:
        payload = verify_token(req.token)
        if payload:
            user_id = payload.get("uid")
            school_id_save = _school_id_for_conversation(payload)
    history = None
    if user_id:
        try:
            n = int(os.environ.get("CHAT_CONTEXT_MAX_TURNS", str(_DEFAULT_CTX_TURNS)))
        except ValueError:
            n = _DEFAULT_CTX_TURNS
        history = get_chat_context_messages(user_id, school_id_save, max_turns=max(1, min(n, 20)))
        if not history:
            history = None
    intervention = p["intervention"].generate(
        user_content=req.message,
        risk_level=risk.value,
        top_emotions=top,
        age_group=req.age_group or "大学",
        history=history,
    )
    try:
        top0 = (top or [{}])[0]
        label = top0.get("label") or top0.get("emotion") or ""
        prob = float(top0.get("probability") or 0.0)
        save_conversation(
            identity=req.age_group or "大学",
            user_text=req.message,
            mood_state=req.mood_state or "",
            top_emotion=label,
            probability=prob,
            risk_level=risk.value,
            intervention=intervention,
            user_id=user_id,
            school_id=school_id_save,
        )
    except Exception:
        pass
    return ChatResponse(
        emotion=top,
        risk_level=risk.value,
        intervention=intervention,
    )
