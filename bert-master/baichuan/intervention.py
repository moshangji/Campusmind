"""
干预方案生成

根据情感、风险、用户内容，调用大模型生成干预回复。
"""

import logging
from typing import Optional, List, Dict

from .api_client import LLMClient
from .prompt_templates import (
    get_intervention_prompt,
    get_intervention_system_prompt,
    get_intervention_latest_user_message,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 正面/负面/危机 情感映射（含中英文）
_POSITIVE = {"joy", "gratitude", "love", "optimism", "relief", "amusement", "approval", "caring",
             "快乐", "感激", "爱", "乐观", "放松", "期待", "自信", "平静"}
_NEGATIVE = {"sadness", "anger", "fear", "anxiety", "disappointment", "grief", "nervousness",
             "悲伤", "愤怒", "恐惧", "焦虑", "失望", "孤独", "无助", "烦躁", "压力", "迷茫", "自卑", "厌烦", "冷漠", "困惑"}
_CRISIS = {"anger", "fear", "grief", "sadness", "愤怒", "恐惧", "悲伤"}


def _emotion_to_type(top_emotions: List[Dict]) -> str:
    """将 top 情感映射为 正面/中性/负面/危机"""
    if not top_emotions:
        return "中性"
    top = top_emotions[0]
    label = (top.get("label") or top.get("emotion", "")).lower().strip()
    prob = top.get("probability", 0)
    if label in _POSITIVE:
        return "正面"
    if label in _CRISIS and prob > 0.5:
        return "危机"
    if label in _NEGATIVE:
        return "负面"
    return "中性"


class InterventionGenerator:
    """干预回复生成器"""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            from configs import get
            provider = provider or get("LLM_PROVIDER", "deepseek")
        except ImportError:
            provider = provider or "deepseek"
        self.client = LLMClient(provider=provider, api_key=api_key)

    def generate(
        self,
        user_content: str,
        risk_level: str,
        top_emotions: List[Dict],
        age_group: str = "大学",
        history: Optional[List[Dict[str, str]]] = None,
        max_history_chars: int = 600,
    ) -> str:
        """
        根据用户内容、风险等级、情感，生成干预回复。

        history: 可选，OpenAI 风格 [{"role":"user"|"assistant","content":"..."}]，
                 仅登录用户由服务端从库中组装；匿名不传。
        """
        emotion_type = _emotion_to_type(top_emotions)
        try:
            if history:
                system_prompt = get_intervention_system_prompt(
                    risk_level=risk_level,
                    age_group=age_group,
                    emotion_type=emotion_type,
                )
                messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
                for m in history:
                    role = (m.get("role") or "").strip().lower()
                    content = (m.get("content") or "").strip()
                    if role not in ("user", "assistant") or not content:
                        continue
                    cap = content[:max_history_chars]
                    if len(content) > max_history_chars:
                        cap = cap + "…"
                    messages.append({"role": role, "content": cap})
                messages.append(
                    {
                        "role": "user",
                        "content": get_intervention_latest_user_message(user_content),
                    }
                )
                max_tokens = 512
                return self.client.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
            prompt = get_intervention_prompt(
                risk_level=risk_level,
                age_group=age_group,
                emotion_type=emotion_type,
                user_content=user_content,
            )
            return self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.7,
            )
        except Exception as e:
            logger.warning(f"干预生成失败: {e}")
            return f"[大模型调用失败，请检查 API Key] {str(e)}"
