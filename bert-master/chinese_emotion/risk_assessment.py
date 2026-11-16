"""
中文情感分析 - 心理风险评估

根据情感标签与敏感词输出关注级/干预级/危机级。
"""

import logging
from enum import Enum
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """风险等级（与申报书一致）"""
    NORMAL = "normal"       # 正常
    ATTENTION = "attention" # 关注级
    INTERVENTION = "intervention"  # 干预级
    CRISIS = "crisis"       # 危机级


# 情感标签到风险权重（负面越高越危险）
EMOTION_RISK_WEIGHT = {
    "快乐": 0, "感激": 0, "放松": 0, "期待": 0, "爱": 0, "乐观": 0, "自信": 0, "平静": 0,
    "焦虑": 1, "困惑": 1, "烦躁": 1, "迷茫": 1, "厌烦": 1, "冷漠": 1,
    "悲伤": 2, "失望": 2, "孤独": 2, "无助": 2, "自卑": 2,
    "愤怒": 2, "恐惧": 2, "愧疚": 2, "压力": 2,
    "中性": 0,
}

# GoEmotions 英文标签（与 test_goemotions_model.EMOTIONS 一致），避免中文模型未启用时情感分恒为 0
EMOTION_RISK_WEIGHT_EN = {
    "admiration": 0, "amusement": 0, "approval": 0, "caring": 0, "excitement": 0,
    "gratitude": 0, "joy": 0, "love": 0, "optimism": 0, "pride": 0, "relief": 0,
    "surprise": 0, "neutral": 0, "curiosity": 0, "realization": 0,
    "confusion": 1, "annoyance": 1, "disapproval": 1, "nervousness": 1,
    "embarrassment": 1, "desire": 1,
    "anger": 2, "disappointment": 2, "disgust": 2, "fear": 2, "grief": 2,
    "remorse": 2, "sadness": 2,
}


def _emotion_risk_weight(label: str) -> int:
    if not label:
        return 0
    if label in EMOTION_RISK_WEIGHT:
        return EMOTION_RISK_WEIGHT[label]
    return EMOTION_RISK_WEIGHT_EN.get(label, 0)


class RiskAssessor:
    """心理风险评估器：情感 + 敏感词 → 风险等级"""

    def __init__(self, attention_threshold: float = 0.35, intervention_threshold: float = 0.52):
        """
        attention_threshold: 综合风险分超过此值为关注级
        intervention_threshold: 超过此值为干预级；若检测到危机敏感词则直接危机级
        """
        self.attention_threshold = attention_threshold
        self.intervention_threshold = intervention_threshold

    def assess(
        self,
        emotion_results: List[Dict],
        sensitive_detected: Optional[Dict[str, List]] = None,
        has_crisis_word: bool = False,
    ) -> RiskLevel:
        """
        根据情感预测结果与敏感词检测结果评估风险等级。

        emotion_results: predict() 返回的 top_k 列表，每项 {"label": ..., "probability": ...}
        sensitive_detected: SensitiveWordDetector.detect() 的返回值，可为 None
        has_crisis_word: 是否包含危机类敏感词（如自伤、自杀）
        """
        if has_crisis_word:
            return RiskLevel.CRISIS

        # 用情感加权求“负面程度”分数（0~1）
        score = 0.0
        for item in emotion_results:
            label = item.get("label") or item.get("emotion") or "中性"
            prob = float(item.get("probability", 0.0))
            w = _emotion_risk_weight(label)
            score += w * prob
        # 归一化到约 0~1（权重最大为 2）
        score = min(1.0, score / 2.0)

        if sensitive_detected:
            n_neg = len(sensitive_detected.get("negative", []))
            n_meta = len(sensitive_detected.get("metaphor", []))
            n_hom = len(sensitive_detected.get("homophone", []))
            if n_neg > 0:
                score = min(1.0, score + 0.28)
            if n_neg >= 2:
                score = min(1.0, score + 0.12)
            if n_meta > 0:
                score = min(1.0, score + 0.1)
            if n_hom > 0:
                score = min(1.0, score + 0.08)

        if score >= self.intervention_threshold:
            return RiskLevel.INTERVENTION
        if score >= self.attention_threshold:
            return RiskLevel.ATTENTION
        return RiskLevel.NORMAL
