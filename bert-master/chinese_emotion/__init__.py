"""
中文情感分析模块

用于校园心理场景的中文文本情感识别、敏感词检测与心理风险评估。
"""

from .emotion_classifier import ChineseEmotionClassifier
from .data_processor import (
    ChineseEmotionProcessor,
    load_chinese_emotion_data,
    ChineseEmotionDataset,
    CHINESE_EMOTION_LABELS,
)
from .sensitive_words import SensitiveWordDetector
from .risk_assessment import RiskAssessor, RiskLevel

__all__ = [
    'ChineseEmotionClassifier',
    'ChineseEmotionProcessor',
    'load_chinese_emotion_data',
    'ChineseEmotionDataset',
    'CHINESE_EMOTION_LABELS',
    'SensitiveWordDetector',
    'RiskAssessor',
    'RiskLevel',
]

__version__ = '1.0.0'
