"""
多模态 - 文本编码器

封装情感识别（翻译+GoEmotions 或中文情感模型），输出文本情感特征向量。
"""

import os
import sys
import logging
from typing import Dict, List, Optional

import torch
import numpy as np

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoder:
    """
    文本编码器：将输入文本转为情感特征向量。
    内部使用：翻译+GoEmotions 或 中文情感模型（若已训练）。
    """

    def __init__(
        self,
        use_translate_goemotions: bool = True,
        goemotions_model_path: Optional[str] = None,
        chinese_model_path: Optional[str] = None,
    ):
        """
        Args:
            use_translate_goemotions: True 则使用 中文→翻译→GoEmotions；False 则尝试中文模型。
            goemotions_model_path: GoEmotions 模型目录（用于翻译方案）
            chinese_model_path: 中文情感模型目录（若已训练）
        """
        self.use_translate_goemotions = use_translate_goemotions
        self._emotion_model = None
        self._tokenizer = None
        self._translator = None
        self._chinese_classifier = None

        if use_translate_goemotions:
            self._init_goemotions(goemotions_model_path)
        else:
            self._init_chinese(chinese_model_path)

    def _init_goemotions(self, model_path: Optional[str]):
        """初始化 翻译+GoEmotions"""
        try:
            from test_goemotions_model import find_model_path, load_model
            path = model_path or find_model_path()
            if path:
                self._emotion_model, self._tokenizer = load_model(path)
                from translation import translate_zh_to_en
                self._translator = translate_zh_to_en
                logger.info("TextEncoder: 使用 翻译+GoEmotions 方案")
            else:
                raise FileNotFoundError("未找到 GoEmotions 模型")
        except Exception as e:
            logger.warning(f"TextEncoder 初始化 GoEmotions 失败: {e}，将使用占位向量")
            self._emotion_model = None

    def _init_chinese(self, model_path: Optional[str]):
        """初始化中文情感模型"""
        try:
            from chinese_emotion import ChineseEmotionClassifier
            path = model_path or os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "output", "chinese_emotion", "final_model"
            )
            if os.path.isdir(path):
                self._chinese_classifier = ChineseEmotionClassifier(model_path=path)
                logger.info("TextEncoder: 使用中文情感模型")
            else:
                raise FileNotFoundError(f"未找到中文模型: {path}")
        except Exception as e:
            logger.warning(f"TextEncoder 初始化中文模型失败: {e}，回退到翻译+GoEmotions")
            self.use_translate_goemotions = True
            self._init_goemotions(None)

    def encode(self, text: str) -> Dict:
        """
        将文本编码为情感特征。
        Returns:
            {
                "emotion_probs": [...],  # 各情感概率（按固定顺序）
                "top_emotions": [{"label": "...", "probability": ...}, ...],
                "text_feature": np.ndarray,  # 固定维度向量，用于融合
            }
        """
        # 情感标签顺序（GoEmotions 27 类 + neutral）
        emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        n_labels = len(emotion_labels)

        if self._chinese_classifier is not None:
            results = self._chinese_classifier.predict(text, top_k=n_labels)
            probs = [0.0] * n_labels
            label2idx = {r["label"]: i for i, r in enumerate(results)}
            for r in results:
                idx = label2idx.get(r["label"], n_labels - 1)
                if idx < n_labels:
                    probs[idx] = r["probability"]
            top_emotions = self._chinese_classifier.predict(text, top_k=5)
        elif self._emotion_model is not None and self._translator is not None:
            try:
                en_text = self._translator(text)
            except Exception:
                en_text = text
            device = next(self._emotion_model.parameters()).device
            from test_goemotions_model import predict_emotion
            top_emotions = predict_emotion(
                self._emotion_model, self._tokenizer, en_text,
                top_k=n_labels, device=device
            )
            probs = [0.0] * n_labels
            emo_to_idx = {e: i for i, e in enumerate(emotion_labels)}
            for r in top_emotions:
                emo = r.get("emotion", r.get("label", "neutral"))
                idx = emo_to_idx.get(emo, n_labels - 1)
                probs[idx] = r["probability"]
        else:
            probs = [1.0 / n_labels] * n_labels
            top_emotions = [{"label": "neutral", "probability": 1.0 / n_labels}]

        probs_arr = np.array(probs, dtype=np.float32)
        return {
            "emotion_probs": probs,
            "top_emotions": top_emotions[:5],
            "text_feature": probs_arr,
        }
