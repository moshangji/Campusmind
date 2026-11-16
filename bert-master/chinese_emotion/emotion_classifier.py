"""
中文情感分析 - 情感分类器

基于 chinese-roberta-wwm-ext 的中文情感分类，支持加载/预测/训练接口。
"""

import os
import logging
from typing import List, Dict, Optional, Union
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)

from .data_processor import CHINESE_EMOTION_LABELS, EMOTION_TO_ID, ID_TO_EMOTION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认中文预训练模型（哈工大）
DEFAULT_CHINESE_MODEL = "hfl/chinese-roberta-wwm-ext"


class ChineseEmotionClassifier:
    """中文情感分类器"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_labels: int = len(CHINESE_EMOTION_LABELS),
        device: Optional[str] = None,
    ):
        """
        model_path: 本地目录或 HuggingFace 模型名。为 None 时使用 DEFAULT_CHINESE_MODEL。
        num_labels: 分类数，默认与 CHINESE_EMOTION_LABELS 一致。
        """
        self.model_path = model_path or DEFAULT_CHINESE_MODEL
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"ChineseEmotionClassifier loaded from {self.model_path}, device={self.device}")

    def _load_tokenizer(self):
        if self.model_path and os.path.isdir(self.model_path):
            try:
                return AutoTokenizer.from_pretrained(self.model_path)
            except Exception:
                pass
            vocab_path = os.path.join(self.model_path, "vocab.txt")
            if os.path.isfile(vocab_path):
                return BertTokenizer(vocab_file=vocab_path)
        return AutoTokenizer.from_pretrained(self.model_path)

    def _load_model(self):
        if self.model_path and os.path.isdir(self.model_path):
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=self.num_labels,
            )
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=self.num_labels,
        )

    def predict(
        self,
        text: str,
        top_k: int = 3,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        预测单条文本的情感，返回 top_k 个标签及概率。
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probs[0], min(top_k, self.num_labels))
        results = []
        for p, idx in zip(top_probs, top_indices):
            idx = idx.item()
            label = ID_TO_EMOTION.get(idx, CHINESE_EMOTION_LABELS[min(idx, len(CHINESE_EMOTION_LABELS) - 1)])
            results.append({"label": label, "probability": p.item()})
        return results

    def get_labels(self) -> List[str]:
        return list(CHINESE_EMOTION_LABELS)
