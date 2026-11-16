"""
多模态 - 融合模型

融合文本情感特征与行为特征，输出统一表示。
支持早期融合（特征拼接）与晚期融合（加权决策）。
"""

import logging
import numpy as np
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFusion:
    """
    多模态融合：文本特征 + 行为特征 → 统一表示
    """

    def __init__(self, text_weight: float = 0.7, behavior_weight: float = 0.3):
        self.text_weight = text_weight
        self.behavior_weight = behavior_weight

    def fuse(
        self,
        text_feature: np.ndarray,
        behavior_feature: np.ndarray,
        strategy: str = "early",
    ) -> Dict:
        """
        融合文本与行为特征。

        Args:
            text_feature: 文本情感向量
            behavior_feature: 行为向量
            strategy: "early" 特征拼接, "late" 加权拼接
        Returns:
            {"fused": np.ndarray, "strategy": str}
        """
        if strategy == "early":
            fused = np.concatenate([text_feature, behavior_feature]).astype(np.float32)
        else:
            # 归一化后加权
            t_norm = text_feature / (np.linalg.norm(text_feature) + 1e-8)
            b_norm = behavior_feature / (np.linalg.norm(behavior_feature) + 1e-8)
            min_len = min(len(t_norm), len(b_norm))
            fused = (
                self.text_weight * t_norm[:min_len] +
                self.behavior_weight * b_norm[:min_len]
            ).astype(np.float32)
        return {"fused": fused, "strategy": strategy}
