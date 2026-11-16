"""
多模态 - 行为数据编码器

将考勤、图书馆借阅、食堂消费等行为数据编码为固定维度向量。
"""

import logging
import numpy as np
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 行为特征维度（可扩展）
BEHAVIOR_DIM = 16


class BehaviorEncoder:
    """
    行为数据编码器。
    输入：{"attendance": 0.9, "library": 5, "canteen": 12, ...}
    输出：固定维度向量。
    """

    def __init__(self, output_dim: int = BEHAVIOR_DIM):
        self.output_dim = output_dim

    def encode(self, behavior: Dict) -> np.ndarray:
        """
        将行为字典编码为向量。

        Args:
            behavior: 例如
                - attendance: 出勤率 0~1
                - library_count: 月借阅次数
                - canteen_count: 月食堂消费次数
                - scale_score: 量表得分 0~100（可选）
                - late_count: 迟到次数（可选）
        Returns:
            output_dim 维向量
        """
        features = []

        # 出勤率 0~1
        att = float(behavior.get("attendance", 0.5))
        features.extend([att, 1 - att])

        # 借阅次数（归一化，假设 0~20 为正常）
        lib = float(behavior.get("library_count", 0))
        lib_norm = min(1.0, lib / 20.0)
        features.extend([lib_norm, 1 - lib_norm])

        # 食堂消费次数（归一化，假设 0~60 为正常）
        can = float(behavior.get("canteen_count", 0))
        can_norm = min(1.0, can / 60.0)
        features.extend([can_norm, 1 - can_norm])

        # 量表得分 0~100 → 0~1
        scale = float(behavior.get("scale_score", 50))
        scale_norm = scale / 100.0
        features.extend([scale_norm, 1 - scale_norm])

        # 迟到次数（归一化）
        late = float(behavior.get("late_count", 0))
        late_norm = min(1.0, late / 10.0)
        features.extend([late_norm, 1 - late_norm])

        # 补足到 output_dim
        while len(features) < self.output_dim:
            features.append(0.0)
        features = features[:self.output_dim]

        return np.array(features, dtype=np.float32)
