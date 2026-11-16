"""
多模态 - 注意力融合层

对文本特征与行为特征做注意力加权融合（简化版，可用于后续扩展）。
"""

import numpy as np
from typing import Tuple


def attention_fusion(
    text_feat: np.ndarray,
    behavior_feat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化注意力：按特征维度的 L2 范数加权。
    Returns:
        (fused_vector, attention_weights)
    """
    t_norm = np.linalg.norm(text_feat) + 1e-8
    b_norm = np.linalg.norm(behavior_feat) + 1e-8
    w_t = t_norm / (t_norm + b_norm)
    w_b = b_norm / (t_norm + b_norm)
    min_len = min(len(text_feat), len(behavior_feat))
    fused = (w_t * text_feat[:min_len] + w_b * behavior_feat[:min_len]).astype(np.float32)
    return fused, np.array([w_t, w_b])
