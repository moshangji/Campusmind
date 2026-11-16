"""
多模态融合模块

融合文本情感、行为数据、量表数据，支持豆包/DeepSeek 等大模型增强理解。
"""

from .text_encoder import TextEncoder
from .behavior_encoder import BehaviorEncoder
from .fusion_model import MultimodalFusion
from .llm_enhancer import LLMEnhancer
from .attention_fusion import attention_fusion
from .vision_client import analyze_image

__all__ = [
    "TextEncoder",
    "BehaviorEncoder",
    "MultimodalFusion",
    "LLMEnhancer",
    "attention_fusion",
    "analyze_image",
]
