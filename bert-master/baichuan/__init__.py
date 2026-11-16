"""
大模型干预模块

支持百川、豆包、DeepSeek 等大模型 API，生成心理干预回复。
"""

from .api_client import LLMClient
from .prompt_templates import (
    get_intervention_prompt,
    get_intervention_system_prompt,
    get_intervention_latest_user_message,
)
from .intervention import InterventionGenerator

__all__ = [
    "LLMClient",
    "get_intervention_prompt",
    "get_intervention_system_prompt",
    "get_intervention_latest_user_message",
    "InterventionGenerator",
]
