"""
大模型 API 客户端

支持百川、豆包、DeepSeek 等 OpenAI 兼容接口。
"""

import os
import logging
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 各厂商默认配置
PROVIDER_CONFIG = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-32k",
        "env_key": "DOUBAO_API_KEY",
    },
    "baichuan": {
        "base_url": "https://api.baichuan-ai.com/v1",
        "model": "Baichuan2-Turbo",
        "env_key": "BAICHUAN_API_KEY",
    },
    "openai": {
        "base_url": None,
        "model": "gpt-3.5-turbo",
        "env_key": "OPENAI_API_KEY",
    },
}


class LLMClient:
    """统一大模型客户端，支持多厂商"""

    def __init__(
        self,
        provider: str = "deepseek",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Args:
            provider: deepseek | doubao | baichuan | openai
            api_key: 不传则从环境变量读取
            base_url: 自定义 base_url
            model: 自定义 model 名称
        """
        self.provider = provider.lower()
        cfg = PROVIDER_CONFIG.get(self.provider, PROVIDER_CONFIG["deepseek"])
        try:
            from configs import get
            self.api_key = api_key or get(cfg["env_key"])
        except ImportError:
            self.api_key = api_key or os.environ.get(cfg["env_key"])
        self.base_url = base_url or cfg["base_url"]
        self.model = model or cfg["model"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        调用 Chat Completions API。
        messages: [{"role": "user", "content": "..."}]
        """
        if not self.api_key:
            raise ValueError(
                f"未配置 {self.provider} API Key，请设置环境变量 "
                f"{PROVIDER_CONFIG.get(self.provider, {}).get('env_key', 'API_KEY')}"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
