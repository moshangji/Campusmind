"""
多模态 - 大模型增强

使用豆包、DeepSeek、百川等大模型 API 对多模态理解进行增强。
统一 OpenAI 兼容接口。
"""

import os
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMEnhancer:
    """
    调用豆包/DeepSeek/百川等大模型，对多模态输入生成增强理解或摘要。
    使用 OpenAI 兼容 API（openai 库）。
    """

    def __init__(
        self,
        provider: str = "deepseek",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            provider: "deepseek" | "doubao" | "baichuan"
            api_key: API Key，不传则从环境变量读取
            base_url: 自定义 base_url（部分厂商需要）
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url or self._get_base_url()

    def _get_api_key(self) -> Optional[str]:
        env_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "doubao": "DOUBAO_API_KEY",
            "baichuan": "BAICHUAN_API_KEY",
        }
        return os.environ.get(env_map.get(self.provider, "OPENAI_API_KEY"))

    def _get_base_url(self) -> Optional[str]:
        urls = {
            "deepseek": "https://api.deepseek.com/v1",
            "doubao": "https://ark.cn-beijing.volces.com/api/v3",  # 豆包火山
            "baichuan": "https://api.baichuan-ai.com/v1",
        }
        return urls.get(self.provider)

    def enhance(
        self,
        text_summary: str,
        behavior_summary: str,
        risk_level: str,
        max_tokens: int = 256,
    ) -> str:
        """
        根据文本情感、行为、风险等级，调用大模型生成增强理解或建议摘要。

        Args:
            text_summary: 文本情感摘要
            behavior_summary: 行为数据摘要
            risk_level: 风险等级
        Returns:
            大模型返回的文本
        """
        if not self.api_key:
            return f"[未配置{self.provider} API Key] 文本摘要: {text_summary}, 行为: {behavior_summary}, 风险: {risk_level}"

        prompt = f"""你是一位校园心理健康辅助助手。请根据以下信息，用一两句话简要概括学生的综合心理状态，并给出一个温和的建议方向（不要写具体干预话术）：

文本情感摘要：{text_summary}
行为数据摘要：{behavior_summary}
当前风险等级：{risk_level}

请用简洁、专业的语言回复。"""

        try:
            return self._call_chat(prompt, max_tokens)
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            return f"[LLM 调用失败] {str(e)}"

    def _call_chat(self, prompt: str, max_tokens: int) -> str:
        """调用 Chat Completions API（OpenAI 兼容）"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self._get_model(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def _get_model(self) -> str:
        models = {
            "deepseek": "deepseek-chat",
            "doubao": "doubao-32k",
            "baichuan": "Baichuan2-Turbo",
        }
        return models.get(self.provider, "gpt-3.5-turbo")
