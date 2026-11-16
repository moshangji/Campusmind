"""
多模态 - 视觉客户端（豆包 / 火山方舟）

使用豆包 Responses API 的多模态模型，对图像+文本进行联合理解。
"""

import base64
import os
import logging
from pathlib import Path
from typing import Union, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_doubao_client():
    """
    获取豆包（火山方舟）客户端。

    依赖：
      - 环境变量 ARK_API_KEY（见火山方舟环境变量配置指南）
      - 可选 ARK_BASE_URL（默认 https://ark.cn-beijing.volces.com/api/v3）
    """
    try:
        from configs import load_config
        load_config()
    except ImportError:
        # 若在独立脚本中使用，可不依赖 configs
        pass

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai: pip install openai>=1.0")

    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise ValueError("未配置 ARK_API_KEY，请在 .env 或环境变量中设置（火山方舟控制台获取）。")

    base_url = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_image_content(image_source: Union[str, Path, bytes]) -> Dict[str, Any]:
    """
    构造豆包多模态 input 中的图片部分：
      - 若是以 http/https/data: 开头的字符串：作为 image_url 直接传
      - 若是本地路径：读文件并作为 base64 传 image_base64
      - 若是 bytes：同样编码为 base64
    """
    if isinstance(image_source, (str, Path)):
        s = str(image_source)
        if s.startswith("http://") or s.startswith("https://") or s.startswith("data:"):
            return {"type": "input_image", "image_url": s}

        path = Path(s)
        if not path.exists():
            raise FileNotFoundError(f"图像不存在: {path}")
        with open(path, "rb") as f:
            data = f.read()
    elif isinstance(image_source, bytes):
        data = image_source
    else:
        raise TypeError("image_source 应为 路径(str/Path)、URL 字符串或 bytes")

    b64 = base64.b64encode(data).decode()
    # 豆包官方示例仅给出了 image_url 字段，这里使用 data URL 形式传递本地图片
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}


def _extract_text_from_response(resp: Any) -> str:
    """
    从火山方舟 Responses API 返回结构中尽量提取文本。
    若解析失败，则退回 str(resp)。
    """
    try:
        output = getattr(resp, "output", None)
        # Responses API 一般返回一个列表：[reasoning_item, message_item]
        if isinstance(output, list):
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for part in content:
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text.strip():
                            return text.strip()
                elif isinstance(content, str) and content.strip():
                    return content.strip()
        # 兜底：尝试兼容可能存在的 output.choices 结构
        choices = getattr(output, "choices", None)
        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    for part in content:
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text.strip():
                            return text.strip()
                elif isinstance(content, str) and content.strip():
                    return content.strip()
    except Exception:
        pass
    return str(resp)


def analyze_image(
    image_source: Union[str, Path, bytes],
    prompt: str = "请简要分析这张图片中可能反映的情绪或心理状态，1-2句话。",
    model: str = None,
) -> str:
    """
    使用豆包多模态模型（Responses API）分析图像。

    Args:
        image_source: 图像路径/URL/bytes
        prompt: 文本提示
        model: 模型名，默认从环境变量 DOUBAO_VISION_MODEL
               若未设置，则使用示例模型 doubao-seed-2-0-pro-260215
    """
    content: List[Dict[str, Any]] = [
        _build_image_content(image_source),
        {"type": "input_text", "text": prompt},
    ]

    client = _get_doubao_client()
    model_name = model or os.environ.get("DOUBAO_VISION_MODEL", "doubao-seed-2-0-pro-260215")
    resp = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": content}],
    )
    return _extract_text_from_response(resp)


__all__ = ["analyze_image"]
