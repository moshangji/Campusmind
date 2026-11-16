"""
翻译工具模块

当前实现：百度翻译 API（国内稳定、中英效果好），用于「中文 → 英文 → GoEmotions」情绪识别。

使用前准备（在服务器上执行一次即可）：
1. 注册百度翻译开放平台并创建应用，获取 APP ID 和密钥：
   - 打开：https://fanyi-api.baidu.com/
   - 登录/注册 → 管理控制台 → 创建应用（选择「通用翻译」）
   - 记下「APPID」和「密钥」（Secret Key）

2. 在服务器上设置环境变量：
   - 临时（当前会话有效）：
       export BAIDU_TRANSLATE_APPID="你的_appid"
       export BAIDU_TRANSLATE_SECRET="你的_密钥"
   - 或写入 ~/.bashrc 后 source ~/.bashrc

3. 安装依赖：
   - pip install requests
"""

import hashlib
import os
import logging
import random
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BAIDU_API_URL = "https://fanyi-api.baidu.com/api/trans/vip/translate"


class TranslationError(Exception):
    """翻译异常"""


def _baidu_translate(
    text: str,
    appid: str,
    secret: str,
    from_lang: str = "zh",
    to_lang: str = "en",
) -> str:
    """调用百度翻译 API，返回翻译结果。"""
    salt = str(random.randint(32768, 65536))
    sign_str = appid + text + salt + secret
    sign = hashlib.md5(sign_str.encode("utf-8")).hexdigest()

    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": appid,
        "salt": salt,
        "sign": sign,
    }

    try:
        resp = requests.get(BAIDU_API_URL, params=params, timeout=10)
    except Exception as e:
        raise TranslationError(f"请求百度翻译失败：{e}") from e

    if resp.status_code != 200:
        raise TranslationError(
            f"百度翻译返回错误状态码 {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except Exception as e:
        raise TranslationError(f"解析百度翻译返回失败：{e}; 原始: {resp.text[:200]}") from e

    if "error_code" in data:
        raise TranslationError(
            f"百度翻译 API 错误 {data.get('error_code')}: {data.get('error_msg', '')}"
        )

    trans_result = data.get("trans_result", [])
    if not trans_result:
        raise TranslationError(f"百度翻译返回结果为空：{data}")

    # 多段会返回多条，拼成一句
    return " ".join(item.get("dst", "") for item in trans_result).strip()


def translate_zh_to_en(
    text: str,
    appid: Optional[str] = None,
    secret: Optional[str] = None,
) -> str:
    """
    将中文翻译为英文（用于 GoEmotions 英文模型输入）。

    Args:
        text: 中文输入
        appid: 可选，百度翻译 APP ID；不传则从环境变量 BAIDU_TRANSLATE_APPID 读取
        secret: 可选，百度翻译密钥；不传则从环境变量 BAIDU_TRANSLATE_SECRET 读取

    Returns:
        英文翻译结果字符串

    Raises:
        TranslationError: 当 APPID/密钥缺失或调用失败时抛出
    """
    text = (text or "").strip()
    if not text:
        return ""

    try:
        from configs import get
        aid = appid or get("BAIDU_TRANSLATE_APPID")
        sec = secret or get("BAIDU_TRANSLATE_SECRET")
    except ImportError:
        aid = appid or os.environ.get("BAIDU_TRANSLATE_APPID")
        sec = secret or os.environ.get("BAIDU_TRANSLATE_SECRET")

    if not aid or not sec:
        raise TranslationError(
            "未检测到百度翻译配置。请设置环境变量：\n"
            "  export BAIDU_TRANSLATE_APPID=\"你的_appid\"\n"
            "  export BAIDU_TRANSLATE_SECRET=\"你的_密钥\""
        )

    return _baidu_translate(text, appid=aid, secret=sec, from_lang="zh", to_lang="en")


__all__ = ["translate_zh_to_en", "TranslationError"]
