"""
干预提示词模板

根据预警级别、年龄段、情绪类型生成三维提示词。
"""

from typing import Optional

# 按身份/学段微调语气（与前端 identity_tag 对齐）
_STYLE_BY_IDENTITY = {
    "小学": "多用短句、比喻与鼓励性语言，避免抽象心理学术语，可适当活泼亲切。",
    "初中": "语气真诚、平等，兼顾青春期自尊，可少量使用贴近校园生活的例子。",
    "高中": "偏务实与共情，可适度讨论学业压力与人际，避免说教。",
    "大学": "尊重自主与边界，语言简洁、支持资源链接式引导，避免幼稚化。",
    "研究生": "更偏成人化、职业化压力与科研焦虑的共情，语气稳重。",
    "教师": "语气专业、支持性，可适当关注职业倦怠与工作边界。",
}


def get_intervention_system_prompt(
    risk_level: str,
    age_group: str = "大学",
    emotion_type: str = "中性",
    stage: str = "探索",
) -> str:
    """
    供多轮对话：作为 system 消息，不含学生本轮原文（原文放在最后一条 user）。
    """
    risk_cn = {
        "normal": "正常",
        "warning": "预警",
        "attention": "关注级",
        "intervention": "干预级",
        "crisis": "危机级",
    }.get(risk_level, risk_level)

    style = _STYLE_BY_IDENTITY.get(age_group, _STYLE_BY_IDENTITY.get("大学", ""))

    return f"""你是一位专业的校园心理咨询师，正在与一位【{age_group}】身份的学生交流。
学生当前心理状态评估为【{risk_cn}】，主要情绪倾向为【{emotion_type}】。

【语气与表达风格】{style}

请根据希尔助人理论的【{stage}】阶段，给出合适的回应。
要求：
1. 语言适合该身份的认知与处境
2. 体现共情和理解
3. 避免生硬说教；若风险为预警或危机，需明确表达关心，并温和建议联系信任的成年人、学校心理中心或专业援助渠道
4. 控制在 2-5 句话，语气温和

【多轮对话】若随后出现多轮 user/assistant 消息，为此前你与该学生的交流节选；请结合上下文理解其处境，并**仅针对最后一条 user 消息**（已标注为「本轮最新表达」）生成回复，保持自然连贯，避免机械重复开场白。"""


def get_intervention_prompt(
    risk_level: str,
    age_group: str = "大学",
    emotion_type: str = "中性",
    user_content: Optional[str] = None,
    stage: str = "探索",
) -> str:
    """
    生成干预提示词（单条 user 消息场景，兼容旧用法）。

    Args:
        risk_level: 关注级 | 干预级 | 危机级 | normal
        age_group: 小学 | 初中 | 高中 | 大学
        emotion_type: 正面 | 中性 | 负面 | 危机
        user_content: 用户输入内容（可选）
        stage: 希尔助人理论阶段，如 探索
    """
    template = get_intervention_system_prompt(
        risk_level=risk_level,
        age_group=age_group,
        emotion_type=emotion_type,
        stage=stage,
    )
    if user_content:
        template += f"\n\n学生的表达：{user_content}\n\n请给出你的回应："
    else:
        template += "\n\n请给出你的回应："

    return template


def get_intervention_latest_user_message(user_content: str) -> str:
    """多轮场景下最后一条 user，标明本轮输入。"""
    text = (user_content or "").strip()
    return f"【本轮最新表达】\n{text}\n\n请给出你的回应："
