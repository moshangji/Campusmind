"""多模态图像情绪分析接口（豆包视觉）"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from multimodal.vision_client import analyze_image

router = APIRouter(prefix="/multimodal", tags=["multimodal"])


class MultimodalResponse(BaseModel):
    analysis: str


@router.post("/analyze", response_model=MultimodalResponse)
async def analyze_multimodal_image(
    image: UploadFile = File(..., description="要分析的图片（jpg/png/webp）"),
    prompt: str = Form("请简要分析这张图片中可能反映的情绪或心理状态，1-2句话。"),
):
    """
    图像情绪/心理状态分析。

    - 接收上传图片 + 文本提示
    - 使用 multimodal.vision_client.analyze_image 调用豆包视觉模型
    - 返回一段中文分析文本
    """
    try:
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传的图片为空")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"读取图片失败: {e}")

    try:
        text = analyze_image(data, prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"多模态分析失败: {e}")

    return MultimodalResponse(analysis=text)

