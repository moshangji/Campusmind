"""API 请求/响应模型"""

from typing import List, Dict, Optional
from pydantic import BaseModel


class EmotionRequest(BaseModel):
    text: str


class EmotionResponse(BaseModel):
    top_emotions: List[Dict]
    risk_level: str
    translated: Optional[str] = None


class RiskRequest(BaseModel):
    text: str


class RiskResponse(BaseModel):
    risk_level: str
    top_emotions: List[Dict]
    sensitive_detected: Optional[Dict] = None


class ChatRequest(BaseModel):
    message: str
    age_group: Optional[str] = "大学"
    token: Optional[str] = None
    mood_state: Optional[str] = None


class ChatResponse(BaseModel):
    emotion: List[Dict]
    risk_level: str
    intervention: str
