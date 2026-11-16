"""风险评估接口（复用 emotion 逻辑）"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter
from ..models.schemas import RiskRequest, RiskResponse
from .emotion import _get_pipeline

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post("/assess", response_model=RiskResponse)
def assess_risk(req: RiskRequest):
    """风险评估（优先中文模型）"""
    p = _get_pipeline()
    if p["use_chinese"]:
        top = p["classifier"].predict(req.text, top_k=5)
    else:
        try:
            en_text = p["translator"](req.text)
        except Exception:
            en_text = req.text
        from test_goemotions_model import predict_emotion
        device = next(p["model"].parameters()).device
        results = predict_emotion(p["model"], p["tokenizer"], en_text, top_k=5, device=device)
        top = [{"label": r["emotion"], "probability": r["probability"]} for r in results]
    sensitive = p["detector"].detect(req.text)
    risk = p["assessor"].assess(top, sensitive_detected=sensitive, has_crisis_word=p["detector"].has_risk(req.text))
    return RiskResponse(risk_level=risk.value, top_emotions=top, sensitive_detected=sensitive)
