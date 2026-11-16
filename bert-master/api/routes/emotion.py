"""情感分析接口"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from ..models.schemas import EmotionRequest, EmotionResponse
from ..utils.risk_lexicon import load_risk_lexicon_snapshot

router = APIRouter(prefix="/emotion", tags=["emotion"])

_emotion_pipeline = None


def reset_emotion_pipeline() -> None:
    global _emotion_pipeline
    _emotion_pipeline = None


def _use_chinese_emotion():
    """是否启用中文情感模型（默认否）"""
    v = os.environ.get("USE_CHINESE_EMOTION", "0").strip().lower()
    return v in ("1", "true", "yes")


def _find_chinese_model_path():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for path in [
        os.path.join(base, "output", "chinese_emotion", "final_model"),
        os.path.join(base, "..", "output", "chinese_emotion", "final_model"),
        "/root/autodl-tmp/Campusmind/output/chinese_emotion/final_model",
    ]:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and os.path.exists(os.path.join(abs_path, "config.json")):
            return abs_path
    return None


def _get_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        from chinese_emotion import SensitiveWordDetector, RiskAssessor

        snap = load_risk_lexicon_snapshot()
        detector = (
            SensitiveWordDetector(snapshot=snap)
            if snap is not None
            else SensitiveWordDetector()
        )
        assessor = (
            RiskAssessor(
                attention_threshold=snap.attention_threshold,
                intervention_threshold=snap.intervention_threshold,
            )
            if snap is not None
            else RiskAssessor()
        )

        chinese_path = _find_chinese_model_path() if _use_chinese_emotion() else None
        if chinese_path:
            from chinese_emotion import ChineseEmotionClassifier
            _emotion_pipeline = {
                "use_chinese": True,
                "classifier": ChineseEmotionClassifier(model_path=chinese_path),
                "detector": detector,
                "assessor": assessor,
            }
        else:
            from test_goemotions_model import find_model_path, load_model
            from translation import translate_zh_to_en
            path = find_model_path()
            if not path:
                raise HTTPException(status_code=503, detail="未找到情感模型")
            model, tokenizer = load_model(path)
            _emotion_pipeline = {
                "use_chinese": False,
                "model": model,
                "tokenizer": tokenizer,
                "translator": translate_zh_to_en,
                "detector": detector,
                "assessor": assessor,
            }
    return _emotion_pipeline


@router.post("/analyze", response_model=EmotionResponse)
def analyze_emotion(req: EmotionRequest):
    """情感分析（默认翻译+GoEmotions；USE_CHINESE_EMOTION=1 时用中文模型）→ 风险"""
    p = _get_pipeline()
    if p["use_chinese"]:
        top = p["classifier"].predict(req.text, top_k=5)
        en_text = None
    else:
        try:
            en_text = p["translator"](req.text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"翻译失败: {e}")
        from test_goemotions_model import predict_emotion
        device = next(p["model"].parameters()).device
        results = predict_emotion(p["model"], p["tokenizer"], en_text, top_k=5, device=device)
        top = [{"label": r["emotion"], "probability": r["probability"]} for r in results]
    sensitive = p["detector"].detect(req.text)
    risk = p["assessor"].assess(top, sensitive_detected=sensitive, has_crisis_word=p["detector"].has_risk(req.text))
    return EmotionResponse(top_emotions=top, risk_level=risk.value, translated=en_text)
