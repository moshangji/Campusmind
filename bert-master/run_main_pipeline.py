"""
主流程：用户输入中文 → 情感识别 → 风险评估 → 大模型干预回复

用法：
  python run_main_pipeline.py

配置：将 .env.example 复制为 .env，填入 API Key，项目启动时自动加载。
"""

import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 最先加载配置（从 .env 或 config.json）
from configs import load_config
load_config()


def _fix_stdin_encoding():
    """修复终端中文输入时的 UnicodeDecodeError（服务器 SSH 环境）"""
    try:
        if hasattr(sys.stdin, "buffer"):
            sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from chinese_emotion import SensitiveWordDetector, RiskAssessor
from baichuan import InterventionGenerator


def _use_chinese_emotion():
    """是否启用中文情感模型（默认否，用翻译+GoEmotions）"""
    v = os.environ.get("USE_CHINESE_EMOTION", "0").strip().lower()
    return v in ("1", "true", "yes")


def _find_chinese_model_path():
    """查找已训练的中文情感模型路径"""
    base = os.path.dirname(os.path.abspath(__file__))
    for path in [
        os.path.join(base, "output", "chinese_emotion", "final_model"),
        os.path.join(base, "..", "output", "chinese_emotion", "final_model"),
        "/root/autodl-tmp/Campusmind/output/chinese_emotion/final_model",
    ]:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and os.path.exists(os.path.join(abs_path, "config.json")):
            return abs_path
    return None


def main():
    _fix_stdin_encoding()
    print("加载模型...")
    detector = SensitiveWordDetector()
    assessor = RiskAssessor()
    intervention = InterventionGenerator()

    # 默认使用 翻译+GoEmotions；仅当 USE_CHINESE_EMOTION=1 且存在中文模型时用中文模型
    chinese_model_path = _find_chinese_model_path() if _use_chinese_emotion() else None
    if chinese_model_path:
        from chinese_emotion import ChineseEmotionClassifier
        classifier = ChineseEmotionClassifier(model_path=chinese_model_path)
        use_chinese = True
        print(f"使用中文情感模型: {chinese_model_path}")
    else:
        from translation import translate_zh_to_en, TranslationError
        from test_goemotions_model import find_model_path, load_model, predict_emotion
        go_path = find_model_path()
        if not go_path:
            print("未找到情感模型（请训练中文情感模型或 GoEmotions）。")
            sys.exit(1)
        model, tokenizer = load_model(go_path)
        device = next(model.parameters()).device
        use_chinese = False
        print("使用 翻译+GoEmotions 方案")

    print("\n" + "=" * 60)
    print("百川——校园智心守护系统 | 主流程演示")
    print("输入中文，将依次完成：情感识别 → 风险判断 → AI 干预回复")
    print("输入 q 或空行退出")
    print("=" * 60)

    while True:
        try:
            text = input("\n请输入: ").strip()
        except EOFError:
            break
        if not text or text.lower() == "q":
            print("退出。")
            break

        # 1. 情感
        if use_chinese:
            top = classifier.predict(text, top_k=5)
            print(f"  情感: {top[0]['label']} ({top[0]['probability']:.2%})")
        else:
            try:
                en_text = translate_zh_to_en(text)
            except Exception as e:
                print(f"[翻译失败] {e}")
                continue
            print(f"  翻译: {en_text}")
            results = predict_emotion(model, tokenizer, en_text, top_k=5, device=device)
            top = [{"label": r["emotion"], "probability": r["probability"]} for r in results]
            print(f"  情感: {top[0]['label']} ({top[0]['probability']:.2%})")

        # 2. 风险
        sensitive = detector.detect(text)
        risk = assessor.assess(top, sensitive_detected=sensitive, has_crisis_word=detector.has_risk(text))
        print(f"  风险: {risk.value}")

        # 3. 干预
        reply = intervention.generate(user_content=text, risk_level=risk.value, top_emotions=top)
        print(f"  干预回复: {reply}")


if __name__ == "__main__":
    main()
