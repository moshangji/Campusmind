"""
中文情感模型测试与使用

用法：
  1. 训练完成后直接运行（自动查找 output/chinese_emotion/final_model）：
     python test_chinese_emotion_model.py

  2. 指定模型路径：
     python test_chinese_emotion_model.py --model_path ./output/chinese_emotion/final_model

  3. 只跑示例、不交互：
     python test_chinese_emotion_model.py --no_interactive
"""

import os
import sys
import argparse

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chinese_emotion import (
    ChineseEmotionClassifier,
    SensitiveWordDetector,
    RiskAssessor,
    RiskLevel,
)


def find_model_path():
    """自动查找中文情感模型路径"""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "output", "chinese_emotion", "final_model"),
        os.path.join(base, "..", "output", "chinese_emotion", "final_model"),
        "/root/autodl-tmp/Campusmind/output/chinese_emotion/final_model",
        "/root/autodl-tmp/Campusmind/bert-master/output/chinese_emotion/final_model",
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and os.path.exists(os.path.join(abs_path, "config.json")):
            return abs_path
    return None


def run_examples(classifier, detector, assessor):
    """运行内置示例"""
    examples = [
        "今天心情很好，考试通过了。",
        "压力好大，有点绷不住了。",
        "谢谢你的帮助，真的很感激。",
        "最近很焦虑，睡不好觉。",
    ]
    print("\n" + "=" * 60)
    print("示例：中文情感 + 敏感词 + 风险等级")
    print("=" * 60)
    for text in examples:
        _print_one(classifier, detector, assessor, text)


def _print_one(classifier, detector, assessor, text):
    """对单句打印：情感、敏感词、风险等级"""
    print(f"\n输入: {text}")
    print("-" * 50)
    emotion_results = classifier.predict(text, top_k=3)
    for i, r in enumerate(emotion_results, 1):
        print(f"  情感 {i}: {r['label']}  概率: {r['probability']:.2%}")
    sensitive = detector.detect(text)
    has_risk = detector.has_risk(text)
    risk = assessor.assess(emotion_results, sensitive_detected=sensitive, has_crisis_word=has_risk)
    if any(sensitive.values()):
        print(f"  敏感词: {sensitive}")
    print(f"  风险等级: {risk.value}")


def run_interactive(classifier, detector, assessor):
    """交互式输入中文，输出情感与风险"""
    print("\n" + "=" * 60)
    print("交互模式：输入中文句子，识别情感与风险（输入 q 或空行退出）")
    print("=" * 60)
    while True:
        try:
            text = input("\n请输入中文: ").strip()
        except EOFError:
            break
        if not text or text.lower() == "q":
            print("退出。")
            break
        _print_one(classifier, detector, assessor, text)


def main():
    parser = argparse.ArgumentParser(description="中文情感模型测试")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="训练好的中文情感模型目录（不填则自动查找）",
    )
    parser.add_argument(
        "--no_interactive",
        action="store_true",
        help="只跑示例，不进入交互输入",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="直接预测这一句，不跑示例、不交互",
    )
    args = parser.parse_args()

    model_path = args.model_path or find_model_path()
    if not model_path:
        print("未找到中文情感模型。请先训练模型，或使用 --model_path 指定目录。")
        print("示例:")
        print("  python test_chinese_emotion_model.py --model_path ./output/chinese_emotion/final_model")
        sys.exit(1)

    print(f"加载模型: {model_path}")
    classifier = ChineseEmotionClassifier(model_path=model_path)
    detector = SensitiveWordDetector()
    assessor = RiskAssessor()
    print("加载完成。")

    if args.text:
        _print_one(classifier, detector, assessor, args.text)
        return

    run_examples(classifier, detector, assessor)
    if not args.no_interactive:
        run_interactive(classifier, detector, assessor)


if __name__ == "__main__":
    main()
