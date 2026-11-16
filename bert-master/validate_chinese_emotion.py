"""
中文情感模型效果验证（在 dev 集上算准确率等）

用法（需在 bert-master 目录下）：
  1. 使用默认路径（output/chinese_emotion/final_model、data/chinese_emotion/dev.json）：
     python validate_chinese_emotion.py

  2. 指定模型与数据：
     python validate_chinese_emotion.py --model_path ./output/chinese_emotion/final_model --data_dir ./data/chinese_emotion

  3. 只验证前 N 条（快速看效果）：
     python validate_chinese_emotion.py --max_samples 200
"""

import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chinese_emotion import ChineseEmotionClassifier


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


def load_dev_examples(data_dir: str, dev_file: str = "dev.json"):
    """加载 dev 集：列表 of {"text": ..., "label": ...}"""
    path = os.path.join(data_dir, dev_file)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return [data]


def main():
    parser = argparse.ArgumentParser(description="中文情感模型效果验证（dev 集）")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="训练好的中文情感模型目录",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "chinese_emotion"),
        help="数据目录（内含 dev.json）",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default="dev.json",
        help="验证集文件名",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多验证条数（不指定则全部）",
    )
    parser.add_argument(
        "--show_errors",
        type=int,
        default=10,
        help="打印前 N 个预测错误样本（0 表示不打印）",
    )
    args = parser.parse_args()

    model_path = args.model_path or find_model_path()
    if not model_path:
        print("未找到中文情感模型。请用 --model_path 指定，或先完成训练。")
        sys.exit(1)

    dev = load_dev_examples(args.data_dir, args.dev_file)
    if not dev:
        print(f"未找到验证集: {os.path.join(args.data_dir, args.dev_file)}")
        sys.exit(1)

    if args.max_samples:
        dev = dev[: args.max_samples]
    print(f"加载模型: {model_path}")
    classifier = ChineseEmotionClassifier(model_path=model_path)
    print(f"验证集样本数: {len(dev)}\n")

    correct = 0
    total = 0
    errors = []
    by_label = defaultdict(lambda: {"tp": 0, "total": 0, "pred_total": 0})

    for i, item in enumerate(dev):
        text = item.get("text", "").strip()
        gold = item.get("label", "").strip()
        if not text:
            continue
        total += 1
        preds = classifier.predict(text, top_k=1)
        pred = preds[0]["label"] if preds else ""
        if pred == gold:
            correct += 1
            by_label[gold]["tp"] += 1
        else:
            errors.append({"text": text[:50], "gold": gold, "pred": pred})
        by_label[gold]["total"] += 1
        by_label[pred]["pred_total"] += 1

    acc = correct / total if total else 0
    print("=" * 60)
    print("验证结果（dev 集）")
    print("=" * 60)
    print(f"  准确率 (accuracy): {correct}/{total} = {acc:.2%}\n")

    if by_label:
        print("按真实标签统计（部分）：")
        for label in sorted(by_label.keys())[:15]:
            st = by_label[label]
            rec = st["tp"] / st["total"] if st["total"] else 0
            print(f"  {label}: 召回 {st['tp']}/{st['total']} = {rec:.2%}")
        if len(by_label) > 15:
            print(f"  ... 共 {len(by_label)} 类")

    if args.show_errors and errors:
        print(f"\n预测错误示例（前 {min(args.show_errors, len(errors))} 条）：")
        for e in errors[: args.show_errors]:
            print(f"  文本: {e['text']}...")
            print(f"    真实: {e['gold']}  预测: {e['pred']}")

    print("\n验证完成。若效果满意，可再确认主流程/API 的调用顺序（优先中文模型）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
