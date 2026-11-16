"""
GoEmotions 情绪识别 - 测试与使用

用法：
  1. 直接运行：会跑内置示例 + 交互输入
     python test_goemotions_model.py

  2. 指定模型路径：
     python test_goemotions_model.py --model_path /path/to/final_model

  3. 只跑示例、不交互：
     python test_goemotions_model.py --no_interactive
"""

import os
import sys
import argparse
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# 27种情感类别（与训练时一致）
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# 情感中文对照（便于理解）
EMOTIONS_CN = {
    "admiration": "钦佩", "amusement": "娱乐", "anger": "愤怒", "annoyance": "恼怒",
    "approval": "赞同", "caring": "关心", "confusion": "困惑", "curiosity": "好奇",
    "desire": "渴望", "disappointment": "失望", "disapproval": "反对",
    "disgust": "厌恶", "embarrassment": "尴尬", "excitement": "兴奋",
    "fear": "恐惧", "gratitude": "感激", "grief": "悲伤", "joy": "快乐",
    "love": "爱", "nervousness": "紧张", "optimism": "乐观", "pride": "骄傲",
    "realization": "领悟", "relief": "宽慰", "remorse": "懊悔", "sadness": "难过",
    "surprise": "惊讶", "neutral": "中性"
}


def find_model_path():
    """自动查找模型路径（兼容 bert-master/output 与 Campusmind/output，优先当前目录下 output）"""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # bert-master 下 output（当前目录训练产出优先）
        os.path.join(base, "output", "goemotions_hf", "final_model"),
        os.path.join(base, "output", "federated", "final_model"),
        # 上一级 Campusmind/output
        os.path.join(base, "..", "output", "goemotions_hf", "final_model"),
        os.path.join(base, "..", "output", "federated", "final_model"),
        # 本地 Windows 绝对路径
        r"D:\Campusmind\output\goemotions_hf\final_model",
        r"D:\Campusmind\output\federated\final_model",
        # 服务器 AutoDL 绝对路径（两处都查）
        "/root/autodl-tmp/Campusmind/bert-master/output/goemotions_hf/final_model",
        "/root/autodl-tmp/Campusmind/bert-master/output/federated/final_model",
        "/root/autodl-tmp/Campusmind/output/goemotions_hf/final_model",
        "/root/autodl-tmp/Campusmind/output/federated/final_model",
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and os.path.exists(os.path.join(abs_path, "config.json")):
            return abs_path
    return None


def _load_tokenizer_local(model_path):
    """从本地加载 tokenizer，不访问 HuggingFace 网络（与 federated/utils 逻辑一致）"""
    path = os.path.abspath(model_path)
    vocab_in_model = os.path.join(path, "vocab.txt")
    # 1. 模型目录：先 from_pretrained，再目录内 vocab.txt
    if os.path.isdir(path):
        try:
            return BertTokenizer.from_pretrained(path)
        except Exception:
            pass
        if os.path.isfile(vocab_in_model):
            return BertTokenizer(vocab_file=vocab_in_model)
    # 2. 项目内 BERT 预训练目录
    for base in [
        os.path.dirname(path),
        os.path.dirname(os.path.dirname(path)),
        os.path.dirname(os.path.dirname(os.path.dirname(path))),
        "/root/autodl-tmp/Campusmind",
        os.path.expanduser("~/autodl-tmp/Campusmind"),
    ]:
        if not base:
            continue
        for rel in ["bert_model/cased_L-12_H-768_A-12/vocab.txt", "bert_model/vocab.txt"]:
            vocab_path = os.path.join(base, rel)
            if os.path.isfile(vocab_path):
                return BertTokenizer(vocab_file=vocab_path)
    return None


def load_model(model_path):
    """加载训练好的模型和分词器"""
    print(f"正在加载模型: {model_path}")
    if not os.path.isdir(model_path):
        print(f"错误: 路径不存在或不是目录: {model_path}")
        sys.exit(1)
    model = BertForSequenceClassification.from_pretrained(model_path)
    # 分词器：优先本地（不访问网络），失败再尝试 bert-base-cased（可能需网络）
    tokenizer = _load_tokenizer_local(model_path)
    if tokenizer is None:
        try:
            tokenizer = BertTokenizer.from_pretrained(model_path)
        except Exception:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("模型加载完成。")
    return model, tokenizer


def predict_emotion(model, tokenizer, text, top_k=3, device=None):
    """
    预测单条文本的情感（返回 top_k 个情感及概率）
    """
    if device is None:
        device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs[0], top_k)
    results = []
    for p, idx in zip(top_probs, top_indices):
        emotion = EMOTIONS[idx.item()]
        results.append({
            "emotion": emotion,
            "emotion_cn": EMOTIONS_CN.get(emotion, emotion),
            "probability": p.item(),
        })
    return results


def print_result(text, results):
    """打印一条预测结果"""
    print(f"\n输入: {text}")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['emotion']:15s} ({r['emotion_cn']:4s})  概率: {r['probability']:.2%}")
    # 主情感
    main_emotion = results[0]["emotion"]
    main_cn = results[0]["emotion_cn"]
    print(f"  => 主情感: {main_emotion} ({main_cn})")


def run_examples(model, tokenizer, device):
    """运行内置示例"""
    examples = [
        "I'm so happy today! This is amazing!",
        "This is terrible. I hate it.",
        "I feel confused about what to do.",
        "Thank you so much for your help!",
        "I'm really nervous about the exam tomorrow.",
        "I'm proud of my achievements.",
    ]
    print("\n" + "=" * 60)
    print("示例：GoEmotions 情绪识别")
    print("=" * 60)
    for text in examples:
        results = predict_emotion(model, tokenizer, text, top_k=3, device=device)
        print_result(text, results)


def run_interactive(model, tokenizer, device):
    """交互式：输入英文句子，输出识别的情感"""
    print("\n" + "=" * 60)
    print("交互模式：输入英文句子，识别情绪（输入 q 或 空行 退出）")
    print("=" * 60)
    while True:
        try:
            text = input("\n请输入英文句子: ").strip()
        except EOFError:
            break
        if not text or text.lower() == "q":
            print("退出。")
            break
        results = predict_emotion(model, tokenizer, text, top_k=3, device=device)
        print_result(text, results)


def main():
    parser = argparse.ArgumentParser(description="GoEmotions 情绪识别测试")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="训练好的模型目录路径（不填则自动查找）",
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
        help="直接预测这一句，不跑示例、不交互。例如: --text \"I am so happy\"",
    )
    args = parser.parse_args()

    model_path = args.model_path or find_model_path()
    if not model_path:
        print("未找到模型。请先训练模型，或使用 --model_path 指定目录。")
        print("示例:")
        print("  服务器: python test_goemotions_model.py --model_path /root/autodl-tmp/Campusmind/output/goemotions_hf/final_model")
        print("  本地:   python test_goemotions_model.py --model_path D:\\Campusmind\\output\\goemotions_hf\\final_model")
        sys.exit(1)

    model, tokenizer = load_model(model_path)
    device = next(model.parameters()).device

    if args.text:
        results = predict_emotion(model, tokenizer, args.text, top_k=3, device=device)
        print_result(args.text, results)
        return

    run_examples(model, tokenizer, device)
    if not args.no_interactive:
        run_interactive(model, tokenizer, device)


if __name__ == "__main__":
    main()
