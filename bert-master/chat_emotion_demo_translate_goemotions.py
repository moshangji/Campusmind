"""
聊天情绪识别 Demo（方案一）：中文 → 英文翻译 → GoEmotions 英文模型

配置：将 .env.example 复制为 .env，填入百度翻译 API Key，项目启动时自动加载。

流程：
  用户输入中文句子
    → 调用 DeepL 将中文翻译为英文
    → 使用已训练好的 GoEmotions 模型预测情绪
    → 输出英文情绪 + 对应中文含义

使用前准备（在服务器上）：
1. 确保已经训练并保存好 GoEmotions 模型：
   - 通常路径类似：
       /root/autodl-tmp/Campusmind/output/goemotions_hf/final_model
   - 或使用你自己训练时指定的输出目录

2. 按 translation.py 中说明：
   - 注册 DeepL，获取 API Key
   - 设置环境变量：
       export DEEPL_API_KEY="你的_deepl_api_key"
   - 安装依赖：
       pip install requests

3. 运行本脚本：
   - 自动查找模型路径（若你使用的是推荐路径）：
       python chat_emotion_demo_translate_goemotions.py
   - 或手动指定模型路径：
       python chat_emotion_demo_translate_goemotions.py --model_path /root/autodl-tmp/Campusmind/output/goemotions_hf/final_model
"""

import argparse
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

from translation import translate_zh_to_en, TranslationError
from test_goemotions_model import (
    find_model_path,
    load_model,
    predict_emotion,
    print_result,
)


def run_chat_loop(model, tokenizer, device):
    print("\n" + "=" * 60)
    print("中文聊天情绪识别 Demo（中文 → 英文翻译 → GoEmotions）")
    print("输入中文句子，程序会自动翻译成英文并用英文情感模型识别情绪。")
    print("输入 q 或 空行退出。")
    print("=" * 60)

    while True:
        try:
            zh_text = input("\n请输入中文句子: ").strip()
        except EOFError:
            break

        if not zh_text or zh_text.lower() == "q":
            print("退出。")
            break

        # 1. 翻译中文为英文
        try:
            en_text = translate_zh_to_en(zh_text)
        except TranslationError as e:
            print(f"[翻译失败] {e}")
            continue

        if not en_text:
            print("[翻译结果为空] 请换一句再试。")
            continue

        print(f"\n翻译后的英文: {en_text}")

        # 2. 调用 GoEmotions 模型预测
        results = predict_emotion(model, tokenizer, en_text, top_k=3, device=device)

        # 3. 打印结果（沿用原有英文打印逻辑，只是补充显示原始中文）
        print("\n--- 情绪识别结果（基于英文翻译） ---")
        print_result(en_text, results)
        print(f"\n原始中文: {zh_text}")


def main():
    parser = argparse.ArgumentParser(
        description="中文 → 英文翻译 + GoEmotions 聊天情绪识别 Demo"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="训练好的 GoEmotions 模型目录路径（不填则自动查找）",
    )
    args = parser.parse_args()

    model_path = args.model_path or find_model_path()
    if not model_path:
        print("未找到 GoEmotions 模型。请先训练模型，或使用 --model_path 指定目录。")
        print("示例:")
        print(
            "  服务器: python chat_emotion_demo_translate_goemotions.py "
            "--model_path /root/autodl-tmp/Campusmind/output/goemotions_hf/final_model"
        )
        print(
            "  本地:   python chat_emotion_demo_translate_goemotions.py "
            "--model_path D:\\Campusmind\\output\\goemotions_hf\\final_model"
        )
        sys.exit(1)

    _fix_stdin_encoding()
    model, tokenizer = load_model(model_path)
    device = next(model.parameters()).device

    run_chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()

