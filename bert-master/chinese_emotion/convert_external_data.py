"""
将外部中文情感数据集转换为本项目 train.json / dev.json 格式

用法示例：
  # 转换 CSV/TSV（列名为 text, label 或 微博原文,情感 等）
  python -m chinese_emotion.convert_external_data --input your_data.csv --output_dir ./data/chinese_emotion

  # 指定文本列、标签列、标签映射（如 simplifyweibo_4_moods 的 喜悦/愤怒/厌恶/低落）
  python -m chinese_emotion.convert_external_data --input weibo_4moods.csv --text_col review --label_col mood \\
    --label_map 喜悦:快乐 愤怒:愤怒 厌恶:厌烦 低落:悲伤 --dev_ratio 0.1
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 常见外部标签 → 本项目 CHINESE_EMOTION_LABELS 的映射（simplifyweibo_4_moods 等）
DEFAULT_LABEL_MAP_4MOODS: Dict[str, str] = {
    "喜悦": "快乐",
    "喜": "快乐",
    "joy": "快乐",
    "愤怒": "愤怒",
    "怒": "愤怒",
    "anger": "愤怒",
    "厌恶": "厌烦",
    "厌": "厌烦",
    "disgust": "厌烦",
    "低落": "悲伤",
    "哀": "悲伤",
    "sadness": "悲伤",
    "悲伤": "悲伤",
    "正面": "快乐",
    "正面情感": "快乐",
    "负面": "悲伤",
    "负面情感": "悲伤",
    "positive": "快乐",
    "negative": "悲伤",
    "0": "悲伤",
    "1": "快乐",
}


def detect_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    if "\t" in first and "," not in first:
        return "\t"
    if "," in first:
        return ","
    return "\t"


def read_csv_tsv(
    path: str,
    text_col: str,
    label_col: str,
    label_map: Optional[Dict[str, str]] = None,
    skip_header: bool = True,
) -> List[Dict[str, str]]:
    delim = detect_delimiter(path)
    label_map = label_map or {}
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return examples
    header = lines[0].strip().split(delim) if skip_header else None
    if skip_header and header:
        try:
            ti = header.index(text_col)
            li = header.index(label_col)
        except ValueError:
            ti, li = 0, 1
            if len(header) >= 2:
                logger.warning(f"未找到列 {text_col}/{label_col}，使用第 1、2 列")
        start = 1
    else:
        ti, li = 0, 1
        start = 0
    for line in lines[start:]:
        parts = line.strip().split(delim)
        if len(parts) <= max(ti, li):
            continue
        text = parts[ti].strip()
        raw_label = parts[li].strip()
        if not text:
            continue
        label = label_map.get(raw_label, raw_label)
        examples.append({"text": text, "label": label})
    return examples


def split_train_dev(examples: List[Dict], dev_ratio: float = 0.1, seed: int = 42):
    import random
    random.seed(seed)
    indices = list(range(len(examples)))
    random.shuffle(indices)
    n_dev = max(1, int(len(examples) * dev_ratio))
    dev_idx = set(indices[:n_dev])
    train_list = [examples[i] for i in indices if i not in dev_idx]
    dev_list = [examples[i] for i in indices if i in dev_idx]
    return train_list, dev_list


def main():
    parser = argparse.ArgumentParser(description="将外部情感数据转为 train.json / dev.json")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径（CSV 或 TSV）")
    parser.add_argument("--output_dir", "-o", default=None, help="输出目录，默认与输入同目录")
    parser.add_argument("--text_col", default="text", help="文本列名（或第 1 列）")
    parser.add_argument("--label_col", default="label", help="标签列名（或第 2 列）")
    parser.add_argument("--label_map", type=str, default=None, help="标签映射，如 喜悦:快乐 愤怒:愤怒（空格分隔多组）")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--no_header", action="store_true", help="输入文件无表头")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        logger.error(f"文件不存在: {args.input}")
        return
    output_dir = args.output_dir or os.path.dirname(args.input)
    os.makedirs(output_dir, exist_ok=True)

    label_map = dict(DEFAULT_LABEL_MAP_4MOODS)
    if args.label_map:
        for pair in args.label_map.split():
            if ":" in pair:
                k, v = pair.split(":", 1)
                label_map[k.strip()] = v.strip()

    examples = read_csv_tsv(
        args.input,
        text_col=args.text_col,
        label_col=args.label_col,
        label_map=label_map,
        skip_header=not args.no_header,
    )
    if not examples:
        logger.error("未解析到任何样本，请检查列名与分隔符")
        return
    train_list, dev_list = split_train_dev(examples, dev_ratio=args.dev_ratio)
    train_path = os.path.join(output_dir, "train.json")
    dev_path = os.path.join(output_dir, "dev.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_list, f, ensure_ascii=False, indent=2)
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_list, f, ensure_ascii=False, indent=2)
    logger.info(f"已写入 train: {len(train_list)} 条 -> {train_path}")
    logger.info(f"已写入 dev:   {len(dev_list)} 条 -> {dev_path}")


if __name__ == "__main__":
    main()
