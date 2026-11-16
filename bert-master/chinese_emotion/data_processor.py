"""
中文情感分析 - 数据处理器

支持 JSON / TSV 格式，适配校园心理场景的情感标签。
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 中文情感标签（适配校园心理，可与 GoEmotions 对应或简化）
CHINESE_EMOTION_LABELS = [
    "快乐", "感激", "放松", "期待", "爱", "乐观", "自信", "平静",  # 正面
    "焦虑", "悲伤", "愤怒", "失望", "恐惧", "愧疚", "孤独", "无助",  # 负面
    "困惑", "烦躁", "压力", "迷茫", "自卑", "厌烦", "冷漠",  # 负面/中性
    "中性"
]

# 标签到 id
EMOTION_TO_ID = {label: i for i, label in enumerate(CHINESE_EMOTION_LABELS)}
ID_TO_EMOTION = {i: label for i, label in enumerate(CHINESE_EMOTION_LABELS)}


class ChineseEmotionProcessor:
    """中文情感数据处理器"""

    def __init__(self, label_list: Optional[List[str]] = None):
        self.label_list = label_list or CHINESE_EMOTION_LABELS
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}

    def load_json(self, filepath: str) -> List[Dict]:
        """加载 JSON 格式数据，期望每行或整体为 [{"text": "...", "label": "..."}, ...]"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        return [data]

    def load_tsv(self, filepath: str) -> List[Dict]:
        """加载 TSV：text\tlabel"""
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    examples.append({"text": parts[0], "label": parts[1]})
        return examples

    def get_label_id(self, label: str) -> int:
        """标签转 id，未知标签归为中性或最后一类"""
        if label in self.label2id:
            return self.label2id[label]
        return len(self.label_list) - 1  # 默认最后一类

    def get_label(self, idx: int) -> str:
        """id 转标签"""
        return self.id2label.get(idx, self.label_list[-1])


def load_chinese_emotion_data(
    data_dir: str,
    train_file: str = "train.json",
    dev_file: str = "dev.json",
    format: str = "json",
) -> Tuple[List[Dict], List[Dict]]:
    """
    加载中文情感数据。
    format: "json" 或 "tsv"
    返回 (train_examples, dev_examples)，每项为 {"text": ..., "label": ...}
    """
    processor = ChineseEmotionProcessor()
    train_path = os.path.join(data_dir, train_file)
    dev_path = os.path.join(data_dir, dev_file)

    load_fn = processor.load_json if format == "json" else processor.load_tsv
    train_examples = load_fn(train_path) if os.path.exists(train_path) else []
    dev_examples = load_fn(dev_path) if os.path.exists(dev_path) else []

    logger.info(f"Loaded train: {len(train_examples)}, dev: {len(dev_examples)}")
    return train_examples, dev_examples


class ChineseEmotionDataset(Dataset):
    """中文情感 PyTorch Dataset"""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 128,
        label2id: Optional[Dict[str, int]] = None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or EMOTION_TO_ID

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        text = item.get("text", "")
        label_str = item.get("label", "中性")
        label_id = self.label2id.get(label_str, len(self.label2id) - 1)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label_id,
        }
