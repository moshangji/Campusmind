"""
中文情感分析训练脚本

使用 chinese-roberta-wwm-ext + 中文情感数据训练分类模型。
数据格式：data/chinese_emotion/ 下 train.json、dev.json（或 train.tsv、dev.tsv）。
"""
import os
import sys
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# 兼容项目根目录与子目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chinese_emotion.data_processor import (
    load_chinese_emotion_data,
    ChineseEmotionDataset,
    CHINESE_EMOTION_LABELS,
    EMOTION_TO_ID,
)

# 默认中文预训练模型（哈工大）
DEFAULT_CHINESE_MODEL = "hfl/chinese-roberta-wwm-ext"


def load_tokenizer(pretrained: str):
    """加载 tokenizer，优先本地目录，避免无谓访问 HF。"""
    if pretrained and os.path.isdir(pretrained):
        try:
            return AutoTokenizer.from_pretrained(pretrained)
        except Exception:
            pass
        vocab_path = os.path.join(pretrained, "vocab.txt")
        if os.path.isfile(vocab_path):
            from transformers import BertTokenizer
            return BertTokenizer(vocab_file=vocab_path)
    return AutoTokenizer.from_pretrained(pretrained)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="中文情感分类模型训练")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "chinese_emotion"),
        help="中文情感数据目录（内含 train.json/dev.json 或 train.tsv/dev.tsv）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output", "chinese_emotion"),
        help="模型与日志输出目录",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=DEFAULT_CHINESE_MODEL,
        help="预训练模型路径或 HuggingFace 模型名，如 hfl/chinese-roberta-wwm-ext",
    )
    parser.add_argument("--max_length", type=int, default=128, help="序列最大长度")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="每设备 batch 大小")
    parser.add_argument("--warmup_steps", type=int, default=500, help="warmup 步数")
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "tsv"],
        default="json",
        help="数据格式：json 或 tsv",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.json",
        help="训练集文件名（如 train.json / train.tsv）",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default="dev.json",
        help="验证集文件名（如 dev.json / dev.tsv）",
    )
    args = parser.parse_args()

    num_labels = len(CHINESE_EMOTION_LABELS)
    os.makedirs(args.output_dir, exist_ok=True)

    print("加载 tokenizer...")
    tokenizer = load_tokenizer(args.pretrained)

    print("加载数据...")
    train_examples, dev_examples = load_chinese_emotion_data(
        args.data_dir,
        train_file=args.train_file,
        dev_file=args.dev_file,
        format=args.format,
    )
    if not train_examples:
        print(
            f"未找到训练数据。请在 {args.data_dir} 下放置 {args.train_file}（格式：每行/每条含 text 与 label）。"
        )
        sys.exit(1)

    train_dataset = ChineseEmotionDataset(
        train_examples,
        tokenizer,
        max_length=args.max_length,
        label2id=EMOTION_TO_ID,
    )
    eval_dataset = None
    if dev_examples:
        eval_dataset = ChineseEmotionDataset(
            dev_examples,
            tokenizer,
            max_length=args.max_length,
            label2id=EMOTION_TO_ID,
        )

    print("加载模型...")
    if args.pretrained and os.path.isdir(args.pretrained):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained,
            num_labels=num_labels,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained,
            num_labels=num_labels,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("开始训练...")
    trainer.train()

    if eval_dataset:
        print("评估...")
        results = trainer.evaluate()
        print(f"评估结果: {results}")

    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"模型已保存到: {final_model_dir}")


if __name__ == "__main__":
    main()
