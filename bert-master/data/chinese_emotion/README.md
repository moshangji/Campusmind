# 中文情感数据目录

本目录用于存放中文情感分类的训练/验证数据。

**重要**：基座模型 `hfl/chinese-roberta-wwm-ext` **不附带**任何情感数据或情感模型，和 GoEmotions 里的 bert-base-cased 一样，只是预训练权重。要得到「中文情感模型」，必须在本目录准备好 `train.json` / `dev.json` 后，用项目根目录的 `train_chinese_emotion.py` 训练。数据需自行从公开数据集获取并转换（见下方「数据来源建议」与转换脚本）。

## 数据格式

### JSON 格式（推荐）

- **train.json** / **dev.json**：每项为 `{"text": "句子", "label": "情感标签"}`。
- 也可以是数组的 JSON 文件：`[{"text": "...", "label": "..."}, ...]`。

示例 `train.json`：

```json
[
  {"text": "今天心情很好，考试通过了。", "label": "快乐"},
  {"text": "压力好大，不知道该怎么办。", "label": "压力"},
  {"text": "谢谢你的帮助，真的很感激。", "label": "感激"}
]
```

### TSV 格式

- **train.tsv** / **dev.tsv**：每行 `文本\t标签`，如：
  ```
  今天心情很好	快乐
  压力好大	压力
  ```

## 情感标签

需与 `chinese_emotion/data_processor.py` 中的 `CHINESE_EMOTION_LABELS` 一致，例如：

- 正面：快乐、感激、放松、期待、爱、乐观、自信、平静
- 负面：焦虑、悲伤、愤怒、失望、恐惧、愧疚、孤独、无助、困惑、烦躁、压力、迷茫、自卑、厌烦、冷漠
- 中性：中性

未知标签在训练时会映射为“中性”。

## 数据来源建议（基座不带数据，需自行获取）

- **simplifyweibo_4_moods**（[SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)）：36 万+ 条微博，4 类情感（喜悦/愤怒/厌恶/低落），可用 `chinese_emotion/convert_external_data.py` 转成 train.json，并映射为 快乐/愤怒/厌烦/悲伤。
- **weibo_senti_100k**（同上）：10 万+ 条微博，正/负二分类，可当 2～3 类使用。
- **weibo2018**、**NLPCC 2013 微博情感**等：下载后按本项目格式转换。
- 自建校园心理语料（脱敏后）：按 `{"text":"...","label":"..."}` 整理成 JSON 即可。

转换外部 CSV/TSV 为 train.json、dev.json 示例（在 `bert-master` 下执行）：

```bash
python -m chinese_emotion.convert_external_data --input /path/to/weibo_4moods.csv --output_dir ./data/chinese_emotion --text_col review --label_col mood --label_map "喜悦:快乐 愤怒:愤怒 厌恶:厌烦 低落:悲伤"
```

训练时指定数据目录与格式，例如：

```bash
python train_chinese_emotion.py --data_dir ./data/chinese_emotion --format json
```

若使用 TSV，则：

```bash
python train_chinese_emotion.py --data_dir ./data/chinese_emotion --format tsv --train_file train.tsv --dev_file dev.tsv
```
