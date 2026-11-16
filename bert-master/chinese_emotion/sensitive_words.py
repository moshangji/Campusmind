"""
中文情感分析 - 敏感词检测（隐喻、谐音等）

支持：内置默认词表、外部文件、或由 API 层注入数据库加载的快照（RiskLexiconSnapshot）。
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional, Iterator, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- 内置默认（无数据库或与库合并失败时使用；亦可写入 DB 作为种子）-----

DEFAULT_CRISIS_PHRASES_ZH = (
    "我不想活了",
    "不想活了",
    "不想活",
    "活不下去",
    "活着没意思",
    "活着好累",
    "死了算了",
    "一死了之",
    "一了百了",
    "结束生命",
    "了结性命",
    "自我了断",
    "自我了结",
    "轻生",
    "寻短见",
    "自杀",
    "自残",
    "割腕",
    "跳楼",
    "上吊",
    "烧炭",
    "喝农药",
    "服毒",
    "去死",
    "让我去死",
    "还不如死",
    "弄死自己",
    "结束自己",
    "我想死",
    "好想死",
    "真想死",
    "就想死",
    "有点想死",
    "准备去死",
    "想去死",
    "要去死",
    "别拦我死",
)

DEFAULT_CRISIS_REGEX_ZH = (r"想死(?![了你呐])",)

DEFAULT_CRISIS_PHRASES_EN = (
    "want to die",
    "wish i were dead",
    "wish i was dead",
    "kill myself",
    "killing myself",
    "end my life",
    "end it all",
    "commit suicide",
    "self-harm",
    "cut myself",
    "jump off",
)

DEFAULT_WORDS_NEGATIVE = [
    "不想活了", "去死", "自杀", "自残", "崩溃", "绝望", "没意思",
    "没人懂", "没人关心", "被抛弃", "被孤立", "讨厌自己",
]

DEFAULT_WORDS_METAPHOR = [
    "摆烂", "躺平", "内卷", "emo", "破防", "绷不住了", "裂开",
    "社死", "卷不动", "精神内耗", "佛系",
]

DEFAULT_WORDS_HOMOPHONE = [
    "鸭梨山大", "亚历山大", "桑心", "蓝瘦", "香菇", "想哭",
]


def iter_default_lexicon_rows() -> Iterator[Tuple[str, str]]:
    """用于写入 risk_lexicon 表的 (phrase, category) 种子数据。"""
    for p in DEFAULT_CRISIS_PHRASES_ZH:
        yield (p, "crisis_zh")
    for p in DEFAULT_CRISIS_REGEX_ZH:
        yield (p, "crisis_regex")
    for p in DEFAULT_CRISIS_PHRASES_EN:
        yield (p, "crisis_en")
    for p in DEFAULT_WORDS_NEGATIVE:
        yield (p, "negative")
    for p in DEFAULT_WORDS_METAPHOR:
        yield (p, "metaphor")
    for p in DEFAULT_WORDS_HOMOPHONE:
        yield (p, "homophone")


class SensitiveWordDetector:
    """敏感词检测器：隐喻、谐音、负面表达等"""

    def __init__(self, word_file: Optional[str] = None, snapshot: Any = None):
        """
        word_file: 敏感词文件路径，每行一个词或「词\t类别」。
        snapshot: 来自数据库的快照对象（需含 crisis_phrases_zh 等属性，见 api.utils.risk_lexicon）。
        """
        self.words_negative: List[str] = []
        self.words_metaphor: List[str] = []
        self.words_homophone: List[str] = []
        self._crisis_phrases_zh: List[str] = []
        self._crisis_phrases_en: List[str] = []
        self._crisis_regexes: List[re.Pattern] = []

        if word_file and os.path.isfile(word_file):
            self._load_from_file(word_file)
            self._merge_builtin_crisis()
        elif snapshot is not None:
            self._load_from_snapshot(snapshot)
        else:
            self._load_defaults()

    def _merge_builtin_crisis(self):
        """文件模式：仍合并内置危机短语与正则，避免文件未覆盖自杀相关规则。"""
        self._load_crisis_from_lists(
            list(DEFAULT_CRISIS_PHRASES_ZH),
            list(DEFAULT_CRISIS_PHRASES_EN),
            list(DEFAULT_CRISIS_REGEX_ZH),
        )

    def _load_from_snapshot(self, snap: Any) -> None:
        self._crisis_phrases_zh = list(getattr(snap, "crisis_phrases_zh", []) or [])
        self._crisis_phrases_en = list(getattr(snap, "crisis_phrases_en", []) or [])
        raw_rx = getattr(snap, "crisis_regex_patterns", []) or []
        self._crisis_regexes = []
        for pat in raw_rx:
            try:
                self._crisis_regexes.append(re.compile(pat))
            except re.error as e:
                logger.warning(f"危机正则无效，已跳过: {pat!r} ({e})")
        self.words_negative = list(getattr(snap, "words_negative", []) or [])
        self.words_metaphor = list(getattr(snap, "words_metaphor", []) or [])
        self.words_homophone = list(getattr(snap, "words_homophone", []) or [])
        logger.info(
            "风险词表来自数据库快照: crisis_zh=%s regex=%s negative=%s",
            len(self._crisis_phrases_zh),
            len(self._crisis_regexes),
            len(self.words_negative),
        )

    def _load_crisis_from_lists(self, zh: List[str], en: List[str], regexes: List[str]) -> None:
        self._crisis_phrases_zh = zh
        self._crisis_phrases_en = en
        self._crisis_regexes = []
        for pat in regexes:
            try:
                self._crisis_regexes.append(re.compile(pat))
            except re.error as e:
                logger.warning(f"危机正则无效，已跳过: {pat!r} ({e})")

    def _load_defaults(self) -> None:
        self.words_negative = list(DEFAULT_WORDS_NEGATIVE)
        self.words_metaphor = list(DEFAULT_WORDS_METAPHOR)
        self.words_homophone = list(DEFAULT_WORDS_HOMOPHONE)
        self._load_crisis_from_lists(
            list(DEFAULT_CRISIS_PHRASES_ZH),
            list(DEFAULT_CRISIS_PHRASES_EN),
            list(DEFAULT_CRISIS_REGEX_ZH),
        )

    def _load_from_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                word = parts[0].strip()
                category = parts[1].strip() if len(parts) > 1 else "negative"
                if category == "metaphor":
                    self.words_metaphor.append(word)
                elif category == "homophone":
                    self.words_homophone.append(word)
                else:
                    self.words_negative.append(word)
        logger.info(
            "Loaded sensitive words from file: negative=%s, metaphor=%s, homophone=%s",
            len(self.words_negative),
            len(self.words_metaphor),
            len(self.words_homophone),
        )

    def detect(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        检测文本中的敏感词。
        返回 {"negative": [(词, 原文片段)], "metaphor": [...], "homophone": [...]}
        """
        result: Dict[str, List[Tuple[str, str]]] = {"negative": [], "metaphor": [], "homophone": []}
        for word in self.words_negative:
            if word in text:
                result["negative"].append((word, word))
        for word in self.words_metaphor:
            if word in text:
                result["metaphor"].append((word, word))
        for word in self.words_homophone:
            if word in text:
                result["homophone"].append((word, word))
        return result

    def has_risk(self, text: str) -> bool:
        """是否存在高风险表达（自杀/自伤意念或行为），命中即应判 crisis。"""
        if not text or not str(text).strip():
            return False
        s = str(text).strip()
        if any(p in s for p in self._crisis_phrases_zh):
            return True
        for rx in self._crisis_regexes:
            if rx.search(s):
                return True
        lo = s.lower()
        if any(p in lo for p in self._crisis_phrases_en):
            return True
        return False
