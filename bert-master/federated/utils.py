"""
联邦学习工具函数

包括：
- 数据分片（模拟多校数据）
- 模型参数序列化
- 通信工具
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset, Subset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tokenizer_local(model_path: str):
    """
    从本地加载 tokenizer，不访问 HuggingFace 网络。
    优先：模型目录 from_pretrained → 模型目录内 vocab.txt → 项目下 bert_model 的 vocab.txt
    """
    try:
        from transformers import BertTokenizer
    except ImportError:
        raise ImportError("请先安装: pip install transformers")

    path = os.path.abspath(model_path)
    vocab_in_model = os.path.join(path, "vocab.txt")

    # 1. 模型目录：先尝试完整 from_pretrained，再尝试目录内 vocab.txt
    if os.path.isdir(path):
        try:
            tok = BertTokenizer.from_pretrained(path)
            logger.info(f"Loaded tokenizer from {path}")
            return tok
        except Exception:
            pass
        if os.path.isfile(vocab_in_model):
            tok = BertTokenizer(vocab_file=vocab_in_model)
            logger.info(f"Loaded tokenizer from vocab.txt in {path}")
            return tok

    # 2. 项目内 BERT 预训练目录（如 cased_L-12_H-768_A-12）
    for base in [
        os.path.dirname(path),
        os.path.dirname(os.path.dirname(path)),
        os.path.dirname(os.path.dirname(os.path.dirname(path))),
        "/root/autodl-tmp/Campusmind",
        os.path.expanduser("~/autodl-tmp/Campusmind"),
    ]:
        if not base:
            continue
        vocab_path = os.path.join(base, "bert_model", "cased_L-12_H-768_A-12", "vocab.txt")
        if os.path.isfile(vocab_path):
            tok = BertTokenizer(vocab_file=vocab_path)
            logger.info(f"Loaded tokenizer from {vocab_path}")
            return tok
        vocab_path = os.path.join(base, "bert_model", "vocab.txt")
        if os.path.isfile(vocab_path):
            tok = BertTokenizer(vocab_file=vocab_path)
            logger.info(f"Loaded tokenizer from {vocab_path}")
            return tok

    raise FileNotFoundError(
        f"无法从本地加载 tokenizer。请确保以下之一存在：\n"
        f"  - 目录内有 vocab.txt: {os.path.join(model_path, 'vocab.txt')}\n"
        f"  - 或项目下有: bert_model/cased_L-12_H-768_A-12/vocab.txt"
    )


def split_dataset_by_school(
    dataset: Dataset,
    num_schools: int,
    split_ratio: Optional[List[float]] = None,
    seed: int = 42,
) -> List[Dataset]:
    """
    按学校分割数据集（模拟多校数据）
    
    Args:
        dataset: 原始数据集
        num_schools: 学校数量
        split_ratio: 每个学校的数据比例（默认均匀分配）
        seed: 随机种子
        
    Returns:
        每个学校的数据集列表
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    if split_ratio is None:
        # 均匀分配
        split_ratio = [1.0 / num_schools] * num_schools
    
    # 确保比例归一化
    total_ratio = sum(split_ratio)
    split_ratio = [r / total_ratio for r in split_ratio]
    
    # 计算每个学校的索引范围
    split_indices = []
    start_idx = 0
    for ratio in split_ratio:
        end_idx = start_idx + int(dataset_size * ratio)
        split_indices.append(indices[start_idx:end_idx])
        start_idx = end_idx
    
    # 创建子数据集
    school_datasets = [Subset(dataset, indices) for indices in split_indices]
    
    logger.info(f"Split dataset into {num_schools} schools:")
    for i, school_dataset in enumerate(school_datasets):
        logger.info(f"  School {i}: {len(school_dataset)} samples")
    
    return school_datasets


def save_parameters(
    params: Dict[str, torch.Tensor],
    filepath: str,
):
    """保存模型参数到文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(params, filepath)
    logger.info(f"Saved parameters to {filepath}")


def load_parameters(filepath: str) -> Dict[str, torch.Tensor]:
    """从文件加载模型参数"""
    params = torch.load(filepath)
    logger.info(f"Loaded parameters from {filepath}")
    return params


def serialize_parameters(params: Dict[str, torch.Tensor]) -> bytes:
    """序列化参数为字节流（用于网络传输）"""
    return pickle.dumps(params)


def deserialize_parameters(data: bytes) -> Dict[str, torch.Tensor]:
    """反序列化字节流为参数"""
    return pickle.loads(data)


def compute_model_difference(
    params1: Dict[str, torch.Tensor],
    params2: Dict[str, torch.Tensor],
) -> float:
    """
    计算两个模型参数的差异（L2距离）
    
    Args:
        params1: 模型参数1
        params2: 模型参数2
        
    Returns:
        L2距离
    """
    total_diff = 0.0
    for key in params1.keys():
        if key in params2:
            diff = params1[key] - params2[key]
            total_diff += torch.norm(diff) ** 2
    
    return np.sqrt(total_diff.item())


def check_parameters_shape(
    params_list: List[Dict[str, torch.Tensor]],
) -> bool:
    """
    检查参数形状是否一致
    
    Args:
        params_list: 参数列表
        
    Returns:
        是否一致
    """
    if len(params_list) == 0:
        return True
    
    first_params = params_list[0]
    for params in params_list[1:]:
        if set(params.keys()) != set(first_params.keys()):
            logger.error("Parameter keys mismatch")
            return False
        
        for key in first_params.keys():
            if params[key].shape != first_params[key].shape:
                logger.error(f"Shape mismatch for {key}")
                return False
    
    return True
