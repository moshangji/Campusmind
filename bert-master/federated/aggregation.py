"""
联邦学习聚合算法实现

包括：
- FedAvg: 基础联邦平均
- FedAvg++: 自适应权重联邦平均
- FedProx: 带近端项的正则化联邦学习
"""

import copy
import logging
from typing import Dict, List, Optional
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def weighted_average(
    params_list: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    加权平均聚合
    
    Args:
        params_list: 参数列表
        weights: 权重列表（需归一化）
        
    Returns:
        聚合后的参数
    """
    if len(params_list) == 0:
        raise ValueError("Empty parameter list")
    
    if len(params_list) != len(weights):
        raise ValueError("Number of parameters and weights must match")
    
    # 归一化权重
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # 初始化聚合参数
    aggregated = {}
    for key in params_list[0].keys():
        aggregated[key] = torch.zeros_like(params_list[0][key])
    
    # 加权求和
    for params, weight in zip(params_list, weights):
        for key in aggregated.keys():
            aggregated[key] += params[key] * weight
    
    return aggregated


class FedAvg:
    """FedAvg: 基础联邦平均算法"""
    
    def aggregate(
        self,
        params_list: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        client_sizes: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        聚合参数
        
        Args:
            params_list: 客户端参数列表
            client_weights: 客户端权重（可选）
            client_sizes: 客户端数据大小（用于计算权重）
            
        Returns:
            聚合后的参数
        """
        if client_weights is None:
            if client_sizes is not None:
                # 按数据量加权
                total_size = sum(client_sizes)
                client_weights = [size / total_size for size in client_sizes]
            else:
                # 均匀加权
                client_weights = [1.0 / len(params_list)] * len(params_list)
        
        return weighted_average(params_list, client_weights)


class FedAvgPlus:
    """FedAvg++: 自适应权重联邦平均
    
    根据客户端数据质量和训练效果自适应调整权重
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: 自适应权重系数（0-1之间）
        """
        self.alpha = alpha
    
    def aggregate(
        self,
        params_list: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        client_sizes: Optional[List[int]] = None,
        client_metrics: Optional[List[Dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        自适应权重聚合
        
        Args:
            params_list: 客户端参数列表
            client_weights: 基础权重
            client_sizes: 客户端数据大小
            client_metrics: 客户端训练指标（用于自适应权重）
            
        Returns:
            聚合后的参数
        """
        # 计算基础权重（按数据量）
        if client_weights is None:
            if client_sizes is not None:
                total_size = sum(client_sizes)
                base_weights = [size / total_size for size in client_sizes]
            else:
                base_weights = [1.0 / len(params_list)] * len(params_list)
        else:
            base_weights = client_weights.copy()
        
        # 自适应调整权重
        if client_metrics is not None:
            adaptive_weights = self._compute_adaptive_weights(
                base_weights,
                client_metrics
            )
        else:
            adaptive_weights = base_weights
        
        logger.info(f"FedAvg++: Using adaptive weights {adaptive_weights}")
        
        return weighted_average(params_list, adaptive_weights)
    
    def _compute_adaptive_weights(
        self,
        base_weights: List[float],
        client_metrics: List[Dict],
    ) -> List[float]:
        """
        计算自适应权重
        
        根据训练损失、准确率等指标调整权重
        """
        # 提取指标
        losses = [m.get('loss', 1.0) for m in client_metrics]
        accuracies = [m.get('accuracy', 0.0) for m in client_metrics]
        
        # 归一化损失（越小越好）
        max_loss = max(losses) if max(losses) > 0 else 1.0
        normalized_losses = [1.0 - (loss / max_loss) for loss in losses]
        
        # 归一化准确率（越大越好）
        max_acc = max(accuracies) if max(accuracies) > 0 else 1.0
        normalized_accs = [acc / max_acc if max_acc > 0 else 0.0 for acc in accuracies]
        
        # 综合质量分数
        quality_scores = [
            self.alpha * n_loss + (1 - self.alpha) * n_acc
            for n_loss, n_acc in zip(normalized_losses, normalized_accs)
        ]
        
        # 结合基础权重和质量分数
        adaptive_weights = [
            base_w * (1 + qs) / 2  # 平滑调整
            for base_w, qs in zip(base_weights, quality_scores)
        ]
        
        # 归一化
        total = sum(adaptive_weights)
        adaptive_weights = [w / total for w in adaptive_weights]
        
        return adaptive_weights


class FedProx:
    """FedProx: 带近端项的正则化联邦学习
    
    在客户端训练时添加全局模型正则化项，减少客户端漂移
    """
    
    def __init__(self, mu: float = 0.01):
        """
        Args:
            mu: 近端项系数
        """
        self.mu = mu
    
    def aggregate(
        self,
        params_list: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        client_sizes: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        聚合参数（FedProx的聚合与FedAvg相同）
        
        注意：FedProx的区别在于客户端训练时添加正则化项
        """
        if client_weights is None:
            if client_sizes is not None:
                total_size = sum(client_sizes)
                client_weights = [size / total_size for size in client_sizes]
            else:
                client_weights = [1.0 / len(params_list)] * len(params_list)
        
        return weighted_average(params_list, client_weights)
    
    def compute_proximal_term(
        self,
        local_params: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算近端项（用于客户端训练）
        
        Args:
            local_params: 本地模型参数
            global_params: 全局模型参数
            
        Returns:
            近端项损失值
        """
        prox_term = 0.0
        for key in local_params.keys():
            if key in global_params:
                diff = local_params[key] - global_params[key]
                prox_term += torch.norm(diff) ** 2
        
        return (self.mu / 2) * prox_term
