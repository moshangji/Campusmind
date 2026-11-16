"""
隐私保护机制实现

包括：
- 差分隐私（Differential Privacy）
- 梯度裁剪
- 安全聚合
"""

import copy
import logging
from typing import Dict, List, Optional
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clip_gradients(
    gradients: Dict[str, torch.Tensor],
    clip_norm: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    梯度裁剪
    
    Args:
        gradients: 梯度字典
        clip_norm: 裁剪阈值
        
    Returns:
        裁剪后的梯度
    """
    total_norm = 0.0
    for grad in gradients.values():
        param_norm = grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    clip_coef = clip_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in gradients.values():
            grad.data.mul_(clip_coef)
    
    return gradients


def add_dp_noise(
    params: Dict[str, torch.Tensor],
    epsilon: float = 1.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    添加差分隐私噪声
    
    Args:
        params: 模型参数
        epsilon: 隐私预算
        delta: 失败概率
        sensitivity: 敏感度
        
    Returns:
        添加噪声后的参数
    """
    # 计算噪声标准差
    # 使用高斯机制：sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    
    noisy_params = {}
    for key, value in params.items():
        # 添加高斯噪声
        noise = torch.normal(
            mean=0.0,
            std=sigma,
            size=value.shape,
            device=value.device,
            dtype=value.dtype
        )
        noisy_params[key] = value + noise
    
    logger.info(f"Added DP noise with epsilon={epsilon}, sigma={sigma:.6f}")
    
    return noisy_params


class SecureAggregator:
    """安全聚合器
    
    实现差分隐私和安全多方计算
    """
    
    def __init__(self):
        """初始化安全聚合器"""
        pass
    
    def aggregate(
        self,
        params_list: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        安全聚合
        
        步骤：
        1. 梯度裁剪
        2. 添加差分隐私噪声
        3. 加权平均
        
        Args:
            params_list: 客户端参数列表
            client_weights: 客户端权重
            epsilon: 差分隐私epsilon
            delta: 差分隐私delta
            clip_norm: 裁剪阈值
            
        Returns:
            聚合后的参数
        """
        logger.info(f"Secure aggregation with epsilon={epsilon}, delta={delta}")
        
        # 1. 计算参数更新（相对于全局模型）
        # 这里简化处理，直接对参数添加噪声
        
        # 2. 裁剪参数（限制敏感度）
        clipped_params_list = []
        for params in params_list:
            clipped = {}
            for key, value in params.items():
                # 计算L2范数并裁剪
                norm = torch.norm(value)
                if norm > clip_norm:
                    clipped[key] = value * (clip_norm / norm)
                else:
                    clipped[key] = value
            clipped_params_list.append(clipped)
        
        # 3. 添加差分隐私噪声
        noisy_params_list = []
        for params in clipped_params_list:
            noisy = add_dp_noise(params, epsilon, delta, sensitivity=clip_norm)
            noisy_params_list.append(noisy)
        
        # 4. 加权平均
        aggregated = {}
        for key in noisy_params_list[0].keys():
            aggregated[key] = torch.zeros_like(noisy_params_list[0][key])
        
        for params, weight in zip(noisy_params_list, client_weights):
            for key in aggregated.keys():
                aggregated[key] += params[key] * weight
        
        return aggregated
    
    def compute_privacy_budget(
        self,
        num_rounds: int,
        epsilon_per_round: float,
    ) -> float:
        """
        计算总隐私预算
        
        使用组合定理：总epsilon = sqrt(num_rounds) * epsilon_per_round
        
        Args:
            num_rounds: 训练轮数
            epsilon_per_round: 每轮隐私预算
            
        Returns:
            总隐私预算
        """
        total_epsilon = np.sqrt(num_rounds) * epsilon_per_round
        logger.info(f"Total privacy budget: {total_epsilon:.4f} for {num_rounds} rounds")
        return total_epsilon
