"""
联邦学习服务端实现

负责：
1. 管理多个客户端
2. 聚合客户端模型参数
3. 分发全局模型
4. 分层联邦学习架构
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

from .aggregation import FedAvg, FedAvgPlus, FedProx, weighted_average
from .privacy import SecureAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedServer:
    """联邦学习服务端"""
    
    def __init__(
        self,
        aggregation_method: str = 'fedavg',
        use_secure_aggregation: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        clip_norm: float = 1.0,
    ):
        """
        初始化联邦学习服务端
        
        Args:
            aggregation_method: 聚合方法 ('fedavg', 'fedavg++', 'fedprox')
            use_secure_aggregation: 是否使用安全聚合
            dp_epsilon: 差分隐私epsilon参数
            dp_delta: 差分隐私delta参数
            clip_norm: 梯度裁剪阈值
        """
        self.aggregation_method = aggregation_method
        self.use_secure_aggregation = use_secure_aggregation
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.clip_norm = clip_norm
        
        # 聚合器
        if aggregation_method == 'fedavg':
            self.aggregator = FedAvg()
        elif aggregation_method == 'fedavg++':
            self.aggregator = FedAvgPlus()
        elif aggregation_method == 'fedprox':
            self.aggregator = FedProx()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # 安全聚合器
        if use_secure_aggregation:
            self.secure_aggregator = SecureAggregator()
        else:
            self.secure_aggregator = None
        
        # 全局模型参数
        self.global_params = None
        
        # 训练历史
        self.history = {
            'round': [],
            'loss': [],
            'accuracy': [],
        }
        
        logger.info(f"FederatedServer initialized with method: {aggregation_method}")
    
    def aggregate(
        self,
        client_params_list: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        client_sizes: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型参数
        
        Args:
            client_params_list: 客户端参数列表
            client_weights: 客户端权重（可选）
            client_sizes: 客户端数据大小（用于加权）
            
        Returns:
            聚合后的全局参数
        """
        logger.info(f"Aggregating parameters from {len(client_params_list)} clients")
        
        # 计算权重
        if client_weights is None:
            if client_sizes is not None:
                total_size = sum(client_sizes)
                client_weights = [size / total_size for size in client_sizes]
            else:
                client_weights = [1.0 / len(client_params_list)] * len(client_params_list)
        
        # 安全聚合
        if self.use_secure_aggregation and self.secure_aggregator:
            aggregated_params = self.secure_aggregator.aggregate(
                client_params_list,
                client_weights,
                self.dp_epsilon,
                self.dp_delta,
                self.clip_norm
            )
        else:
            # 标准聚合
            if self.aggregation_method == 'fedavg++':
                aggregated_params = self.aggregator.aggregate(
                    client_params_list,
                    client_weights,
                    client_sizes
                )
            else:
                aggregated_params = self.aggregator.aggregate(
                    client_params_list,
                    client_weights
                )
        
        self.global_params = aggregated_params
        logger.info("Aggregation completed")
        
        return aggregated_params
    
    def broadcast(self) -> Dict[str, torch.Tensor]:
        """
        广播全局模型参数
        
        Returns:
            全局模型参数
        """
        if self.global_params is None:
            raise ValueError("No global parameters available. Run aggregate() first.")
        
        logger.info("Broadcasting global parameters to clients")
        return copy.deepcopy(self.global_params)
    
    def update_history(self, round_num: int, metrics: Dict):
        """更新训练历史"""
        self.history['round'].append(round_num)
        self.history['loss'].append(metrics.get('loss', 0))
        self.history['accuracy'].append(metrics.get('accuracy', 0))
    
    def get_history(self) -> Dict:
        """获取训练历史"""
        return self.history.copy()


class HierarchicalFederatedServer:
    """分层联邦学习服务端（学校-区域-云端）"""
    
    def __init__(
        self,
        num_regions: int = 3,
        aggregation_method: str = 'fedavg',
    ):
        """
        初始化分层联邦学习服务端
        
        Args:
            num_regions: 区域数量
            aggregation_method: 聚合方法
        """
        self.num_regions = num_regions
        self.aggregation_method = aggregation_method
        
        # 区域级服务器
        self.region_servers = [
            FederatedServer(aggregation_method=aggregation_method)
            for _ in range(num_regions)
        ]
        
        # 云端服务器
        self.cloud_server = FederatedServer(aggregation_method=aggregation_method)
        
        # 客户端分配到区域
        self.region_clients: Dict[int, List[int]] = {i: [] for i in range(num_regions)}
        
        logger.info(f"HierarchicalFederatedServer initialized with {num_regions} regions")
    
    def assign_clients_to_regions(self, client_ids: List[int]):
        """将客户端分配到区域"""
        clients_per_region = len(client_ids) // self.num_regions
        for i, client_id in enumerate(client_ids):
            region_id = i // clients_per_region
            if region_id >= self.num_regions:
                region_id = self.num_regions - 1
            self.region_clients[region_id].append(client_id)
        
        logger.info(f"Assigned {len(client_ids)} clients to {self.num_regions} regions")
    
    def aggregate_hierarchical(
        self,
        client_params_dict: Dict[int, Dict[str, torch.Tensor]],
        client_sizes_dict: Dict[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        分层聚合
        
        1. 区域级聚合：每个区域内的客户端参数聚合
        2. 云端聚合：各区域参数聚合
        
        Args:
            client_params_dict: {client_id: params}
            client_sizes_dict: {client_id: size}
            
        Returns:
            全局聚合参数
        """
        logger.info("Starting hierarchical aggregation")
        
        # 第一步：区域级聚合
        region_params_list = []
        region_weights = []
        
        for region_id in range(self.num_regions):
            region_client_ids = self.region_clients[region_id]
            if not region_client_ids:
                continue
            
            # 收集该区域的客户端参数
            region_params = [client_params_dict[cid] for cid in region_client_ids]
            region_sizes = [client_sizes_dict[cid] for cid in region_client_ids]
            
            # 区域级聚合
            region_server = self.region_servers[region_id]
            region_aggregated = region_server.aggregate(region_params, client_sizes=region_sizes)
            region_params_list.append(region_aggregated)
            
            # 区域权重（按数据量）
            region_weight = sum(region_sizes)
            region_weights.append(region_weight)
            
            logger.info(f"Region {region_id}: Aggregated {len(region_client_ids)} clients")
        
        # 第二步：云端聚合
        total_size = sum(region_weights)
        region_weights = [w / total_size for w in region_weights]
        
        global_params = self.cloud_server.aggregate(
            region_params_list,
            client_weights=region_weights
        )
        
        logger.info("Hierarchical aggregation completed")
        return global_params
    
    def broadcast_hierarchical(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        分层广播
        
        Returns:
            {region_id: global_params}
        """
        global_params = self.cloud_server.broadcast()
        return {region_id: global_params for region_id in range(self.num_regions)}
