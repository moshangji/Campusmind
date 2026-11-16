"""
联邦学习框架模块

提供完整的联邦学习功能，包括：
- 客户端训练器
- 服务端聚合器
- 多种聚合算法（FedAvg, FedAvg++, FedProx）
- 隐私保护机制（差分隐私、安全聚合）
- 分层联邦学习架构
"""

from .client import FederatedClient
from .server import FederatedServer, HierarchicalFederatedServer
from .aggregation import (
    FedAvg,
    FedAvgPlus,
    FedProx,
    weighted_average
)
from .privacy import (
    add_dp_noise,
    clip_gradients,
    SecureAggregator
)

__all__ = [
    'FederatedClient',
    'FederatedServer',
    'HierarchicalFederatedServer',
    'FedAvg',
    'FedAvgPlus',
    'FedProx',
    'weighted_average',
    'add_dp_noise',
    'clip_gradients',
    'SecureAggregator',
]

__version__ = '1.0.0'
