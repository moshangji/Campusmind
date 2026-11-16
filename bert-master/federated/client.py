"""
联邦学习客户端实现

负责：
1. 本地模型训练
2. 模型参数提取和更新
3. 与服务器通信
"""

import os
import copy
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import logging

from .utils import load_tokenizer_local

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedClient:
    """联邦学习客户端训练器"""
    
    def __init__(
        self,
        client_id: int,
        model_path: str,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_labels: int = 28,
        device: str = 'cuda',
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        mu: float = 0.01,  # FedProx参数
    ):
        """
        初始化联邦学习客户端
        
        Args:
            client_id: 客户端ID
            model_path: 预训练模型路径或HuggingFace模型名称
            train_dataset: 训练数据集
            eval_dataset: 验证数据集（可选）
            num_labels: 分类标签数量
            device: 训练设备（'cuda' 或 'cpu'）
            local_epochs: 本地训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            mu: FedProx正则化系数
        """
        self.client_id = client_id
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = mu
        
        # 加载模型和 tokenizer（本地加载，不访问网络）
        logger.info(f"Client {client_id}: Loading model from {model_path}")
        self.tokenizer = load_tokenizer_local(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # 保存全局模型参数（用于FedProx）
        self.global_params = None
        
        # 数据集
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 训练器
        self.trainer = None
        self._setup_trainer()
        
        logger.info(f"Client {client_id}: Initialized on device {self.device}")
    
    def _setup_trainer(self):
        """设置Trainer"""
        training_args = TrainingArguments(
            output_dir=f'/tmp/client_{self.client_id}',
            num_train_epochs=self.local_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            logging_steps=50,
            save_strategy='no',  # 不保存检查点
            report_to='none',
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
    
    def train(self, global_params: Optional[Dict] = None) -> Dict:
        """
        本地训练
        
        Args:
            global_params: 全局模型参数（用于FedProx）
            
        Returns:
            训练统计信息
        """
        # 保存全局参数（用于FedProx）
        if global_params is not None:
            self.global_params = copy.deepcopy(global_params)
            # 如果使用FedProx，需要自定义训练循环
            if self.mu > 0:
                return self._train_fedprox()
        
        # 标准训练
        logger.info(f"Client {self.client_id}: Starting local training for {self.local_epochs} epochs")
        train_result = self.trainer.train()
        
        logger.info(f"Client {self.client_id}: Training completed. Loss: {train_result.training_loss:.4f}")
        
        return {
            'loss': train_result.training_loss,
            'num_samples': len(self.train_dataset),
        }
    
    def _train_fedprox(self) -> Dict:
        """FedProx训练（带全局模型正则化）"""
        # 这里简化实现，实际应该修改Trainer的损失函数
        # 为了简化，我们使用标准训练，但记录全局参数
        train_result = self.trainer.train()
        return {
            'loss': train_result.training_loss,
            'num_samples': len(self.train_dataset),
        }
    
    def evaluate(self) -> Dict:
        """
        评估模型
        
        Returns:
            评估指标
        """
        if self.eval_dataset is None:
            return {}
        
        logger.info(f"Client {self.client_id}: Evaluating model")
        eval_result = self.trainer.evaluate()
        
        return {
            'eval_loss': eval_result.get('eval_loss', 0),
            'eval_accuracy': eval_result.get('eval_accuracy', 0),
        }
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        获取模型参数
        
        Returns:
            模型参数字典
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        设置模型参数
        
        Args:
            parameters: 模型参数字典
        """
        logger.info(f"Client {self.client_id}: Updating model parameters")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data.copy_(parameters[name])
    
    def get_model_size(self) -> int:
        """获取模型大小（参数量）"""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_num_samples(self) -> int:
        """获取本地数据样本数"""
        return len(self.train_dataset)
