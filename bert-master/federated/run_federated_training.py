"""
联邦学习训练主程序

示例：如何使用联邦学习框架训练GoEmotions模型
"""

import os
import sys
import argparse
import logging
from typing import List, Dict
import torch
from torch.utils.data import Dataset

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.client import FederatedClient
from federated.server import FederatedServer, HierarchicalFederatedServer
from federated.utils import split_dataset_by_school, load_tokenizer_local

# GoEmotions数据集类（如果train_goemotions_hf.py不存在，使用这个）
try:
    from train_goemotions_hf import GoEmotionsDataset
except ImportError:
    # 如果导入失败，定义数据集类
    from torch.utils.data import Dataset
    from transformers import BertTokenizer
    
    class GoEmotionsDataset(Dataset):
        def __init__(self, data_path, tokenizer, max_length=64):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.texts = []
            self.labels = []
            
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        text = parts[0]
                        label_ids = parts[1].split(',')
                        label = int(label_ids[0]) if label_ids[0].isdigit() else 27
                        self.texts.append(text)
                        self.labels.append(label)
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_federated_training(
    model_path: str,
    data_dir: str,
    num_clients: int = 5,
    num_rounds: int = 10,
    local_epochs: int = 1,
    aggregation_method: str = 'fedavg',
    use_hierarchical: bool = False,
    use_secure_aggregation: bool = False,
    output_dir: str = './output/federated',
):
    """
    运行联邦学习训练
    
    Args:
        model_path: 预训练模型路径
        data_dir: 数据目录
        num_clients: 客户端数量
        num_rounds: 联邦学习轮数
        local_epochs: 每轮本地训练轮数
        aggregation_method: 聚合方法
        use_hierarchical: 是否使用分层联邦学习
        use_secure_aggregation: 是否使用安全聚合
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"训练日志将同时写入: {os.path.abspath(log_file)}")

    logger.info("=" * 60)
    logger.info("Starting Federated Learning Training")
    logger.info("=" * 60)
    
    # 1. 加载数据（使用本地 tokenizer，不访问 HuggingFace 网络）
    logger.info("Loading datasets...")
    tokenizer = load_tokenizer_local(model_path)
    train_dataset = GoEmotionsDataset(
        os.path.join(data_dir, 'train.tsv'),
        tokenizer,
        max_length=64
    )
    eval_dataset = GoEmotionsDataset(
        os.path.join(data_dir, 'dev.tsv'),
        tokenizer,
        max_length=64
    )
    
    # 2. 分割数据（模拟多校数据）
    logger.info(f"Splitting dataset into {num_clients} clients...")
    client_train_datasets = split_dataset_by_school(train_dataset, num_clients)
    client_eval_datasets = split_dataset_by_school(eval_dataset, num_clients)
    
    # 3. 创建客户端
    logger.info("Creating clients...")
    clients = []
    for i in range(num_clients):
        client = FederatedClient(
            client_id=i,
            model_path=model_path,
            train_dataset=client_train_datasets[i],
            eval_dataset=client_eval_datasets[i],
            num_labels=28,
            local_epochs=local_epochs,
            batch_size=32,
            learning_rate=2e-5,
        )
        clients.append(client)
    
    # 4. 创建服务端
    if use_hierarchical:
        logger.info("Using hierarchical federated learning")
        server = HierarchicalFederatedServer(
            num_regions=3,
            aggregation_method=aggregation_method
        )
        server.assign_clients_to_regions(list(range(num_clients)))
    else:
        logger.info(f"Using {aggregation_method} aggregation")
        server = FederatedServer(
            aggregation_method=aggregation_method,
            use_secure_aggregation=use_secure_aggregation,
        )
    
    # 5. 联邦学习训练循环
    logger.info(f"Starting {num_rounds} rounds of federated learning...")
    
    for round_num in range(1, num_rounds + 1):
        logger.info("-" * 60)
        logger.info(f"Round {round_num}/{num_rounds}")
        logger.info("-" * 60)
        
        # 5.1 客户端本地训练
        client_params_list = []
        client_sizes = []
        client_metrics_list = []
        
        for client in clients:
            # 获取全局参数
            if round_num == 1:
                global_params = None
            else:
                global_params = server.broadcast()
                client.set_parameters(global_params)
            
            # 本地训练
            metrics = client.train(global_params=global_params)
            client_metrics_list.append(metrics)
            
            # 获取参数和数据大小
            params = client.get_parameters()
            client_params_list.append(params)
            client_sizes.append(client.get_num_samples())
            
            logger.info(f"Client {client.client_id}: Loss={metrics['loss']:.4f}, "
                       f"Samples={client.get_num_samples()}")
        
        # 5.2 服务端聚合
        if use_hierarchical:
            client_params_dict = {i: params for i, params in enumerate(client_params_list)}
            client_sizes_dict = {i: size for i, size in enumerate(client_sizes)}
            global_params = server.aggregate_hierarchical(
                client_params_dict,
                client_sizes_dict
            )
        else:
            global_params = server.aggregate(
                client_params_list,
                client_sizes=client_sizes
            )
        
        # 5.3 评估
        if round_num % 5 == 0 or round_num == num_rounds:
            logger.info("Evaluating global model...")
            avg_loss = sum(m['loss'] for m in client_metrics_list) / len(client_metrics_list)
            logger.info(f"Average client loss: {avg_loss:.4f}")
            
            # 更新历史
            server.update_history(round_num, {'loss': avg_loss})
    
    # 6. 保存最终模型
    logger.info("Saving final model...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择一个客户端保存模型（或使用全局参数）
    final_client = clients[0]
    final_client.set_parameters(global_params)
    final_client.model.save_pretrained(os.path.join(output_dir, 'final_model'))
    
    logger.info(f"Model saved to {output_dir}/final_model")
    logger.info("=" * 60)
    logger.info("Federated Learning Training Completed!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--num_clients', type=int, default=5,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=10,
                       help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=1,
                       help='Local training epochs per round')
    parser.add_argument('--aggregation', type=str, default='fedavg',
                       choices=['fedavg', 'fedavg++', 'fedprox'],
                       help='Aggregation method')
    parser.add_argument('--hierarchical', action='store_true',
                       help='Use hierarchical federated learning')
    parser.add_argument('--secure', action='store_true',
                       help='Use secure aggregation')
    parser.add_argument('--output_dir', type=str, default='./output/federated',
                       help='Output directory')
    
    args = parser.parse_args()
    
    run_federated_training(
        model_path=args.model_path,
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        aggregation_method=args.aggregation,
        use_hierarchical=args.hierarchical,
        use_secure_aggregation=args.secure,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
