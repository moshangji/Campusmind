# 联邦学习框架

完整的联邦学习实现，支持多客户端协作训练，保护数据隐私。

## 目录结构

```
federated/
├── __init__.py              # 模块初始化
├── client.py                # 客户端训练器
├── server.py                # 服务端聚合器（支持分层）
├── aggregation.py           # 聚合算法（FedAvg, FedAvg++, FedProx）
├── privacy.py               # 隐私保护（差分隐私、安全聚合）
├── utils.py                 # 工具函数
├── run_federated_training.py # 训练主程序
└── README.md                # 本文档
```

## 功能特性

### 1. 聚合算法
- **FedAvg**: 基础联邦平均
- **FedAvg++**: 自适应权重联邦平均（根据数据质量和训练效果调整权重）
- **FedProx**: 带近端项的正则化联邦学习（减少客户端漂移）

### 2. 隐私保护
- **差分隐私（DP）**: 添加高斯噪声保护隐私
- **梯度裁剪**: 限制参数更新范围
- **安全聚合**: 结合DP和裁剪的安全聚合机制

### 3. 分层架构
- **学校-区域-云端**: 支持三层联邦学习架构
- **区域级聚合**: 先在学校内聚合，再在区域间聚合
- **全局聚合**: 最终在云端聚合

## 使用方法

### 基础使用

```python
from federated import FederatedClient, FederatedServer

# 1. 创建客户端
client = FederatedClient(
    client_id=0,
    model_path='bert-base-cased',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 2. 创建服务端
server = FederatedServer(aggregation_method='fedavg')

# 3. 训练循环
for round_num in range(num_rounds):
    # 客户端训练
    client.train()
    params = client.get_parameters()
    
    # 服务端聚合
    global_params = server.aggregate([params], [1.0])
    
    # 分发全局参数
    client.set_parameters(server.broadcast())
```

### 命令行运行

```bash
python federated/run_federated_training.py \
    --model_path /path/to/model \
    --data_dir /path/to/data \
    --num_clients 5 \
    --num_rounds 10 \
    --aggregation fedavg++ \
    --hierarchical \
    --secure
```

## 参数说明

### FederatedClient
- `client_id`: 客户端ID
- `model_path`: 预训练模型路径
- `train_dataset`: 训练数据集
- `eval_dataset`: 验证数据集（可选）
- `num_labels`: 分类标签数量
- `local_epochs`: 本地训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `mu`: FedProx正则化系数

### FederatedServer
- `aggregation_method`: 聚合方法（'fedavg', 'fedavg++', 'fedprox'）
- `use_secure_aggregation`: 是否使用安全聚合
- `dp_epsilon`: 差分隐私epsilon参数
- `dp_delta`: 差分隐私delta参数
- `clip_norm`: 梯度裁剪阈值

## 示例：多客户端训练

```python
import torch
from federated import FederatedClient, FederatedServer
from federated.utils import split_dataset_by_school

# 1. 准备数据
train_dataset = ...  # 你的数据集
client_datasets = split_dataset_by_school(train_dataset, num_clients=5)

# 2. 创建客户端
clients = []
for i in range(5):
    client = FederatedClient(
        client_id=i,
        model_path='bert-base-cased',
        train_dataset=client_datasets[i],
    )
    clients.append(client)

# 3. 创建服务端
server = FederatedServer(aggregation_method='fedavg++')

# 4. 训练
for round_num in range(10):
    # 客户端训练
    client_params_list = []
    client_sizes = []
    for client in clients:
        client.train()
        client_params_list.append(client.get_parameters())
        client_sizes.append(client.get_num_samples())
    
    # 聚合
    global_params = server.aggregate(client_params_list, client_sizes=client_sizes)
    
    # 分发
    for client in clients:
        client.set_parameters(global_params)
```

## 分层联邦学习

```python
from federated import HierarchicalFederatedServer

# 创建分层服务端（3个区域）
server = HierarchicalFederatedServer(
    num_regions=3,
    aggregation_method='fedavg'
)

# 分配客户端到区域
server.assign_clients_to_regions(list(range(10)))

# 分层聚合
global_params = server.aggregate_hierarchical(
    client_params_dict={i: params for i, params in enumerate(client_params_list)},
    client_sizes_dict={i: size for i, size in enumerate(client_sizes)}
)
```

## 隐私保护

```python
from federated import FederatedServer

# 启用安全聚合
server = FederatedServer(
    aggregation_method='fedavg',
    use_secure_aggregation=True,
    dp_epsilon=1.0,      # 隐私预算
    dp_delta=1e-5,       # 失败概率
    clip_norm=1.0,        # 裁剪阈值
)

# 聚合时会自动添加差分隐私噪声
global_params = server.aggregate(client_params_list, client_weights)
```

## 注意事项

1. **数据分片**: 使用 `split_dataset_by_school()` 模拟多校数据分布
2. **参数一致性**: 确保所有客户端使用相同的模型架构
3. **通信开销**: 每轮需要传输完整模型参数，注意网络带宽
4. **隐私预算**: 使用差分隐私时，注意隐私预算的消耗

## 参考论文

- FedAvg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- FedProx: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
- 差分隐私: [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
