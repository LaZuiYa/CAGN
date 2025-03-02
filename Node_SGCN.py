import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.linalg import eigh


# 1. 定义一个GCN层（图卷积层）
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))  # 权重矩阵
        self.bias = nn.Parameter(torch.zeros(out_features))  # 偏置项

    def forward(self, X, A):
        # A: 邻接矩阵，X: 输入特征矩阵
        AX = torch.matmul(A, X)  # 图卷积：A * X
        out = torch.matmul(AX, self.weight) + self.bias  # 线性变换
        return out

# 2. 定义GCN模型
class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)  # 第一层GCN
        self.gcn2 = GCNLayer(hidden_features, out_features)  # 第二层GCN

    def forward(self, X, A):
        # 第一层
        X = self.gcn1(X, A)
        X = F.relu(X)  # 使用ReLU激活函数
        # 第二层
        X = self.gcn2(X, A)
        return X  # 不使用softmax，通常在损失函数中会进行softmax

# 3. 邻接矩阵的归一化
def normalize_adj(adj):
    # 计算每个节点的度数
    degree = torch.sum(adj, dim=1)  # 对每行求和，得到每个节点的度数（一个1D张量）

    # 防止除以0的情况
    degree_inv_sqrt = torch.pow(degree, -0.5)  # 计算度数的-0.5次方
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # 防止除零错误

    # 创建度矩阵D^-1/2
    D_inv_sqrt = torch.diag(degree_inv_sqrt)  # 创建对角矩阵D^-1/2

    # 邻接矩阵归一化：A' = D^-1/2 * A * D^-1/2
    A_norm = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)
    return A_norm


# 4. 训练数据
# 假设我们使用一个小的图，包含4个节点和一个邻接矩阵
X = torch.tensor([[1., 2.],  # 节点特征矩阵
                  [2., 3.],
                  [3., 4.],
                  [4., 5.]])
A = torch.tensor([[0., 1., 0., 1.],  # 邻接矩阵
                  [1., 0., 1., 0.],
                  [0., 1., 0., 1.],
                  [1., 0., 1., 0.]])
y = torch.tensor([0, 1, 0, 1])  # 节点的标签

# 5. 模型、损失函数和优化器
model = GCN(in_features=2, hidden_features=4, out_features=2)  # 输入2维特征，输出2类
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 6. 训练过程
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    # 归一化邻接矩阵
    A_norm = normalize_adj(A)

    # 前向传播
    out = model(X, A_norm)

    # 计算损失
    loss = criterion(out, y)

    # 反向传播
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 7. 测试
model.eval()
with torch.no_grad():
    out = model(X, A_norm)
    _, pred = torch.max(out, dim=1)
    print(f'Predictions: {pred}')
