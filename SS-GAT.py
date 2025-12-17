import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

# ===================== 1. 数据预处理模块 =====================
class FinancialFraudDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__(transform)
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        
        # 提取8维社会信号特征并标准化（结构-行为-情感三维）
        signal_cols = [
            'degree_centrality',   # 结构信号：节点度中心性
            'clustering_coeff',    # 结构信号：聚类系数
            'interaction_freq',    # 行为信号：交互频率
            'forward_rate',        # 行为信号：转发率
            'behavior_deviation',  # 行为信号：行为偏离度
            'sentiment_value',     # 情感信号：内容情感值
            'support_rate',        # 情感信号：支持率
            'doubt_rate'           # 情感信号：质疑率
        ]
        self.features = self.scaler.fit_transform(self.data[signal_cols])
        
        # 构建图结构边（基于用户交互关系，source_id/target_id为用户ID）
        edges = self.data[['source_id', 'target_id']].drop_duplicates().values.T
        self.edges = torch.tensor(edges, dtype=torch.long)
        
        # 标签（1：诈骗，0：正常）
        self.labels = torch.tensor(self.data['label'].values, dtype=torch.long)
        
    def len(self):
        return 1  # 单图数据集（整网为一个图）
    
    def get(self, idx):
        # 构建PyTorch Geometric数据对象
        x = torch.tensor(self.features, dtype=torch.float)
        edge_index = self.edges
        y = self.labels
        return Data(x=x, edge_index=edge_index, y=y)

# ===================== 2. SS-GAT模型定义 =====================
class SS_GAT(torch.nn.Module):
    """融合社会信号的图注意力网络模型"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        torch.manual_seed(12345)  # 固定随机种子
        
        # 社会信号注意力融合层（自适应分配信号权重）
        self.signal_attention = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels//2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels//2, 1),
            torch.nn.Softmax(dim=1)
        )
        
        # GAT特征提取层（多头注意力捕捉节点关联）
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.6)
        
        # 分类层（图级特征分类）
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels//2, out_channels)
        )

    def forward(self, x, edge_index, batch):
        # 1. 社会信号注意力加权融合
        attention_weights = self.signal_attention(x)
        x = x * attention_weights  # 按权重融合8维社会信号
        
        # 2. GAT图卷积提取特征
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        
        # 3. 全局平均池化（将节点特征聚合为图级特征）
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 4. 分类预测（Dropout防止过拟合）
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

# ===================== 3. 模型训练与评估函数 =====================
def train(model, train_loader, optimizer, criterion):
    """模型训练函数"""
    model.train()
    total_loss = 0
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()  # 反向传播
        optimizer.step() # 参数更新
        optimizer.zero_grad()  # 梯度清零
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader):
    """模型评估函数（返回准确率、F1值、召回率）"""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():  # 关闭梯度计算
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # 取概率最大的类别
            y_true.extend(data.y.numpy())
            y_pred.extend(pred.numpy())
    # 计算核心评估指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return acc, f1, recall

# ===================== 4. 主函数（运行入口） =====================
if __name__ == "__main__":
    # 1. 数据加载（需替换为实际数据路径）
    train_dataset = FinancialFraudDataset("train_data.csv")
    test_dataset = FinancialFraudDataset("test_data.csv")
    
    # 构建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. 模型初始化
    model = SS_GAT(
        in_channels=8,      # 输入特征维度：8维社会信号
        hidden_channels=64, # 隐藏层维度
        out_channels=2,     # 输出类别：诈骗/正常
        heads=4             # GAT多头注意力头数
    )
    
    # 3. 优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()
    
    # 4. 模型训练与评估
    for epoch in range(1, 51):  # 训练50轮
        loss = train(model, train_loader, optimizer, criterion)
        train_acc, train_f1, train_recall = test(model, train_loader)
        test_acc, test_f1, test_recall = test(model, test_loader)
        
        # 打印训练日志
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
              f'Test F1: {test_f1:.4f}, Test Recall: {test_recall:.4f}')
    
    # 5. 模型保存
    torch.save(model.state_dict(), "ss_gat_fraud_detection.pth")
    print("模型已保存至 ss_gat_fraud_detection.pth")
