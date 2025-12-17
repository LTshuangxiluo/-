SS-GAT：基于社会信号融合的图注意力网络金融诈骗检测模型
项目介绍
本代码实现了论文《基于图神经网络与社会信号融合的金融诈骗检测与干预机制》中的核心模型 ——SS-GAT（Social Signal Fusion Graph Attention Network）。该模型融合 “结构 - 行为 - 情感” 三维社会信号，通过图注意力网络（GAT）捕捉社交网络中金融诈骗的传播特征，实现高精度的诈骗行为识别。
核心特性
构建 8 维社会信号特征体系，覆盖网络结构、用户行为、文本情感维度；
引入自注意力机制实现社会信号的自适应加权融合；
基于多头 GAT 捕捉社交网络节点间的关联特征；
提供完整的训练、评估、模型保存流程，适配金融诈骗检测场景。
环境配置
基础环境
Python 3.8+
CUDA 11.6+（建议 GPU 运行，CPU 亦可但速度较慢）
依赖安装
1. PyTorch 安装（根据 CUDA 版本选择）
bash
运行
# GPU版本（CUDA 11.8）
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# CPU版本
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
2. PyTorch Geometric（图神经网络核心库）
bash
运行
# 自动匹配PyTorch和CUDA版本
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch_geometric==2.3.1
3. 其他依赖
bash
运行
pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.2.2
数据准备
数据格式要求
输入数据为 CSV 文件（训练集train_data.csv、测试集test_data.csv），需包含以下字段：
字段名	数据类型	说明	取值范围
source_id	int	交互发起用户 ID	任意整型
target_id	int	交互接收用户 ID	任意整型
degree_centrality	float	节点度中心性（结构信号）	0~1
clustering_coeff	float	节点聚类系数（结构信号）	0~1
interaction_freq	float	日均交互次数（行为信号）	≥0
forward_rate	float	内容转发率（行为信号）	0~1
behavior_deviation	float	行为偏离度（与用户均值比，行为信号）	≥0
sentiment_value	float	内容情感值（情感信号）	-1~1（负 = 质疑，正 = 信任）
support_rate	float	评论支持率（情感信号）	0~1
doubt_rate	float	评论质疑率（情感信号）	0~1
label	int	标签（1 = 诈骗，0 = 正常）	0/1
示例数据
source_id	target_id	degree_centrality	clustering_coeff	interaction_freq	forward_rate	behavior_deviation	sentiment_value	support_rate	doubt_rate	label
1001	1002	0.78	0.65	12.5	0.08	1.8	0.92	0.85	0.05	1
1003	1004	0.21	0.32	2.1	0.01	0.5	0.15	0.20	0.10	0
代码运行
1. 数据准备
将训练集和测试集 CSV 文件放入代码同级目录，命名为train_data.csv和test_data.csv。
2. 运行主程序
bash
运行
python ss_gat_fraud_detection.py
3. 输出说明
训练过程中实时打印每轮（Epoch）的损失值、训练集 / 测试集准确率、F1 值、召回率；
训练完成后，模型权重保存为ss_gat_fraud_detection.pth；
核心评估指标说明：
Acc（准确率）：整体识别正确率；
F1（F1 值）：兼顾精确率和召回率，诈骗检测核心指标；
Recall（召回率）：漏检率的补集，越高说明漏检的诈骗越少。
核心模块说明
模块 / 类	功能说明
FinancialFraudDataset	自定义 PyG 数据集类，实现社会信号特征标准化、图结构构建、标签加载
SS_GAT	模型核心类，包含：
1. 社会信号注意力融合层
2. 多头 GAT 特征提取层
3. 图级分类层
train()	模型训练函数，实现反向传播和参数更新
test()	模型评估函数，计算准确率、F1 值、召回率
参数调整建议
参数名	默认值	调整建议
in_channels	8	固定值（8 维社会信号），无需调整
hidden_channels	64	小数据集（<10 万条）设为 32，大数据集（>100 万条）设为 128/256
out_channels	2	固定值（二分类），多分类场景可调整
heads	4	注意力头数，建议 3-8，过多易过拟合，过少则特征捕捉不足
lr	0.001	学习率，收敛过慢调大（如 0.005），震荡过大调小（如 0.0005）
epoch	50	训练轮数，建议通过早停机制（EarlyStopping）终止，避免过拟合
dropout	0.5/0.6	正则化参数，过拟合时调大（如 0.7），欠拟合时调小（如 0.3）
常见问题
1. PyTorch Geometric 安装失败
解决方案：手动下载对应 PyTorch/CUDA 版本的 whl 文件安装，参考PyG 官方文档。
2. 显存不足（GPU Out of Memory）
解决方案：
降低hidden_channels至 32；
减小batch_size（代码中默认 1，无需调整）；
使用 CPU 运行（速度较慢）。
3. 模型评估指标过低
可能原因：
数据量不足（建议至少 1 万条样本）；
特征缺失或异常值未处理；
学习率 / 注意力头数等参数设置不合理。
解决方案：补充数据、清洗特征、调整核心参数。
4. 数据格式错误
报错示例：KeyError: 'degree_centrality'
解决方案：检查 CSV 文件字段名是否与要求一致，避免大小写 / 拼写错误。
模型部署
训练完成的模型可通过以下代码加载并预测新数据：
python
运行
import torch
from ss_gat_fraud_detection import SS_GAT
from torch_geometric.data import Data

# 加载模型
model = SS_GAT(in_channels=8, hidden_channels=64, out_channels=2, heads=4)
model.load_state_dict(torch.load("ss_gat_fraud_detection.pth"))
model.eval()

# 构造新数据示例（需根据实际数据调整）
# 特征矩阵：[节点数, 8维社会信号]
x = torch.tensor([[0.65, 0.58, 10.2, 0.06, 1.5, 0.88, 0.79, 0.08]], dtype=torch.float)
# 边索引：[2, 边数]（source_id, target_id）
edge_index = torch.tensor([[1001], [1005]], dtype=torch.long)
# 批次信息（单图场景设为0）
batch = torch.tensor([0], dtype=torch.long)

# 预测
with torch.no_grad():
    pred = model(x, edge_index, batch)
    pred_label = pred.argmax(dim=1).item()  # 0=正常，1=诈骗
    pred_prob = torch.softmax(pred, dim=1).max().item()  # 预测概率
print(f"预测标签：{'诈骗' if pred_label == 1 else '正常'}，置信度：{pred_prob:.4f}")
免责声明
本代码仅用于学术研究，请勿用于商业或非法用途。实际金融诈骗检测场景中，需结合业务规则、合规要求进行调整。
参考资料
论文：《基于图神经网络与社会信号融合的金融诈骗检测与干预机制》
PyTorch Geometric 官方文档：https://pytorch-geometric.readthedocs.io/
Graph Attention Networks (GAT) 原论文：https://arxiv.org/abs/1710.10903# -
