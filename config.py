import torch
import os

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型核心配置（4类退化是关键）
IMG_SIZE = 256          # 输入尺寸（可改512）
EMBED_DIM = 256         # 核心特征维度
DEGRADE_TYPES = 4       # 4类基础退化：高频/低频/光照/局部
NUM_EXPERTS = 4         # 4个专家一一对应

# 训练配置
BATCH_SIZE = 64
LR = 1e-4               # 基础学习率
EPOCHS = 300           # 训练轮数
WEIGHT_DECAY = 1e-4     # 权重衰减（防过拟合）
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# 退化类型映射（便于日志/调试）
DEGRADE_MAP = {
    0: "高频退化（噪声/压缩/雨纹）",
    1: "低频退化（模糊/雾霾）",
    2: "光照退化（低光/过曝）",
    3: "局部退化（反光/眩光）"
}