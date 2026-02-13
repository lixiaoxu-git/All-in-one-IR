import torch
import torch.nn as nn
import torch.nn.functional as F
from config import EMBED_DIM, DEGRADE_TYPES

class DegradeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局特征池化
        self.fc = nn.Linear(EMBED_DIM, DEGRADE_TYPES)  # 4维退化概率
    
    def forward(self, fusion_feat):
        # 全局特征提取
        global_feat = self.global_pool(fusion_feat).flatten(1)  # B×256
        # 4维退化概率（软分类，和为1）
        degrade_prob = F.softmax(self.fc(global_feat), dim=1)  # B×4
        # 退化强度（0~1，概率最大值的均值）
        degrade_strength = degrade_prob.max(dim=1)[0].mean()  # 标量
        return degrade_prob, degrade_strength