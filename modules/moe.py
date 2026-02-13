import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock, ECA
from config import EMBED_DIM, NUM_EXPERTS

# 基础专家模块（不同专家仅调整内部结构）
class Expert(nn.Module):
    def __init__(self, expert_id):
        super().__init__()
        self.expert_id = expert_id
        # 4个专家差异化设计（专精对应退化）
        if expert_id == 0:  # 高频退化：噪声/压缩/雨纹 → ECA注意力
            self.body = nn.Sequential(
                ResBlock(EMBED_DIM),
                ECA(EMBED_DIM),
                ResBlock(EMBED_DIM),
                ECA(EMBED_DIM),
                ResBlock(EMBED_DIM)
            )
        elif expert_id == 1:  # 低频退化：模糊/雾霾 → 空洞卷积（扩大感受野）
            self.body = nn.Sequential(
                nn.Conv2d(EMBED_DIM, EMBED_DIM, 3, 1, 2, dilation=2),
                ResBlock(EMBED_DIM),
                nn.Conv2d(EMBED_DIM, EMBED_DIM, 3, 1, 4, dilation=4),
                ResBlock(EMBED_DIM)
            )
        elif expert_id == 2:  # 光照退化：低光/过曝 → 亮度自适应
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.body = nn.Sequential(*[ResBlock(EMBED_DIM) for _ in range(3)])
        else:  # 局部退化：反光/眩光 → 空间注意力
            self.spatial_att = nn.Sequential(
                nn.Conv2d(EMBED_DIM, 1, 3, 1, 1),
                nn.Sigmoid()
            )
            self.body = nn.Sequential(*[ResBlock(EMBED_DIM) for _ in range(3)])
    
    def forward(self, x):
        if self.expert_id == 2:  # 光照自适应调整
            brightness = self.avg_pool(x).mean(dim=1, keepdim=True)
            x = x * (0.5 / (brightness + 1e-6))  # 亮度归一化
        elif self.expert_id == 3:  # 空间注意力（仅修复局部区域）
            att = self.spatial_att(x)
            x = x * att
        return self.body(x)

# MoE门控融合（核心：按退化概率激活对应专家）
class MoEGate(nn.Module):
    def __init__(self):
        super().__init__()
        # 实例化4个专家
        self.experts = nn.ModuleList([Expert(i) for i in range(NUM_EXPERTS)])
    
    def forward(self, x, degrade_prob, degrade_strength):
        B, C, H, W = x.shape
        # 退化概率转为权重（直接复用4维概率）
        expert_weights = degrade_prob.view(B, NUM_EXPERTS, 1, 1, 1)  # B×4×1×1×1
        
        # 轻度退化（强度<0.3）：不激活专家，直接返回原特征
        if degrade_strength < 0.3:
            return x
        
        # 激活所有专家，按权重融合
        expert_outs = []
        for exp in self.experts:
            out = exp(x).unsqueeze(1)  # B×1×256×H/4×W/4
            expert_outs.append(out)
        expert_outs = torch.cat(expert_outs, dim=1)  # B×4×256×H/4×W/4
        
        # 加权融合 + 残差连接（保留基础特征）
        fused_feat = (expert_outs * expert_weights).sum(dim=1)  # B×256×H/4×W/4
        return x + fused_feat  # 残差避免过拟合