import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础残差块（所有分支/专家共用）
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim)
        )
    
    def forward(self, x):
        return x + self.body(x)

# ECA通道注意力（高频分支/专家专用）
class ECA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//16, dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.fc(self.pool(x))
        return x * att