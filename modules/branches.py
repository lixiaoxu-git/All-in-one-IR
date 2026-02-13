import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock, ECA
from config import EMBED_DIM

# 共享浅层编码器（下采样2倍）
class ShallowEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)  # B×64×H/2×W/2

# 空域分支（专精高频退化：噪声/压缩/雨纹）
class SpatialBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(64, EMBED_DIM, 3, 2, 1)  # 下采样至H/4
        self.layers = nn.Sequential(*[ResBlock(EMBED_DIM) for _ in range(4)])
        self.att = ECA(EMBED_DIM)  # 高频特征注意力
    
    def forward(self, x):
        x = self.proj(x)
        x = self.layers(x)
        return self.att(x)  # B×256×H/4×W/4

# 频域分支（专精低频退化：模糊/雾霾）
class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(64, EMBED_DIM, 3, 2, 1)
        self.layers = nn.Sequential(*[ResBlock(EMBED_DIM) for _ in range(4)])
    
    def forward(self, x):
        x = self.proj(x)
        # 频域变换（突出低频特征）
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft = torch.fft.ifft2(x_fft).real  # 只保留幅值
        x = self.layers(x_fft)
        return x  # B×256×H/4×W/4