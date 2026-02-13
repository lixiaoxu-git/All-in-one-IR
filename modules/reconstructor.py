import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock
from config import EMBED_DIM

class Reconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器（下采样至H/8）
        self.encoder = nn.Sequential(
            ResBlock(EMBED_DIM),
            nn.Conv2d(EMBED_DIM, EMBED_DIM, 3, 2, 1),
            ResBlock(EMBED_DIM),
            nn.Conv2d(EMBED_DIM, EMBED_DIM, 3, 2, 1)
        )
        # 解码器（上采样回H/4）
        self.decoder = nn.Sequential(
            ResBlock(EMBED_DIM),
            nn.ConvTranspose2d(EMBED_DIM, EMBED_DIM, 4, 2, 1),
            ResBlock(EMBED_DIM),
            nn.ConvTranspose2d(EMBED_DIM, EMBED_DIM, 4, 2, 1)
        )
        # 最终重建（3通道输出）
        self.final = nn.Conv2d(EMBED_DIM, 3, 3, 1, 1)
    
    def forward(self, x):
        # 编码+解码
        x = self.encoder(x)
        x = self.decoder(x)
        # 重建+恢复原尺寸
        x = self.final(x)
        return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)