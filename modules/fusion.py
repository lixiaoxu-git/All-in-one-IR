import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from config import EMBED_DIM

class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Conv2d(EMBED_DIM, EMBED_DIM*3, 1)  # 生成Q/K/V
        self.proj = nn.Conv2d(EMBED_DIM*4, EMBED_DIM, 1)  # 融合4路特征
    
    def forward(self, spatial_feat, freq_feat):
        B, C, H, W = spatial_feat.shape
        
        # 空域→频域交叉注意力
        q, k, v = self.qkv(spatial_feat).chunk(3, dim=1)
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(freq_feat, 'b c h w -> b c (h w)')
        v = rearrange(freq_feat, 'b c h w -> b (h w) c')
        att = F.softmax(torch.bmm(q, k) / (C**0.5), dim=-1)
        s2f = rearrange(torch.bmm(att, v), 'b (h w) c -> b c h w', h=H, w=W)
        
        # 频域→空域交叉注意力
        q, k, v = self.qkv(freq_feat).chunk(3, dim=1)
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(spatial_feat, 'b c h w -> b c (h w)')
        v = rearrange(spatial_feat, 'b c h w -> b (h w) c')
        att = F.softmax(torch.bmm(q, k) / (C**0.5), dim=-1)
        f2s = rearrange(torch.bmm(att, v), 'b (h w) c -> b c h w', h=H, w=W)
        
        # 融合4路特征：原始空域+原始频域+交叉空域+交叉频域
        fusion_feat = torch.cat([spatial_feat, freq_feat, s2f, f2s], dim=1)
        return self.proj(fusion_feat)  # B×256×H/4×W/4