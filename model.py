import torch
import torch.nn as nn
from modules import * 
from config import DEVICE, EMBED_DIM

class RestormerMoE(nn.Module):
    def __init__(self):
        super().__init__()
        # 核心模块组装
        self.shallow_enc = ShallowEncoder().to(DEVICE)
        self.spatial_branch = SpatialBranch().to(DEVICE)
        self.freq_branch = FrequencyBranch().to(DEVICE)
        self.fusion = CrossAttentionFusion().to(DEVICE)
        self.degrade_head = DegradeHead().to(DEVICE)
        self.moe = MoEGate().to(DEVICE)
        self.reconstructor = Reconstructor().to(DEVICE)
    
    def forward(self, x):
        # 1. 浅层编码
        shallow_feat = self.shallow_enc(x)
        # 2. 双分支特征提取
        spatial_feat = self.spatial_branch(shallow_feat)
        freq_feat = self.freq_branch(shallow_feat)
        # 3. 交叉融合
        fusion_feat = self.fusion(spatial_feat, freq_feat)
        # 4. 退化感知
        degrade_prob, degrade_strength = self.degrade_head(fusion_feat)
        # 5. MoE专家修复
        moe_feat = self.moe(fusion_feat, degrade_prob, degrade_strength)
        # 6. 最终重建
        restored = self.reconstructor(moe_feat)
        return restored, degrade_prob  # 返回修复图+退化概率（便于日志）

# 模型初始化（带参数量打印）
def init_model():
    model = RestormerMoE().to(DEVICE)
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params/1e6:.2f}M")
    print(f"可训练参数量: {trainable_params/1e6:.2f}M")
    return model

if __name__ == '__main__':

    model = init_model()
    test_x = torch.randn(1, 3, 256, 256).to(DEVICE)
    try:
        pred, prob = model(test_x)
        print(f"前向传播成功！输出尺寸：{pred.shape}，退化概率维度：{prob.shape}")
    except Exception as e:
        print(f"前向传播报错：{e}")