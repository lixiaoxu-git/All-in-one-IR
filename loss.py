import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.transforms import Normalize
from config import DEVICE

# 1. L1损失（基础保真）
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, pred, gt):
        return self.loss(pred, gt)

# 2. MSE损失（提升PSNR）
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, pred, gt):
        return self.loss(pred, gt)

# 3. 感知损失（提升细节/质感）
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练VGG19（仅用前18层）
        vgg = vgg19(pretrained=True).features[:18].to(DEVICE)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        # 图像归一化（适配VGG）
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def forward(self, pred, gt):
        # 归一化+VGG特征提取
        pred_norm = self.norm(pred.clamp(0, 1))
        gt_norm = self.norm(gt.clamp(0, 1))
        pred_feat = self.vgg(pred_norm)
        gt_feat = self.vgg(gt_norm)
        return F.l1_loss(pred_feat, gt_feat)

# 4. 梯度损失（增强边缘/轮廓）
class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel算子（x/y方向）
        self.sobel_x = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).view(1,1,3,3).to(DEVICE)
        self.sobel_y = torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).view(1,1,3,3).to(DEVICE)
    
    def forward(self, pred, gt):
        # 计算梯度
        def calc_grad(img):
            grad = 0
            for c in range(img.size(1)):
                ch = img[:, c:c+1]
                gx = F.conv2d(ch, self.sobel_x, padding=1)
                gy = F.conv2d(ch, self.sobel_y, padding=1)
                grad += torch.sqrt(gx**2 + gy**2 + 1e-8)
            return grad
        
        pred_grad = calc_grad(pred)
        gt_grad = calc_grad(gt)
        return F.l1_loss(pred_grad, gt_grad)

# 组合损失（权重为比赛经验值）
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = L1Loss()
        self.mse = MSELoss()
        self.percep = PerceptualLoss()
        self.grad = GradientLoss()
        # 损失权重（可根据数据集微调）
        self.w_l1 = 1.0
        self.w_mse = 0.1
        self.w_percep = 0.2
        self.w_grad = 0.1
    
    def forward(self, pred, gt):
        loss_l1 = self.l1(pred, gt)
        loss_mse = self.mse(pred, gt)
        loss_percep = self.percep(pred, gt)
        loss_grad = self.grad(pred, gt)
        # 总损失
        total_loss = (
            self.w_l1 * loss_l1
            + self.w_mse * loss_mse
            + self.w_percep * loss_percep
            + self.w_grad * loss_grad
        )
        # 返回总损失+各分项（便于日志）
        return total_loss, {
            "L1": loss_l1.item(),
            "MSE": loss_mse.item(),
            "Perceptual": loss_percep.item(),
            "Gradient": loss_grad.item()
        }