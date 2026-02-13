import torch
import torch.optim as optim
from tqdm import tqdm
from model import init_model
from loss import CombinedLoss
from config import *

# 模拟数据集（替换为你的真实数据集）
class MockDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1000  # 数据集大小
    
    def __getitem__(self, idx):
        # 退化图像+干净图像（替换为真实数据读取逻辑）
        degrade_img = torch.randn(3, IMG_SIZE, IMG_SIZE)
        clean_img = torch.randn(3, IMG_SIZE, IMG_SIZE)
        return degrade_img, clean_img

def train():
    # 1. 初始化模型/损失/优化器
    model = init_model()
    criterion = CombinedLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    # 余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # 2. 加载数据集
    dataset = MockDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  
    )
    
    # 3. 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for degrade_img, clean_img in pbar:
            # 数据移至设备
            degrade_img = degrade_img.to(DEVICE)
            clean_img = clean_img.to(DEVICE)
            
            # 前向传播
            optimizer.zero_grad()
            pred_img, _ = model(degrade_img)
            loss, loss_dict = criterion(pred_img, clean_img)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 日志更新
            total_loss += loss.item()
            pbar.set_postfix({
                "TotalLoss": f"{loss.item():.4f}",
                "L1": f"{loss_dict['L1']:.4f}",
                "Percep": f"{loss_dict['Perceptual']:.4f}"
            })
        
        # 学习率更新
        scheduler.step()
        # 打印epoch日志
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存模型（每10轮）
        if (epoch+1) % 20 == 0:
            torch.save(
                model.state_dict(),
                f"{SAVE_DIR}/epoch_{epoch+1}.pth"
            )
    
    # 保存最终模型
    torch.save(model.state_dict(), f"{SAVE_DIR}/final_model.pth")
    print("训练完成！最终模型已保存至:", SAVE_DIR)

if __name__ == "__main__":
    train()