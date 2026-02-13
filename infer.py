import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from model import init_model
from config import *

# 图像预处理
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 适配训练数据
])

def infer(img_path, model_path, output_path="restored.png"):
    # 1. 加载模型
    model = init_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 2. 加载并预处理图像
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # 3. 推理
    with torch.no_grad():
        pred_tensor, degrade_prob = model(img_tensor)
    
    # 4. 后处理+保存
    pred_tensor = pred_tensor.squeeze(0).cpu()
    # 反归一化
    pred_tensor = pred_tensor * 0.5 + 0.5
    pred_tensor = torch.clamp(pred_tensor, 0, 1)
    # 转为PIL图像
    pred_img = T.ToPILImage()(pred_tensor)
    pred_img.save(output_path)
    
    # 打印退化分析结果
    degrade_prob = degrade_prob.squeeze(0).cpu().numpy()
    print("=== 退化分析结果 ===")
    for i in range(DEGRADE_TYPES):
        print(f"{DEGRADE_MAP[i]}: {degrade_prob[i]:.4f}")
    print(f"修复结果已保存至: {output_path}")

if __name__ == "__main__":
    # 示例：替换为你的路径
    infer(
        img_path="test.jpg",  # 退化图像路径
        model_path=f"{SAVE_DIR}/final_model.pth",  # 训练好的模型
        output_path="restored.jpg"
    )