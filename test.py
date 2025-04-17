import os
import cv2
import numpy as np
import pandas as pd
from model import build_resunet
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# 模型加载
model = build_resunet()
model.load_weights("best_model.h5")

# 文件夹路径
rs_dir = "data/rolling"
gs_dir = "data/global"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# 获取所有图像文件名
rs_files = sorted(os.listdir(rs_dir))
gs_files = sorted(os.listdir(gs_dir))
assert rs_files == gs_files, "RS 和 GS 图像数量或名称不匹配"

results = []

# 图像加载函数：返回 RGB 图像
def load_img(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR 格式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    img_resized = cv2.resize(img_rgb, (384, 384)).astype(np.float32) / 255.0
    return np.expand_dims(img_resized, axis=0), img_resized  # [1, H, W, C], [H, W, C]

# 遍历所有图像
for idx, filename in enumerate(rs_files):
    rs_path = os.path.join(rs_dir, filename)
    gs_path = os.path.join(gs_dir, filename)

    x, x_disp = load_img(rs_path)
    y, y_disp = load_img(gs_path)

    pred = model.predict(x)[0]  # [H, W, C]，RGB 格式

    # 计算指标
    psnr_val = psnr(y_disp, pred, data_range=1.0)
    ssim_val = ssim(y_disp, pred, data_range=1.0, channel_axis=2)
    results.append({
        "Index": idx,
        "Filename": filename,
        "PSNR": round(psnr_val, 3),
        "SSIM": round(ssim_val, 4)
    })

    # 图像可视化（保持 RGB，无需 cvtColor）
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(x_disp)
    axs[0].set_title("RS Input")
    axs[1].imshow(np.clip(pred, 0, 1))
    axs[1].set_title("Corrected")
    axs[2].imshow(y_disp)
    axs[2].set_title("GS Ground Truth")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{idx:03d}_{filename}.png"))
    plt.close()

# 保存评价表格
df = pd.DataFrame(results)
df.to_csv(os.path.join(save_dir, "evaluation.csv"), index=False)

print("✅ 测试完成，颜色显示正确，结果保存在 results/ 文件夹中")
