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

for idx, filename in enumerate(rs_files):
    rs_path = os.path.join(rs_dir, filename)
    gs_path = os.path.join(gs_dir, filename)

    # 加载图像并预处理
    def load_img(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (384, 384)).astype(np.float32) / 255.0
        return np.expand_dims(img_resized, axis=(0, -1)), img_resized

    x, x_disp = load_img(rs_path)
    y, y_disp = load_img(gs_path)

    pred = model.predict(x)[0, :, :, 0]

    # 计算指标
    psnr_val = psnr(y_disp, pred, data_range=1.0)
    ssim_val = ssim(y_disp, pred, data_range=1.0)

    results.append({"Index": idx, "Filename": filename, "PSNR": round(psnr_val, 3), "SSIM": round(ssim_val, 4)})

    # 可视化图像拼图
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(x_disp.squeeze(), cmap='gray')
    axs[0].set_title("RS Input")
    axs[1].imshow(pred, cmap='gray')
    axs[1].set_title("Corrected")
    axs[2].imshow(y_disp.squeeze(), cmap='gray')
    axs[2].set_title("GS Ground Truth")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{idx:03d}_{filename}.png"))
    plt.close()

# 保存结果表格
df = pd.DataFrame(results)
df.to_csv(os.path.join(save_dir, "evaluation.csv"), index=False)

print("✅ 测试完成，结果保存在 results/ 文件夹中")
