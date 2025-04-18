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

# 图像加载函数
def load_img(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512,384)).astype(np.float32) / 255.0
    return np.expand_dims(img_resized, axis=0), img_resized

# 遍历所有图像
for idx, filename in enumerate(rs_files):
    rs_path = os.path.join(rs_dir, filename)
    gs_path = os.path.join(gs_dir, filename)

    x, x_disp = load_img(rs_path)
    y, y_disp = load_img(gs_path)

    pred = model.predict(x)[0]  # [H, W, C]

    # 计算指标
    psnr_val = psnr(y_disp, pred, data_range=1.0)
    ssim_val = ssim(y_disp, pred, data_range=1.0, channel_axis=2)
    results.append({
        "Index": idx,
        "Filename": filename,
        "PSNR": round(psnr_val, 3),
        "SSIM": round(ssim_val, 4)
    })

    # 误差图计算（取绝对值差异）
    error_map = np.abs(pred - y_disp)
    error_map = np.mean(error_map, axis=2)  # 转灰度
    error_map = np.clip(error_map * 5.0, 0, 1)  # 增强对比度

    # 图像拼图：RS / Pred / GT / Error
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(x_disp)
    axs[0].set_title("RS Input")

    axs[1].imshow(np.clip(pred, 0, 1))
    axs[1].set_title("Corrected")

    axs[2].imshow(y_disp)
    axs[2].set_title("GS Ground Truth")

    axs[3].imshow(error_map, cmap='hot')
    axs[3].set_title("Error Map")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{idx:03d}_{filename}.png"))
    plt.close()

# 保存评价表格
df = pd.DataFrame(results)
df.to_csv(os.path.join(save_dir, "evaluation.csv"), index=False)

# 绘制 PSNR / SSIM 曲线
plt.figure(figsize=(10, 4))
plt.plot(df["Index"], df["PSNR"], label="PSNR", marker='o')
plt.plot(df["Index"], df["SSIM"], label="SSIM", marker='x')
plt.xlabel("Frame Index")
plt.ylabel("Metric Value")
plt.title("PSNR and SSIM over Test Set")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "psnr_ssim_curve.png"))
plt.close()
