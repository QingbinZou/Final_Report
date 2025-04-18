import os
import cv2
import numpy as np
from model import build_resunet

# ---------- 配置 ---------- #
rs_dir = "data/rolling"
save_dir = "RS_vs_Test"
os.makedirs(save_dir, exist_ok=True)
video_path = os.path.join(save_dir, "rs_vs_corrected.mp4")
fps = 5  # 每秒帧数
title_height = 50
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2
color_white = (255, 255, 255)

# ---------- 模型加载（768×1024） ---------- #
model = build_resunet(input_shape=(768, 1024, 3))
model.load_weights("best_model.h5")

# ---------- 图像加载函数 ---------- #
def load_img(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 768)).astype(np.float32) / 255.0
    return np.expand_dims(img_resized, axis=0), img_resized

# ---------- 主处理 ---------- #
rs_files = sorted(os.listdir(rs_dir))
frames = []

for idx, filename in enumerate(rs_files):
    rs_path = os.path.join(rs_dir, filename)
    x, x_disp = load_img(rs_path)
    pred = model.predict(x, verbose=0)[0]

    rs_img = (x_disp * 255).astype(np.uint8)
    corrected_img = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    concat_img = np.concatenate([rs_img, corrected_img], axis=1)  # side-by-side

    h, w, _ = concat_img.shape
    canvas = np.zeros((h + title_height, w, 3), dtype=np.uint8)  # 上方预留文字区域
    canvas[title_height:, :] = concat_img

    # 写标题文字（分别写在左/右）
    cv2.putText(canvas, "Rolling Shutter", (80, 36), font, font_scale, color_white, thickness, cv2.LINE_AA)
    cv2.putText(canvas, "Corrected", (w // 2 + 80, 36), font, font_scale, color_white, thickness, cv2.LINE_AA)

    # 保存图像
    save_path = os.path.join(save_dir, f"{idx:03d}_{filename}")
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # 添加帧
    frames.append(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

# ---------- 写入视频 ---------- #
if frames:
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"✅ 视频已保存到 {video_path}")
else:
    print("❌ 没有帧生成，无法创建视频。")

print("✅ 所有拼接图像已保存到 RS_vs_Test/")
