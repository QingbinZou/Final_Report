import os
import cv2
import numpy as np
import pandas as pd
from model.generator import build_generator
from utils.metrics import compute_metrics
from utils.visualization import save_comparison_image
from config import (
    TEST_RS_PATH, TEST_GS_PATH, IMAGE_SIZE, INPUT_SHAPE,
    GENERATOR_WEIGHTS, RESULTS_DIR, EVAL_CSV_PATH
)

# === 初始化模型 ===
generator = build_generator(input_shape=INPUT_SHAPE)
generator.load_weights(GENERATOR_WEIGHTS)

results = []

# === 遍历测试数据 ===
groups = sorted(os.listdir(TEST_RS_PATH))

for group in groups:
    for i in range(1, 49):  # 保证 prev、curr、next 都存在
        idx_name = f"{group}_{i:08d}"
        try:
            prev = cv2.imread(os.path.join(TEST_RS_PATH, group, f"{i-1:08d}.png"))
            curr = cv2.imread(os.path.join(TEST_RS_PATH, group, f"{i:08d}.png"))
            next = cv2.imread(os.path.join(TEST_RS_PATH, group, f"{i+1:08d}.png"))
            gt = cv2.imread(os.path.join(TEST_GS_PATH, group, f"{i:08d}.png"))
        except:
            print(f"⚠ 跳过：图像缺失 - {idx_name}")
            continue

        if any(x is None for x in [prev, curr, next, gt]):
            continue

        def preprocess(img):
            return cv2.resize(img, IMAGE_SIZE).astype(np.float32) / 255.0

        prev = preprocess(prev)
        curr = preprocess(curr)
        next = preprocess(next)
        gt = preprocess(gt)

        input_stack = np.concatenate([prev, curr, next], axis=-1)  # (384, 384, 9)
        pred = generator.predict(np.expand_dims(input_stack, 0))[0]

        psnr_val, ssim_val = compute_metrics(gt, pred)

        results.append({
            "Group": group,
            "Index": i,
            "PSNR": psnr_val,
            "SSIM": ssim_val
        })

        save_path = os.path.join(RESULTS_DIR, f"{idx_name}.png")
        save_comparison_image(curr, pred, gt, save_path)

# === 保存 CSV 指标表格 ===
df = pd.DataFrame(results)
df.to_csv(EVAL_CSV_PATH, index=False)
print(f"✅ 测试完成，结果保存至：{RESULTS_DIR}/")
