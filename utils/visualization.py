import os
import matplotlib.pyplot as plt
import numpy as np

def save_comparison_image(curr, pred, gt, save_path):
    """
    拼图并保存：RS输入 / 修复结果 / GT / 残差图
    """
    residual = np.abs(gt - pred)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(curr[..., ::-1])  # BGR to RGB
    axs[0].set_title("RS Input")

    axs[1].imshow(np.clip(pred, 0, 1))
    axs[1].set_title("Corrected")

    axs[2].imshow(gt)
    axs[2].set_title("GT")

    axs[3].imshow(residual)
    axs[3].set_title("Residual")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
