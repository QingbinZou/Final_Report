import cv2
import numpy as np

def compute_optical_flow(prev, next):
    """
    使用 Farneback 方法计算光流（简化版）
    输入：prev 和 next 为 RGB 图像（0~1 float）
    输出：光流图 (H, W, 2)
    """
    prev_gray = cv2.cvtColor((prev * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor((next * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow

def warp_image(img, flow):
    """
    使用光流将图像warp到参考帧（img必须是 RGB, float32, 0~1）
    """
    h, w = flow.shape[:2]
    flow_map = np.meshgrid(np.arange(w), np.arange(h))
    flow_map = np.stack(flow_map[::-1], axis=-1).astype(np.float32)
    flow_map += flow

    warped = cv2.remap(img, flow_map[..., 0], flow_map[..., 1],
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped
