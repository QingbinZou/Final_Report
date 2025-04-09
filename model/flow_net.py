import cv2
import numpy as np

def compute_flow(prev, next):
    """
    简易光流估计（Farneback）：适用于彩色图像 [0~1] 归一化后格式
    输入：prev/next: (H, W, 3) float32
    输出：flow: (H, W, 2)
    """
    prev_gray = cv2.cvtColor((prev * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor((next * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def warp_by_flow(image, flow):
    """
    使用光流对图像进行 warping
    输入：image: (H, W, 3)，flow: (H, W, 2)
    输出：warped: (H, W, 3)
    """
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped
