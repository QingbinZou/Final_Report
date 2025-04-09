from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_metrics(gt, pred):
    """
    计算 PSNR 和 SSIM（基于 [0,1] 归一化图像）
    """
    psnr_val = psnr(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, channel_axis=2, data_range=1.0)
    return round(psnr_val, 3), round(ssim_val, 4)
