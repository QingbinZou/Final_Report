import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from model import build_resunet
from dataset import RSDataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ---------------- 参数设置 ---------------- #
rs_path = 'data_train/RS'
gs_path = 'data_train/GS'
sample_path = 'sample_rs.png'
save_pred_dir = 'mid_pred'
os.makedirs(save_pred_dir, exist_ok=True)
log_file = open("training_log.txt", "a")

# 微信推送配置（Server酱）
SCKEY = "SCT276699TyovPkiEM6HAQRirM3oMZJkJa"
send_url = f"https://sctapi.ftqq.com/{SCKEY}.send"

# ---------------- 数据集 ---------------- #
train_gen = RSDataset(rs_path, gs_path, batch_size=2)

# ---------------- 构建模型 ---------------- #
model = build_resunet()

# loss_v1: 基础版（三项并列）
# def loss_v1(y_true, y_pred):
#     l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
#     ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
#     sobel = tf.image.sobel_edges(y_pred)
#     grad = tf.reduce_mean(tf.abs(sobel[..., 0]) + tf.abs(sobel[..., 1]))
#     return 0.6 * l1 + 0.2 * ssim_loss + 0.2 * grad

# loss_v2: 使用 GT 梯度差（Gradient Difference）
def loss_v2(y_true, y_pred):
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    grad_loss = tf.reduce_mean(tf.abs(grad_true - grad_pred))
    return 0.6 * l1 + 0.2 * ssim_loss + 0.2 * grad_loss

# loss_v3: Edge-Aware L1（边缘区域误差加权）
# def loss_v3(y_true, y_pred):
#     ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
#     grad = tf.image.sobel_edges(y_true)
#     edge_strength = tf.reduce_mean(tf.abs(grad[..., 0]) + tf.abs(grad[..., 1]), axis=-1, keepdims=True)
#     l1_edge = tf.reduce_mean(edge_strength * tf.abs(y_true - y_pred))
#     pred_grad = tf.image.sobel_edges(y_pred)
#     pred_grad_strength = tf.reduce_mean(tf.abs(pred_grad[..., 0]) + tf.abs(pred_grad[..., 1]))
#     return 0.5 * l1_edge + 0.3 * ssim_loss + 0.2 * pred_grad_strength

model.compile(optimizer='adam', loss=loss_v2)

# ---------------- 回调函数 ---------------- #
checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True)

early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)


# ---------------- 中间回调：只每50轮执行 ---------------- #
class MidCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        # 提前加载 sample 图像
        img = cv2.imread(sample_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 384)).astype(np.float32) / 255.0
        self.sample = np.expand_dims(img, axis=0)
        self.sample_vis = img

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 != 0:
            return

        pred = self.model.predict(self.sample, verbose=0)[0]
        psnr_val = psnr(self.sample_vis, pred, data_range=1.0)

        # 保存中间图像
        out_img = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
        out_path = os.path.join(save_pred_dir, f"epoch_{epoch + 1:03d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

        # 打印与日志
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1}: PSNR={psnr_val:.2f}, Loss={logs['loss']:.4f}"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        # 微信推送
        try:
            requests.post(send_url, data={
                "title": f"训练进度通知 Epoch {epoch + 1}",
                "desp": f"当前 PSNR: {psnr_val:.2f}\nLoss: {logs['loss']:.4f}"
            }, timeout=3)
        except Exception as e:
            print(f"[警告] 微信推送失败：{e}")


# ---------------- 正式训练 ---------------- #
history = model.fit(
    train_gen,
    epochs=500,
    callbacks=[checkpoint, early_stop, MidCallback()]
)

# ---------------- 绘制 loss 曲线 ---------------- #
plt.plot(history.history["loss"], label="Training Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

log_file.close()
