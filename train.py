import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import threading
from datetime import datetime
from model import build_resunet
from dataset import RSDataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ---------- 显存设置 ---------- #
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ---------- 参数设置 ---------- #
rs_path = 'data_train/RS'
gs_path = 'data_train/GS'
sample_path = 'sample_rs.png'
save_pred_dir = 'mid_pred'
os.makedirs(save_pred_dir, exist_ok=True)
log_file = open("training_log.txt", "a")

SCKEY = "填写你的SCKEY"
send_url = f"https://sctapi.ftqq.com/{SCKEY}.send"

# ---------- 数据集加载 ---------- #
train_gen = RSDataset(rs_path, gs_path, size=(768, 1024), batch_size=1)

# ---------- 构建模型 ---------- #
model = build_resunet(input_shape=(768, 1024, 3))

def loss_v2(y_true, y_pred):
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    grad_loss = tf.reduce_mean(tf.abs(grad_true - grad_pred))
    return 0.6 * l1 + 0.2 * ssim_loss + 0.2 * grad_loss

model.compile(optimizer='adam', loss=loss_v2)

# ---------- 回调函数 ---------- #
checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True)
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)

# ---------- 中间结果回调（每100轮，异步执行） ---------- #
class MidCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        img = cv2.imread(sample_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 768)).astype(np.float32) / 255.0
        self.sample = np.expand_dims(img, axis=0)
        self.sample_vis = img

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 100 != 0:
            return

        def process():
            pred = self.model.predict(self.sample, verbose=0)[0]
            psnr_val = psnr(self.sample_vis, pred, data_range=1.0)
            out_img = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
            out_path = os.path.join(save_pred_dir, f"epoch_{epoch + 1:03d}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

            msg = f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1}: PSNR={psnr_val:.2f}, Loss={logs['loss']:.4f}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            try:
                requests.post(send_url, data={
                    "title": f"训练进度通知 Epoch {epoch + 1}",
                    "desp": f"当前 PSNR: {psnr_val:.2f}\nLoss: {logs['loss']:.4f}"
                }, timeout=3)
            except Exception as e:
                print(f"[⚠️] 微信推送失败：{e}")

        threading.Thread(target=process).start()

# ---------- 正式训练 ---------- #
history = model.fit(
    train_gen,
    epochs=500,
    callbacks=[checkpoint, early_stop, MidCallback()]
)

# ---------- 绘制 Loss 曲线 ---------- #
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
