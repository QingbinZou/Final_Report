from model import build_resunet
from dataset import RSDataset
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

# 数据路径
rs_path = 'data_train/RS'
gs_path = 'data_train/GS'

# 数据加载（你可以设置 augment=True 在 dataset.py 中开启模糊增强）
train_gen = RSDataset(rs_path, gs_path, batch_size=4)

# 构建模型
model = build_resunet()


# 自定义损失函数：L1 + SSIM + Gradient
def combined_loss(y_true, y_pred):
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    sobel = tf.image.sobel_edges(y_pred)  # [B, H, W, C, 2]
    sobel_x = sobel[..., 0]
    sobel_y = sobel[..., 1]
    grad = tf.reduce_mean(tf.abs(sobel_x) + tf.abs(sobel_y))

    return 0.6 * l1 + 0.2 * ssim_loss + 0.2 * grad



# 编译模型
model.compile(optimizer='adam', loss=combined_loss)

# 保存最佳模型的回调
checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True)

# 开始训练
history = model.fit(train_gen, epochs=100, callbacks=[checkpoint])

# 绘制 loss 曲线图
plt.plot(history.history["loss"], label="Training Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
