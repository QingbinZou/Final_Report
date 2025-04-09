import os
import tensorflow as tf
import matplotlib.pyplot as plt

from config import (
    TRAIN_RS_PATH, TRAIN_GS_PATH, IMAGE_SIZE, INPUT_SHAPE,
    EPOCHS, BATCH_SIZE, GAN_WEIGHT,
    GENERATOR_WEIGHTS, DISCRIMINATOR_WEIGHTS, LOSS_PLOT_PATH
)

from model.generator import build_generator
from model.discriminator import build_discriminator
from data.aligned_dataset import AlignedRSDataset
from loss.structure_loss import combined_loss
from loss.gan_loss import discriminator_loss, generator_gan_loss
from utils.logger import log_step

# === 初始化模型 ===
generator = build_generator(input_shape=INPUT_SHAPE)
discriminator = build_discriminator(input_shape=(*IMAGE_SIZE, 3))

g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# === 加载训练数据 ===
train_dataset = AlignedRSDataset(TRAIN_RS_PATH, TRAIN_GS_PATH, size=IMAGE_SIZE, batch_size=BATCH_SIZE, augment=True)

g_loss_hist, d_loss_hist = [], []

# === 训练主循环 ===
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            y_pred = generator(x_batch, training=True)

            real_small, real_large = discriminator(y_batch, training=True)
            fake_small, fake_large = discriminator(y_pred, training=True)

            d_loss = discriminator_loss(real_small, real_large, fake_small, fake_large)
            gan_loss = generator_gan_loss(fake_small, fake_large)
            content_loss = combined_loss(y_batch, y_pred, use_perceptual=True)
            g_loss = content_loss + GAN_WEIGHT * gan_loss

        grads_g = gen_tape.gradient(g_loss, generator.trainable_variables)
        grads_d = disc_tape.gradient(d_loss, discriminator.trainable_variables)

        g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

        if step % 20 == 0:
            log_step(epoch+1, step, g_loss, content_loss, gan_loss, d_loss)

        g_loss_hist.append(g_loss)
        d_loss_hist.append(d_loss)

    generator.save_weights(GENERATOR_WEIGHTS)
    discriminator.save_weights(DISCRIMINATOR_WEIGHTS)

# === 保存 Loss 曲线图 ===
plt.plot(g_loss_hist, label='G_loss')
plt.plot(d_loss_hist, label='D_loss')
plt.title("Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(LOSS_PLOT_PATH)
plt.show()
