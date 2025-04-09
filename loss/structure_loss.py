import tensorflow as tf
from loss.perceptual import VGGPerceptual

vgg_loss = VGGPerceptual()

def gradient_loss(y_pred):
    sobel_x = tf.image.sobel_edges(y_pred)[..., 0]
    sobel_y = tf.image.sobel_edges(y_pred)[..., 1]
    return tf.reduce_mean(tf.abs(sobel_x) + tf.abs(sobel_y))

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred, use_perceptual=True):
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)
    grad = gradient_loss(y_pred)
    percep = vgg_loss(y_true, y_pred) if use_perceptual else 0.0

    return 0.5 * l1 + 0.2 * ssim + 0.2 * grad + 0.1 * percep
