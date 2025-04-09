import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

class VGGPerceptual:
    """
    使用预训练的 VGG19 网络从中间层提取特征，计算感知损失（perceptual loss）
    默认使用 block3_conv3 层输出。
    """
    def __init__(self, layer_name="block3_conv3"):
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=(384, 384, 3))
        vgg.trainable = False
        self.model = Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output)

    def __call__(self, y_true, y_pred):
        # 若图像尺寸不同，可启用下行 resize
        y_true = tf.image.resize(y_true, (384, 384))
        y_pred = tf.image.resize(y_pred, (384, 384))
        true_feat = self.model(y_true)
        pred_feat = self.model(y_pred)
        return tf.reduce_mean(tf.square(true_feat - pred_feat))
