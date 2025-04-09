from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Add, Dropout
from tensorflow.keras.models import Model
from config import INPUT_SHAPE

# 简化版残差模块
def res_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = Add()([x, shortcut])
    return x

# 构建 ResUNet（适配 RGB 三帧堆叠输入）
def build_generator(input_shape=INPUT_SHAPE):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = res_block(c1, 32)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = res_block(c2, 64)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = res_block(c3, 128)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    c4 = Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = res_block(c4, 256)

    # Decoder
    u5 = Conv2DTranspose(128, 2, strides=2, padding='same')(c4)
    m5 = Concatenate()([u5, c3])
    c5 = Conv2D(128, 3, activation='relu', padding='same')(m5)
    c5 = res_block(c5, 128)

    u6 = Conv2DTranspose(64, 2, strides=2, padding='same')(c5)
    m6 = Concatenate()([u6, c2])
    c6 = Conv2D(64, 3, activation='relu', padding='same')(m6)
    c6 = res_block(c6, 64)

    u7 = Conv2DTranspose(32, 2, strides=2, padding='same')(c6)
    m7 = Concatenate()([u7, c1])
    c7 = Conv2D(32, 3, activation='relu', padding='same')(m7)
    c7 = res_block(c7, 32)

    outputs = Conv2D(3, 1, activation='sigmoid', padding='same')(c7)  # 输出 RGB 图像

    return Model(inputs, outputs, name="ResUNet_Generator")
