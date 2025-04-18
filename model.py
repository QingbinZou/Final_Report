from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Dropout, Resizing
from tensorflow.keras.models import Model

def res_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    return Add()([x, shortcut])

def build_resunet(input_shape=(384, 512, 3)):
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
    u5 = UpSampling2D()(c4)
    u5 = Conv2D(128, 3, activation='relu', padding='same')(u5)
    u5 = Resizing(96, 128)(u5)
    m5 = Concatenate()([u5, c3])
    c5 = res_block(m5, 256)

    u6 = UpSampling2D()(c5)
    u6 = Conv2D(64, 3, activation='relu', padding='same')(u6)
    u6 = Resizing(192, 256)(u6)
    m6 = Concatenate()([u6, c2])
    c6 = res_block(m6, 128)

    u7 = UpSampling2D()(c6)
    u7 = Conv2D(32, 3, activation='relu', padding='same')(u7)
    u7 = Resizing(384, 512)(u7)
    m7 = Concatenate()([u7, c1])
    c7 = res_block(m7, 64)

    outputs = Conv2D(3, 1, activation='sigmoid', padding='same')(c7)

    return Model(inputs, outputs)
