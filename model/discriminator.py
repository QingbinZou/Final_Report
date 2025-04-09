from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Input, Dropout
from tensorflow.keras.models import Model
from config import IMAGE_SIZE

def build_discriminator(input_shape=(*IMAGE_SIZE, 3)):
    inp = Input(shape=input_shape)

    # ↓ 降通道数，加入 Dropout
    x = Conv2D(32, 4, strides=2, padding='same')(inp)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    small_out = Conv2D(1, 4, strides=1, padding='same', activation='sigmoid')(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    large_out = Conv2D(1, 4, strides=1, padding='same', activation='sigmoid')(x)

    return Model(inp, [small_out, large_out], name="MultiScaleDiscriminator")
