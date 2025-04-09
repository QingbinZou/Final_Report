import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_small, real_large, fake_small, fake_large):
    real_loss = bce(tf.ones_like(real_small) * 0.9, real_small) + \
                bce(tf.ones_like(real_large) * 0.9, real_large)
    fake_loss = bce(tf.zeros_like(fake_small), fake_small) + \
                bce(tf.zeros_like(fake_large), fake_large)
    return real_loss + fake_loss

def generator_gan_loss(fake_small, fake_large):
    return bce(tf.ones_like(fake_small), fake_small) + \
           bce(tf.ones_like(fake_large), fake_large)
