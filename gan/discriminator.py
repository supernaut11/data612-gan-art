from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

def build(name, dimensions, learning_rate=0.0002):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=[256, 256, 3])
    
    x = Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(in_image)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # x = Conv2D(256, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(x)
    # x = InstanceNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.2)(x)

    # x = Conv2D(512, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(x)
    # x = InstanceNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.2)(x)

    # x = Conv2D(512, (4,4), padding="same", kernel_initializer=init)(x)
    # x = InstanceNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.2)(x)

    outputs = Conv2D(1, (4,4), padding="same", kernel_initializer=init)(x)

    return keras.Model(in_image, outputs, name=name)
