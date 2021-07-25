from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)

    x = Conv2D(n_filters, (3,3), padding="same")(input_layer)
    x = InstanceNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters, (3,3), padding="same")(x)
    x = InstanceNormalization(axis=-1)(x)

    x = Concatenate()([x, input_layer])
    return x

def build(dimensions, n_resnet=9):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=dimensions)

    x = Conv2D(64, (7,7), padding="same")(in_image)
    x = InstanceNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3,3), strides=(2,2), padding="same")(x)
    x = InstanceNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = Conv2D(256, (3,3), strides=(2,2), padding="same")(x)
    x = InstanceNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    for _ in range(n_resnet):
        x = resnet_block(256, x)

    x = Conv2DTranspose(128, (3,3), strides=(2,2), padding="same")(x)
    x = InstanceNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(64, (3,3), strides=(2,2), padding="same")(x)
    x = InstanceNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = Conv2D(3, (7,7), padding="same")(x)
    x = InstanceNormalization(axis=-1)(x)
    outputs = Activation("tanh")(x)

    model = keras.Model(in_image, outputs)
    return model
