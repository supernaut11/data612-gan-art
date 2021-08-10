from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, Dropout, Input
from tensorflow_addons.layers import InstanceNormalization

def downsample(num_filters, kernel_size, **kwargs):
    result = keras.Sequential()
    
    result.add(Conv2D(num_filters, kernel_size, **kwargs))
    result.add(InstanceNormalization())
    result.add(Activation("relu"))

    return result

def upsample(num_filters, kernel_size, dropout=0., **kwargs):
    result = keras.Sequential()

    result.add(Conv2DTranspose(num_filters, kernel_size, **kwargs))
    result.add(InstanceNormalization())
    result.add(Activation("relu"))

    if dropout > 0:
        result.add(Dropout(dropout))

    return result

def build(dimensions, n_resnet=9, dropout=0.):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=[256, 256, 3])

    downsamples = [
        downsample(64, (7,7), padding="same", kernel_initializer=init),
        downsample(128, (3,3), strides=(2,2), padding="same", kernel_initializer=init),
        downsample(256, (3,3), strides=(2,2), padding="same", kernel_initializer=init),
        #downsample(512, (3,3), strides=(2,2), padding="same", kernel_initializer=init),
        #downsample(1024, (3,3), strides=(2,2), padding="same", kernel_initializer=init)
    ]

    upsamples = [
        upsample(128, (3,3,), dropout=dropout, strides=(2,2), padding="same", kernel_initializer=init),
        upsample(64, (3,3), dropout=dropout, strides=(2,2), padding="same", kernel_initializer=init)
    ]

    x = in_image

    # Downsampling through the model
    skips = []
    for down in downsamples:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(upsamples, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = Conv2DTranspose(3, (7,7), padding="same", kernel_initializer=init, activation="tanh")(x)

    return keras.Model(in_image, outputs=x)
