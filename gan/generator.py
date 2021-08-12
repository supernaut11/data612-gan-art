'''
Generator model implementation. Reused for both Monet and photo generator models.

Concepts and portions of code base borrowed and/or repurposed from the following sources:
  * https://www.kaggle.com/amyjang/monet-cyclegan-tutorial#Visualize-our-Monet-esque-photos
  * https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
'''

from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, Dropout, Input
from tensorflow_addons.layers import InstanceNormalization

def downsample(num_filters, kernel_size, **kwargs):
    '''
    Creates a stack of CNN layers representing one level of downsampling.

    Args:
        num_filters (int): Number of convolutional filters to use in this layer
        kernel_size (tuple): 2-tuple describing dimensions of filter kernel to use
        kwargs: Keyword args to send to Conv2D layer (e.g., padding)
    
    Returns:
        keras.Sequential: Stack of layers representing one level of downsampling
    '''
    result = keras.Sequential()
    
    result.add(Conv2D(num_filters, kernel_size, **kwargs))
    result.add(InstanceNormalization())
    result.add(Activation("relu"))

    return result

def upsample(num_filters, kernel_size, dropout=0., **kwargs):
    '''
    Creates a stack of CNN layers representing one level of upsampling.

    Args:
        num_filters (int): Number of convolutional filters to use in this layer
        kernel_size (tuple): 2-tuple describing dimensions of filter kernel to use
        dropout (float): Dropout rate to use in this convolutional layer
        kwargs: Keyword args to send to Conv2D layer (e.g., padding)
    
    Returns:
        keras.Sequential: Stack of layers representing one level of upsampling
    '''
    result = keras.Sequential()

    result.add(Conv2DTranspose(num_filters, kernel_size, **kwargs))
    result.add(InstanceNormalization())
    result.add(Activation("relu"))

    if dropout > 0:
        result.add(Dropout(dropout))

    return result

def build(name, dropout=0.):
    '''
    Implementation of U-Net generator model.

    Args:
        name (str): Name to give to Keras model (helps with Tensorboard outputs)
        dropout (float): Dropout rate to use in each upsampling layer
    
    Returns:
        keras.Model: Generator model for use in CycleGAN
    '''
    init = RandomNormal(stddev=0.02)

    # Downsamples are the descending (i.e., "left") part of the "U" shape, plus the bottom of the "U"
    downsamples = [
        downsample(64, (7,7), padding="same", kernel_initializer=init),
        downsample(128, (3,3), strides=(2,2), padding="same", kernel_initializer=init),
        downsample(256, (3,3), strides=(2,2), padding="same", kernel_initializer=init)
    ]

    # Upsamples are the ascending (i.e., "right") part of the "U" shape
    upsamples = [
        upsample(128, (3,3,), dropout=dropout, strides=(2,2), padding="same", kernel_initializer=init),
        upsample(64, (3,3), dropout=dropout, strides=(2,2), padding="same", kernel_initializer=init)
    ]

    # This part borrows heavily from U-Net tutorial by Amy Jang
    # https://www.kaggle.com/amyjang/monet-cyclegan-tutorial#Visualize-our-Monet-esque-photos
    image_input = Input(shape=[256, 256, 3])
    x = image_input

    # Create downsampling layers (i.e., the left and bottom of the "U"). Also create potential skip
    # origins from these levels
    skips = []
    for down in downsamples:
        x = down(x)
        skips.append(x)

    # Reverse the order of the descending layers so they will match with corresponding layers
    # in the ascending direction. Cut the final layer from the skip possibilities since you 
    # cannot skip across the bottom of the "U"
    skips = reversed(skips[:-1])

    # Create upsampling layers (i.e., the right of the "U"). Concatenate each potential skip
    # origin from the downsampling layers to the corresponding layer in the upsampling section
    for up, skip in zip(upsamples, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    # Create new outputs over the normalized [-1, 1] range using tanh activation function
    x = Conv2DTranspose(3, (7,7), padding="same", kernel_initializer=init, activation="tanh")(x)

    return keras.Model(image_input, outputs=x, name=name)
