'''
Discriminator model implementation. Reused for both Monet and photo discriminator models.

Concepts and portions of code base borrowed and/or repurposed from the following sources:
  * https://www.kaggle.com/amyjang/monet-cyclegan-tutorial#Visualize-our-Monet-esque-photos
  * https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
'''

from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

def build(name, improved=True):
    '''
    Implementation of convolutional network for discriminator model. Uses 

    Args:
        name (str): Name to give to Keras model (helps with Tensorboard outputs)
        improved (bool): Flag for using improved CycleGAN implementation, instead of original
    
    Returns:
        keras.Model: Discriminator model for use in CycleGAN
    '''
    # Confgiure inputs to the discriminator
    init = RandomNormal(stddev=0.02)
    image_input = Input(shape=[256, 256, 3])
    
    # Establish convolutional layers for image classification
    x = Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(image_input)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # If we are using the improved model, skip over the additional convolutional layers used
    # in the original CycleGAN implementation
    if not improved:
        x = Conv2D(256, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(512, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(512, (4,4), padding="same", kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)

    outputs = Conv2D(1, (4,4), padding="same", kernel_initializer=init)(x)

    return keras.Model(image_input, outputs, name=name)
