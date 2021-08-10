from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

def build(origin, discriminator, target, dimensions, learning_rate=0.0002):
    origin.trainable = True

    discriminator.trainable = False
    target.trainable = False

    input_gen = Input(shape=dimensions)
    origin_out = origin(input_gen)
    disc_out = discriminator(origin_out)

    input_id = Input(shape=dimensions)
    output_id = origin(input_id)

    output_forward = target(origin_out)

    target_out = target(input_id)
    output_backward = origin(target_out)

    model = keras.Model([input_gen, input_id], [disc_out, output_id, output_forward, output_backward])
    opt = Adam(learning_rate=learning_rate, beta_1=0.5)
    model.compile(loss=["mse", "mae", "mae", "mae"], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model
