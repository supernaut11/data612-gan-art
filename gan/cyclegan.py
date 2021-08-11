import datetime
import tensorflow as tf
from tensorflow.keras import Model

class CycleGAN(Model):
    def __init__(self, monet_gen, photo_gen, monet_disc, photo_disc, lambda_cycle=10):
        super().__init__()
        self._monet_gen = monet_gen
        self._photo_gen = photo_gen
        self._monet_disc = monet_disc
        self._photo_disc = photo_disc
        self._lambda_cycle = lambda_cycle
    
    def compile(self, m_gen_optimizer, p_gen_optimizer,
                m_disc_optimizer, p_disc_optimizer,
                gen_loss_fn, disc_loss_fn, cycle_loss_fn,
                identity_loss_fn):
        super().compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        real_monet, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_monet = self._monet_gen(real_photo, training=True)
            cycled_photo = self._photo_gen(fake_monet, training=True)

            fake_photo = self._photo_gen(real_monet, training=True)
            cycled_monet = self._monet_gen(fake_photo, training=True)

            same_monet = self._monet_gen(real_monet, training=True)
            same_photo = self._photo_gen(real_photo, training=True)

            disc_real_monet = self._monet_disc(real_monet, training=True)
            disc_real_photo = self._photo_disc(real_photo, training=True)

            disc_fake_monet = self._monet_disc(fake_monet, training=True)
            disc_fake_photo = self._photo_disc(fake_photo, training=True)

            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self._lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self._lambda_cycle)
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet, self._lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self._lambda_cycle)

            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        monet_generator_grads = tape.gradient(total_monet_gen_loss, self._monet_gen.trainable_variables)
        photo_generator_grads = tape.gradient(total_photo_gen_loss, self._photo_gen.trainable_variables)
        monet_disc_grads = tape.gradient(monet_disc_loss, self._monet_disc.trainable_variables)
        photo_disc_grads = tape.gradient(photo_disc_loss, self._photo_disc.trainable_variables)

        self.m_gen_optimizer.apply_gradients(zip(monet_generator_grads, self._monet_gen.trainable_variables))
        self.p_gen_optimizer.apply_gradients(zip(photo_generator_grads, self._photo_gen.trainable_variables))
        self.m_disc_optimizer.apply_gradients(zip(monet_disc_grads, self._monet_disc.trainable_variables))
        self.p_disc_optimizer.apply_gradients(zip(photo_disc_grads, self._photo_disc.trainable_variables))

        return {
            "monet_gen_loss" : total_monet_gen_loss,
            "photo_gen_loss" : total_photo_gen_loss,
            "monet_disc_loss" : monet_disc_loss,
            "photo_disc_loss" : photo_disc_loss
        }

def disc_loss(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

    gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + gen_loss

    return total_disc_loss * 0.5

def gen_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)

def cycle_loss(real, cycled, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real - cycled))

    return LAMBDA * loss1

def identity_loss(real, same, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real - same))

    return LAMBDA * 0.5 * loss

def build(monet_gen, photo_gen, monet_disc, photo_disc):
    model = CycleGAN(monet_gen, photo_gen, monet_disc, photo_disc)
    model.compile(
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        gen_loss,
        disc_loss,
        cycle_loss,
        identity_loss
    )

    return model