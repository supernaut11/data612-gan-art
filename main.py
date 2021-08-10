import argparse
import datetime
from gan import discriminator, generator, composite, cyclegan
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow_addons.layers import InstanceNormalization

tf.compat.v1.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

def generate_real_images(dataset, n_samples, patch_shape):
    ix = np.random.randint(0, len(list(dataset)), n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

def generate_fake_images(gen_model, dataset, patch_shape):
    X = gen_model.predict(dataset)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def update_image_pool(pool, images, max_size=50):
    selected = list()
    for i in images:
        if len(pool) - 1 < max_size:
            pool.append(i)
            selected.append(i)
        elif np.random.random() < 0.5:
            selected.append(i)
        else:
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = i
    
    return np.asarray(selected)

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, trainA, trainB, n_steps, n_batch):
    n_patch = d_model_A.output_shape[1]

    poolA = list()
    poolB = list()

    for i in range(n_steps):
        X_realA, y_realA = generate_real_images(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_images(trainB, n_batch, n_patch)

        X_fakeA, y_fakeA = generate_fake_images(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_images(g_model_AtoB, X_realA, n_patch)

        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))

IMAGE_SIZE = [256, 256]

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image

def load_data(filenames, labeled=True, ordered=False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset

def train_main(n_epochs, n_batch, dropout):
    dimensions = (256,256,3)

    MONET_FILENAMES = tf.io.gfile.glob('./monet_tfrec/*.tfrec')
    PHOTO_FILENAMES = tf.io.gfile.glob('./photo_tfrec/*.tfrec')
    monet_data = load_data(MONET_FILENAMES, labeled=True)
    photo_data = load_data(PHOTO_FILENAMES, labeled=True)
    dataset = (monet_data, photo_data)

    trainA, trainB = dataset
    trainA = np.asarray(list(trainA.as_numpy_iterator()))
    trainB = np.asarray(list(trainB.as_numpy_iterator()))

    bat_per_epoch = int(len(list(trainA)) / n_batch)

    n_steps = bat_per_epoch * n_epochs
    print(f"n_steps = {n_steps}")

    #lr = learning_rate.GanLearningRateSchedule(0.0002, n_steps)
    #lr = 0.0002

    g_model_AtoB = generator.build(dimensions, 9, dropout)
    g_model_BtoA = generator.build(dimensions, 9, dropout)
    d_model_A = discriminator.build(dimensions, tf.keras.optimizers.schedules.PolynomialDecay(0.0002, n_steps, 0))
    d_model_B = discriminator.build(dimensions, tf.keras.optimizers.schedules.PolynomialDecay(0.0002, n_steps, 0))

    c_model_AtoBtoA = composite.build(g_model_AtoB, d_model_B, g_model_BtoA, dimensions, tf.keras.optimizers.schedules.PolynomialDecay(0.0002, n_steps, 0))
    c_model_BtoAtoB = composite.build(g_model_BtoA, d_model_A, g_model_AtoB, dimensions, tf.keras.optimizers.schedules.PolynomialDecay(0.0002, n_steps, 0))

    train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoBtoA, c_model_BtoAtoB, trainA, trainB, n_steps, n_batch)

    g_model_BtoA.save(f"photo_to_monet_model_e{n_epochs}_b{n_batch}_d{int(dropout * 100)}")
    g_model_AtoB.save(f"monet_to_photo_model_e{n_epochs}_b{n_batch}_d{int(dropout * 100)}")

def new_train(n_epochs, n_batch, dropout):
    dimensions = [256, 256, 3]

    MONET_FILENAMES = tf.io.gfile.glob('./monet_tfrec/*.tfrec')
    PHOTO_FILENAMES = tf.io.gfile.glob('./photo_tfrec/*.tfrec')
    monet_data = load_data(MONET_FILENAMES, labeled=True).batch(n_batch)
    photo_data = load_data(PHOTO_FILENAMES, labeled=True).batch(n_batch)
    
    g_model_AtoB = generator.build(dimensions, 2, dropout)
    g_model_BtoA = generator.build(dimensions, 2, dropout)
    d_model_A = discriminator.build(dimensions)
    d_model_B = discriminator.build(dimensions)

    cycle_model = cyclegan.build(g_model_AtoB, g_model_BtoA, d_model_A, d_model_B)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)

    cycle_model.fit(tf.data.Dataset.zip((monet_data, photo_data)), epochs=n_epochs, callbacks=[tensorboard_callback])
    
    g_model_AtoB.save(f"new_photo_to_monet_model_e{n_epochs}_b{n_batch}_d{int(dropout * 100)}")
    g_model_BtoA.save(f"new_monet_to_photo_model_e{n_epochs}_b{n_batch}_d{int(dropout * 100)}")

def eval_main(model_name, out_path, n=20):
    model = keras.models.load_model(model_name, compile=False)

    model.compile()

    PHOTO_FILENAMES = tf.io.gfile.glob('./photo_tfrec/*.tfrec')
    photo_data = load_data(PHOTO_FILENAMES, labeled=True).batch(1)

    _, ax = plt.subplots(n, 2, figsize=(12, 5 * n))
    for i, img in enumerate(photo_data.take(n)):
        prediction = model(img, training=False)[0].numpy()
        prediction = (np.asarray(prediction) * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 0].set_title("Photo")
        ax[i, 1].set_title("Fake Monet")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
    
    plt.savefig(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for training Monet art generator")
    subparsers = parser.add_subparsers(required=True, dest="command", help="Arguments for specific script modes")
    
    train_parser = subparsers.add_parser("train", description="Arguments for training mode")
    train_parser.add_argument("--epoch", default=5, type=int, help="Number of epochs in training")
    train_parser.add_argument("--batch", default=1, type=int, help="Batch size in training")
    train_parser.add_argument("--dropout", default=0., type=float, help="Percentage dropout to use in generator")

    eval_parser = subparsers.add_parser("eval", description="Arguments for evaluation mode")
    eval_parser.add_argument("model_name", help="Folder name for existing model")
    eval_parser.add_argument("--n-samples", default=20, type=int, help="Number of photos to evaluate")
    eval_parser.add_argument("--output", default="out.png", help="Path for output file")

    args = parser.parse_args()

    if args.command == "train":
        new_train(args.epoch, args.batch, args.dropout)
    elif args.command == "eval":
        eval_main(args.model_name, args.output, args.n_samples)
