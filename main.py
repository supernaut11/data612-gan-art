from gan import discriminator, generator, composite
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

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

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
    #n_epochs = 100
    n_epochs = 10
    n_batch = 1

    n_patch = d_model_A.output_shape[1]

    trainA, trainB = dataset
    trainA = np.asarray(list(trainA.as_numpy_iterator()))
    trainB = np.asarray(list(trainB.as_numpy_iterator()))

    poolA = list()
    poolB = list()

    bat_per_epoch = int(len(list(trainA)) / n_batch)

    n_steps = bat_per_epoch * n_epochs
    print(f"n_steps = {n_steps}")

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

def fresh_main():
    dimensions = (256,256,3)

    model = discriminator.build(dimensions)
    model.summary()
    plot_model(model, to_file="test.png", show_shapes=True, show_layer_names=True)

    g_model_AtoB = generator.build(dimensions)
    g_model_BtoA = generator.build(dimensions)
    d_model_A = discriminator.build(dimensions)
    d_model_B = discriminator.build(dimensions)

    c_model_AtoBtoA = composite.build(g_model_AtoB, d_model_B, g_model_BtoA, dimensions)
    c_model_BtoAtoB = composite.build(g_model_BtoA, d_model_A, g_model_AtoB, dimensions)

    MONET_FILENAMES = tf.io.gfile.glob('./monet_tfrec/*.tfrec')
    PHOTO_FILENAMES = tf.io.gfile.glob('./photo_tfrec/*.tfrec')
    monet_data = load_data(MONET_FILENAMES, labeled=True)
    photo_data = load_data(PHOTO_FILENAMES, labeled=True)

    #dataset = tf.data.Dataset.zip((monet_data, photo_data))

    train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoBtoA, c_model_BtoAtoB, (monet_data, photo_data))

    g_model_BtoA.save("photo_to_monet_model")
    g_model_AtoB.save("monet_to_photo_model")

    _, ax = plt.subplots(5, 2, figsize=(12, 12))
    for i, img in enumerate(photo_data.take(5)):
        prediction = g_model_BtoA(np.asarray([img]), training=False)[0]
        prediction = (np.asarray(prediction) * 127.5 + 127.5).astype(np.uint8)
        img = (np.asarray([img])[0] * 127.5 + 127.5).astype(np.uint8)

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 0].set_title("Input Photo")
        ax[i, 1].set_title("Monet-esque")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
    plt.savefig("out.png")

def cache_main():
    model = keras.models.load_model("photo_to_monet_model")

    dimensions = (256,256,3)
    MONET_FILENAMES = tf.io.gfile.glob('./monet_tfrec/*.tfrec')
    PHOTO_FILENAMES = tf.io.gfile.glob('./photo_tfrec/*.tfrec')
    monet_data = load_data(MONET_FILENAMES, labeled=True)
    photo_data = load_data(PHOTO_FILENAMES, labeled=True)

    _, ax = plt.subplots(20, 2, figsize=(12, 100))
    for i, img in enumerate(photo_data.take(20)):
        prediction = model(np.asarray([img]), training=False)[0]
        prediction = (np.asarray(prediction) * 127.5 + 127.5).astype(np.uint8)
        img = (np.asarray([img])[0] * 127.5 + 127.5).astype(np.uint8)

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 0].set_title("Input Photo")
        ax[i, 1].set_title("Monet-esque")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
    plt.savefig("out.png")

cache_main()