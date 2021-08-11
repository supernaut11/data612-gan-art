import argparse
import datetime
from gan import discriminator, generator, cyclegan
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.experimental.AUTOTUNE

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
    dimensions = [256, 256, 3]

    monet_files = tf.io.gfile.glob('./monet_tfrec/*.tfrec')
    photo_files = tf.io.gfile.glob('./photo_tfrec/*.tfrec')
    monet_data = load_data(monet_files, labeled=True).batch(n_batch)
    photo_data = load_data(photo_files, labeled=True).batch(n_batch)
    
    g_model_AtoB = generator.build("monet_gen_model", dimensions, dropout)
    g_model_BtoA = generator.build("photo_gen_model", dimensions, dropout)
    d_model_A = discriminator.build("monet_disc_model", dimensions)
    d_model_B = discriminator.build("photo_disc_model", dimensions)

    cycle_model = cyclegan.build(g_model_AtoB, g_model_BtoA, d_model_A, d_model_B)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-e{n_epochs}_b{n_batch}_d{int(dropout * 100)}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)

    cycle_model.fit(tf.data.Dataset.zip((monet_data, photo_data)), epochs=n_epochs, callbacks=[tensorboard_callback])
    
    g_model_AtoB.save(f"p2m_model_e{n_epochs}_b{n_batch}_d{int(dropout * 100)}")
    g_model_BtoA.save(f"m2p_model_e{n_epochs}_b{n_batch}_d{int(dropout * 100)}")

    evaluate_p2m(g_model_AtoB, f"out_p2m_e{n_epochs}_b{n_batch}_d{int(dropout * 100)}.png", 100)
    evaluate_m2p(g_model_BtoA, f"out_m2p_e{n_epochs}_b{n_batch}_d{int(dropout * 100)}.png", 100)

def evaluate_results(model, out_path, n, data):
    _, ax = plt.subplots(n, 2, figsize=(12, 5 * n))
    for i, img in enumerate(data.take(n)):
        prediction = model(img, training=False)[0].numpy()
        prediction = (np.asarray(prediction) * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 0].set_title("Original")
        ax[i, 1].set_title("Fake")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
    
    plt.savefig(out_path)

def evaluate_p2m(model, out_path, n):
    photo_files = tf.io.gfile.glob('./photo_tfrec/*.tfrec')
    photo_data = load_data(photo_files, labeled=True).batch(1)
    evaluate_results(model, out_path, n, photo_data)

def evaluate_m2p(model, out_path, n):
    monet_files = tf.io.gfile.glob('./monet_tfrec/*.tfrec')
    monet_data = load_data(monet_files, labeled=True).batch(1)
    evaluate_results(model, out_path, n, monet_data)

def eval_main(model_name, out_path, n=20):
    model = keras.models.load_model(model_name, compile=False)
    model.compile()
    evaluate_p2m(model, out_path, n)

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
        train_main(args.epoch, args.batch, args.dropout)
    elif args.command == "eval":
        eval_main(args.model_name, args.output, args.n_samples)
