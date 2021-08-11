# Generative Adversarial Network (GAN) Art
DATA612 group project to develop Generative Adversarial Network (GAN).

# Execution instructions

There are two execution modes for the script:
  * `train` - Trains implementation of CycleGAN to convert photos into Monet paintings, and vice-versa.
  * `eval` - Uses a pre-trained CycleGAN network to convert photos into Monet paintings.

## Train Mode

To execute training mode, invoke the `main.py` script with the `train` argument:

    python main.py train --help

You can provide arguments for the number of epochs and size of batch. These are described in the help menu.

When training is complete, the photo-to-monet and monet-to-photo models are saved to the current working
directory. The template for the directory name follows these rules:

  1. Starts with "p2m_model" for photo-to-monet, "m2p_model" for monet-to-photo
  1. Ends with "e{num}_b{num}_d{num}", where each number corresponds to the number of epochs, batch size, and dropout rate, respectively

At the end of training, a sample of 100 transformations are saved to an image. The template for the image name follows these rules:
  
  1. Starts with "out_p2m" for photo-to-monet, "out_m2p" for monet-to-photo
  1. Ends with "e{num}_b{num}_d{num}", where each number corresponds to the number of epochs, batch size, and dropout rate, respectively
  1. Saved as a PNG image

## Eval Mode

To execute evaluation mode, invoke the `main.py` script with the `eval` argument:

    python main.py eval --help

You can provide arguments for the number of samples to convert, the model to use, and output path. These are described in the help menu.

# Viewing detailed training metrics

This project uses TensorBoard to log and visualize metrics from the training process. Launch TensorBoard from the root directory
of this project using the following command:

    tensorboard --logdir logs/fit

You can use the TensorBoard dashboard to view information including loss per epoch and network graphs.