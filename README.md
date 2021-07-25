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

## Eval Mode

To execute evaluation mode, invoke the `main.py` script with the `eval` argument:

  python main.py eval --help

You can provide arguments for the number of samples to convert, the model to use, and output path. These are described in the help menu.