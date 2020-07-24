# Mnist

This dataset comes from keras.datasets.mnist, contains images (28*28px) of hand drawn digits in grayscale.

## Test and train data samples

Contains 60000 train samples and 10000 test samples

## Data samples structure

Numpy array files with the following shape : (\[nb_samples\], 28, 28) for the features,  (\[nb_samples\],) for the labels

## Opener usage

The opener exposes 4 methods:
* `get_X` returns all features data
* `get_y` returns all labels data
* `save_pred` saves a Numpy array as npy,
* `get_pred` loads the npy saved with `save_pred` and returns a Numpy array
