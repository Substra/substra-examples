import substratools as tools
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
num_classes = 10

#model hyperparameter values
epochs = 1
batch_size = 250

# DP-SGD privacy-specific hyperparameters. See description.md for more information.
l2_norm_clip = 1.5
noise_multiplier = 1.3
num_microbatches = 250
# vanilla SGD parameter
learning_rate = 0.25

class Algo(tools.algo.Algo):
    def _normalize_X(self, X):

        # Scale images to the [0, 1] range
        X = np.array(X, dtype=np.float32) / 255
        # Make sure images have shape (28, 28, 1)
        X = X.reshape(X.shape[0], 28, 28, 1)

        print(f"X shape: {X.shape}")

        return X

    def _init_new_model(self):

        if batch_size % num_microbatches != 0:
            raise ValueError('Batch size should be an integer multiple of the number of microbatches')

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 8,
                                strides=2,
                                padding='same',
                                activation='relu',
                                input_shape=input_shape),
            tf.keras.layers.MaxPool2D(2, 1),
            tf.keras.layers.Conv2D(32, 4,
                                strides=2,
                                padding='valid',
                                activation='relu'),
            tf.keras.layers.MaxPool2D(2, 1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)

        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model

    def train(self, X, y, models, rank):
        
        X = self._normalize_X(X)

        # convert class vectors to binary class matrices
        y = np.array(y, dtype=np.int32)
        y = keras.utils.to_categorical(y, num_classes)

        model = self._init_new_model()

        model.fit(
            X,
            y,
            batch_size=batch_size,
            epochs=epochs,
        )

        epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=X.shape[0], batch_size=batch_size, noise_multiplier=noise_multiplier, epochs=epochs, delta=1/X.shape[0])
        print(f"Computed privacy budget Epsilon: {epsilon}")

        return model

    def predict(self, X, model):
        X = self._normalize_X(X)
        y_pred = np.argmax(model.predict(X), axis=-1)
        return y_pred

    def load_model(self, path):
        os.rename(path, path+'.h5')
        model = tf.keras.models.load_model(path+'.h5')
        os.rename(path+'.h5', path)
        return model

    def save_model(self, model, path):
        model.save(path+'.h5')
        os.rename(path+'.h5', path)


if __name__ == "__main__":
    tools.algo.execute(Algo())
