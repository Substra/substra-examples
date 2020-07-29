import substratools as tools
import os
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
num_classes = 10

class Algo(tools.algo.Algo):
    def _normalize_X(self, X):

        # Scale images to the [0, 1] range
        X = X.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        X = np.expand_dims(X, -1)

        print(f"X shape: {X.shape}")
        print(f"{X.shape[0]} samples")

        return X

    def _init_new_model(self):

        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def train(self, X, y, models, rank):
        
        X = self._normalize_X(X)

        # convert class vectors to binary class matrices
        y = keras.utils.to_categorical(y, num_classes)

        model = self._init_new_model()

        model.fit(
            X,
            y,
            batch_size=128,
            epochs=1,
            validation_split=0.1,
        )

        return model

    def predict(self, X, model):
        X = self._normalize_X(X)
        y_pred = np.argmax(model.predict(X), axis=-1)
        return y_pred

    def load_model(self, path):
        os.rename(path, path+'.h5')
        model = tensorflow.keras.models.load_model(path+'.h5')
        os.rename(path+'.h5', path)
        return model

    def save_model(self, model, path):
        model.save(path+'.h5')
        os.rename(path+'.h5', path)


if __name__ == "__main__":
    tools.algo.execute(Algo())
