import substratools as tools
import os
from pathlib import Path
import numpy as np
import pandas as pd


class Opener(tools.Opener):
    @classmethod
    def _get_files(cls, folders):
        """Return list of X and y file given a folder location"""
        X_files, y_files = [], []

        for folder in folders:

            X_files_current = Path(folder).rglob('x*.npy')
            y_files_current = Path(folder).rglob('y*.npy')

            X_files.extend(X_files_current)
            y_files.extend(y_files_current)

        return X_files, y_files

    def get_X(self, folders):
        """Get X :-) """

        print("Finding features file...")
        X_files, _ = self._get_files(folders)

        X_paths = []
        for X_file in X_files:
            X_paths.append(np.load(X_file))
        X_paths = np.concatenate(X_paths)
        return X_paths

    def get_y(self, folders):
        """Get y :-)"""
        print("Finding label file...")
        _, y_files = self._get_files(folders)

        print("Loading labels...")
        y = []
        for y_file in y_files:
            y.append(np.load(y_file))
        y = np.concatenate(y)

        return y 

    def save_predictions(self, y_pred, path):
        """Save prediction"""
        with open(path, 'w') as f:
            y_pred.to_csv(f, index=False)

    def get_predictions(self, path):
        """Get predictions which were saved using the save_pred function"""
        return pd.read_csv(path)

    def fake_X(self):
        return np.random.randn(22, 28, 28).astype(np.float32)

    def fake_y(self):
        return np.random.choice(np.arange(10), size=(22)).astype(np.int)
