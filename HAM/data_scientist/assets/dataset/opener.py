"test"


import os
import torch
import glob
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import substratools as tools


class HAM(Dataset):

    def __init__(self, df, transform, paths_image):
        self.df = df
        self.transform = transform
        self.paths_image = paths_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        filename = self.paths_image[index]
        image_id = filename.split('/')[-1]
        df_index = np.where(self.df['image_path'] == image_id)[0]
        X = Image.open(filename)
        y = torch.tensor(int(self.df['dx_idx'][df_index]))

        if self.transform:
            X = self.transform(X)

        return X, y


class HAMOpener(tools.Opener):
    def get_X(self, folders):
        X, _ = self._get_data(folders)
        return X

    def get_y(self, folders):
        _, y = self._get_data(folders)
        return y
    
    def save_predictions(self, y_pred, path):
        torch.save(y_pred, path)
        return

    def get_predictions(self, path):
        # return pd.read_csv(path)
        return torch.load(path)

    def fake_X(self):
        raise NotImplementedError

    def fake_y(self):
        raise NotImplementedError
    
    @classmethod
    def _get_X(cls, data):
        return 
    
    @classmethod
    def _get_y(cls, data):
        return data.loc[:, 'dx_idx']

    @classmethod
    def _fake_data(cls):
        raise NotImplementedError
    
    @classmethod
    def _get_data(cls, folders):
        # #find images files

        paths_image = []
        paths_csv = []
        for folder in folders:
            paths_image += [os.path.join(folder, f) for f in os.listdir(folder) if f[-4:] == '.jpg']
            paths_csv += [os.path.join(folder, f) for f in os.listdir(folder) if f[-4:] == '.csv']

        for path in paths_csv:
            df_val = pd.read_csv(path)
            df_val['dx_idx'] = pd.Categorical(df_val['dx']).codes

        val_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    ])

        params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}
        val_dataset = HAM(df_val.reset_index(), val_transform, paths_image)
        val_loader = DataLoader(val_dataset, **params)

        X = []
        y = []
        for i, j in val_loader:
            X.append(i)
            y.append(j)

        return X, y



