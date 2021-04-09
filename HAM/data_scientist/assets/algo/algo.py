import logging

import numpy as numpy
import pandas as pd
import os
import shutil

import substratools as tools

import torch
import torchvision.models as models
from torch import optim, nn
from torch.autograd import Variable

logger = logging.getLogger(__name__)
current_directory = os.path.dirname(__file__)

class HAMAlgo(tools.Algo):

    def initialize_model(self, num_classes, use_pretrained=True, predict=False, weight_path=''):
        model_ft = None
        input_size = 0

        model_ft = models.resnet50(use_pretrained)

        if not predict:
            if use_pretrained:
                pretrained_model = torch.load(weight_path)
                model_ft.load_state_dict(pretrained_model)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        return model_ft, input_size


    def train (self, X, y, models, rank):
        
        num_classes = 7
        root_path = os.path.dirname(__file__)
        weight_path = os.path.join(root_path,"resnet.pth")

        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'

        print('Device available: ' + dev)
        device = torch.device(dev)

        model, input_size = self.initialize_model(num_classes=num_classes,
                                                  use_pretrained=False,
                                                  weight_path=weight_path)
        print('model initialized', flush=True)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss().to(device)

        print('Get data in X, y', flush=True)
        for images, labels in zip(X, y):
            N = images.size(0)
            logger.info("images and label")

            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return model


    def predict(self, X, model):
        predictions = []
        pred = []
        with torch.no_grad():
            for i, data in enumerate(X):
                outputs = model(data)
                predictions.append(outputs.max(1, keepdim=False)[1])
        return predictions


    def load_model(self, path):
        model,_ = self.initialize_model(num_classes=7, use_pretrained=False, predict=True)
        model.load_state_dict(torch.load(path))
        return model
    
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

if __name__ == '__main__':
    tools.algo.execute(HAMAlgo())