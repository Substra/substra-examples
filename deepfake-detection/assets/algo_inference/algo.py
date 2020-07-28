#Algo based on inference demo Kaggle DFDC notebook : https://www.kaggle.com/humananalog/inference-demo 
#and https://www.kaggle.com/humananalog/binary-image-classifier-training-demo for training

import substratools as tools
import os
import numpy as np
import cv2

import pandas as pd
import time

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import torch.nn.functional as F

class MyResNeXt(models.resnet.ResNet):
    """
        The ResNeXt model is a very simple binary image classifier using BCELoss. 
        The model will be initialized with the standard pretrained ImageNet weights from torchvision.
    """
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)
        self.fc = nn.Linear(2048, 1)

class Algo(tools.algo.Algo):

    normalize_transform = None
    frames_per_video = None
    face_extractor = None
    input_size = None
    gpu = None
    root_path = os.path.dirname(__file__)

    def _normalize_X(self, X):
        # mean and standard deviation from the set of images used to train the resnet,
        # based on the torchvision standard for pretrained models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        #define a normalization function with the previous parameters
        normalize_transform = Normalize(mean, std)

        normalized_x = normalize_transform(X)
        return normalized_x

    def _init_new_model(self):

        #input size of the deepfake detection model
        self.input_size = 224

        #Init the Resnet deepfake detection model
        model = MyResNeXt().to(self.gpu)

        return model

    def _predict_pandas(self, y_pred):
        return pd.DataFrame(columns=['proba_fake'], data=y_pred)

    def _freeze_until(self, model, param_name):
        found_name = False
        for name, params in model.named_parameters():
            if name == param_name:
                found_name = True
            params.requires_grad = found_name
    
    def _fit(self, model, train_videos, y_true, epochs):
        global history, iteration, epochs_done, lr, optimizer

        batch_size = 64

        lr = 0.01
        wd = 0.

        history = { "train_bce": [], "val_bce": [] }
        iteration = 0
        epochs_done = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


        with tqdm(total=len(train_videos), leave=False) as pbar:
            for epoch in range(epochs):
                pbar.reset()
                pbar.set_description("Epoch %d" % (epochs_done + 1))
                
                bce_loss = 0
                total_examples = 0

                model.train(True)

                for filename, label in zip(train_videos, y_true):
                    #load faces and train the model
                    model, batch_bce = self._train_on_video(filename, label, batch_size=batch_size, model=model, face_extractor=self.face_extractor)
                    
                    bce_loss += batch_bce * batch_size
                    history["train_bce"].append(batch_bce)

                    total_examples += batch_size
                    iteration += 1
                    pbar.update()

                bce_loss /= total_examples
                epochs_done += 1
                print("Epoch: %3d, train BCE: %.4f" % (epochs_done, bce_loss))
                
                
        return model

    def train(self, X_files, y, models, rank):
       
        print("Loading features...")
        train_videos = X_files
        y_true = y
        print("Nb of train videos:", len(train_videos))

        print("PyTorch version:", torch.__version__)
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())

        #find gpu if there is one, else use cpu
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("torch.device: ", self.gpu)

        print(f"Current working directory: {os.getcwd()}")

        #init and load the face extractor (implemented in deepfakes-inference-demo/helpers/face_extract_1)
        self.face_extractor = self._load_face_extractor()

        if not models: #if input model is not given
            print("No input model, creating a new one pretrained on ImageNet")
            model = self._init_new_model()
            #load a pretrained Resnet on ImageNet for the deepfake detection model (result : 0.7268 on train_data_samples_0 to 4)
            """ model scores (log-loss):
            result on train_data_samples_0 to 4 when trained on 0_4: 0.4787
            result on train_data_samples_0 to 4 when trained on all train_data_damples: 0.4557
            result on test_data_samples when trained on all train_data_damples: 0.5779 
            result on test_data_samples when trained 2* on all train_data_damples: 0.5733 (on cpu: Elapsed 4918.225055 sec. Average per video: 15.369453 sec)
            """
            model.fc = nn.Linear(2048, 1000)
            checkpoint = torch.load(os.path.join(self.root_path,"deepfakes-inference-demo/resnext50_32x4d-7cdf4587.pth"))
            model.load_state_dict(checkpoint)
            # Override the existing FC layer with a new one.
            model.fc = nn.Linear(2048, 1)
            del checkpoint
        else: 
            model = models[0]
            print("training input model", model.__name__)

        #Freeze the early layers of the model
        self._freeze_until(model, "layer4.0.conv1.weight")

        #put the model in training mode
        _ = model.train()

        #init speedtest
        start_time = time.time()

        model = self._fit(model, train_videos, y_true, epochs=1)

        #compute speedtest results
        elapsed = time.time() - start_time
        print("Elapsed %f sec. Average per video: %f sec." % (elapsed, elapsed / len(X_files)))

        return model


    def predict(self, X_files, model):
        """
            Get predictions from test data.
            This task corresponds to the creation of a testtuple on the Substra
            Platform.

            :param X_files: testing data samples paths loaded with `Opener.get_X()`. 
                            Example : array(['data/DFDC/train_sample_videos/eqvuznuwsa.mp4', ...])
            :param model: input model load with `Algo.load_model()` used for predictions.
            :type X_files: numpy.ndarray
            :type model: type
            :return: predictions object.
            :rtype: type de la valeur de retour
        """

        test_videos = X_files
        print("Nb of Test videos:", len(test_videos))
        print("PyTorch version:", torch.__version__)
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())

        #find gpu if there is one, else use cpu
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.gpu)

        print("Current working directory:", os.getcwd())

        #init and load the face extractor (implemented in deepfakes-inference-demo/helpers/face_extract_1)
        face_extractor = self._load_face_extractor()

        #load the trained model by humananalog for higher scores
        """
        #init model 
        model = self._init_new_model()

        #load a pretrained Resnet on ImageNet+DFDC for the deepfake detection model 
        #result : 0.3639 on train_data_samples_0 to 4
        # 0.3299 on test_data_samples
        checkpoint = torch.load(os.path.join(self.root_path,"deepfakes-inference-demo/resnext.pth", map_location=self.gpu) 
        model.load_state_dict(checkpoint)
        """

        #put the model in evaluation mode
        _ = model.eval()

        #init speedtest
        start_time = time.time()

        #get a list of predictions for each video
        predictions = self._predict_on_video_set(X_files, num_workers=4, model=model, face_extractor=face_extractor)

        #compute speedtest results
        elapsed = time.time() - start_time
        print("Elapsed %f sec. Average per video: %f sec." % (elapsed, elapsed / len(X_files)))

        #format the predictions in a pandas dataframe
        #y_pred = pd.DataFrame({"filename": test_videos, "label": predictions})

        return self._predict_pandas(predictions)

    def load_model(self, path):
        model = self._init_new_model()
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        return model

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
        return None

    def _predict_on_video(self, video_path, batch_size, model, face_extractor):
        """
            Get prediction for a video : extract cropped faces from the video with the face_extractor, then 
            do a prediction for each face and return the mean of these predictions as the prediction of the video being fake

            :param X_files: testing data samples paths loaded with `Opener.get_X()`. 
                            Example : array(['data/DFDC/train_sample_videos/eqvuznuwsa.mp4', ...])
            :param model: input model load with `Algo.load_model()` used for predictions.
            :type X_files: numpy.ndarray
            :type model: model
            :return: predicted probability of the video being FAKE
            :rtype: type de la valeur de retour
        """

        try:
            print(f"extracting faces from: {video_path}")
            # Find the faces for N frames in the video.
            #extract face crops for each frames in the video
            faces = face_extractor.process_video(video_path)
            # Only look at one face per frame.
            face_extractor.keep_only_best_face(faces)
            
            if len(faces) > 0:
                #store the resized faces in x
                x = np.zeros((batch_size, self.input_size, self.input_size, 3), dtype=np.uint8)

                # If we found any faces, prepare them for the model.
                # n = counter of faces 
                n = 0
                for frame_data in faces:
                    for face in frame_data["faces"]:
                        # Resize to the model's required input size.
                        # We keep the aspect ratio intact and add zero
                        # padding if necessary.                    
                        resized_face = self._isotropically_resize_image(face, self.input_size)
                        resized_face = self._make_square_image(resized_face)

                        if n < batch_size:
                            x[n] = resized_face
                            n += 1
                        else:
                            print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
                
                if n > 0:
                    # put the array of cropped resized faces x into a tensor
                    x = torch.tensor(x, device=self.gpu).float()

                    # Preprocess the images.
                    x = x.permute((0, 3, 1, 2))

                    # Normalize each faces with the mean and standard deviation from the set of images used to train the resnet
                    for i in range(len(x)):
                        x[i] = self._normalize_X(x[i] / 255.)

                    # Make a prediction (predicted probability of the cropped face being FAKE) for each face, then take the average.
                    with torch.no_grad():
                        y_pred = model(x)
                        y_pred = torch.sigmoid(y_pred.squeeze())
                        return y_pred[:n].mean().item()

            print("no face found")

        except Exception as e:
            print("Prediction error on video %s: %s" % (video_path, str(e)))

        return 0.5
    
    def _predict_on_video_set(self, videos, num_workers, model, face_extractor):
        """
            Launches multiple threads to predict for each video of the testset
            Return a list of predictions (predicted probability of the video being FAKE) for each video
        """
        def process_file(i):
            filename = videos[i]
            y_pred = self._predict_on_video(filename, batch_size=self.frames_per_video, model=model, face_extractor=face_extractor)
            return y_pred

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            predictions = ex.map(process_file, range(len(videos)))

        return list(predictions)

    def _load_face_extractor(self):
        """
            Init and Return the face extractor object (implemented in deepfakes-inference-demo/helpers/face_extract_1) 
            that consists of a video reader function and a facedetector 
        """
        import sys

        sys.path.insert(0, os.path.join(self.root_path,"blazeface-pytorch"))
        sys.path.insert(0, os.path.join(self.root_path,"deepfakes-inference-demo"))

        #Load the face detection model BlazeFace, based on https://github.com/tkat0/PyTorch_BlazeFace/ 
        from blazeface import BlazeFace
        facedet = BlazeFace().to(self.gpu)
        #Load the pretrained weights
        facedet.load_weights(os.path.join(self.root_path,"blazeface-pytorch/blazeface.pth"))
        facedet.load_anchors(os.path.join(self.root_path,"blazeface-pytorch/anchors.npy"))
        #Set the module in evaluation mode
        _ = facedet.train(False)

        from helpers.read_video_1 import VideoReader
        from helpers.face_extract_1 import FaceExtractor

        #set number of frames to be read from the video, taken regulary from the beggining to the end of the video
        self.frames_per_video = 17
        #init video reader
        video_reader = VideoReader()
        #create a lambda function to read the frames where x is the video path
        video_read_fn = lambda x: video_reader.read_frames(x, num_frames=self.frames_per_video)
        #init the face extractor with the video reader function and the facedetector 
        face_extractor = FaceExtractor(video_read_fn, facedet)

        return face_extractor

    def _isotropically_resize_image(self, img, size, resample=cv2.INTER_AREA):
        h, w = img.shape[:2]
        if w > h:
            h = h * size // w
            w = size
        else:
            w = w * size // h
            h = size

        resized = cv2.resize(img, (w, h), interpolation=resample)
        return resized


    def _make_square_image(self, img):
        h, w = img.shape[:2]
        size = max(h, w)
        t = 0
        b = size - h
        l = 0
        r = size - w
        return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

    def _train_on_video(self, video_path, label, batch_size, model, face_extractor):
    
        # Find the faces for N frames in the video.
        print(f"extracting faces from: {video_path}")
        #extract face crops for each frames in the video
        faces = face_extractor.process_video(video_path)
        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            #store the resized faces in x
            x = np.zeros((batch_size, self.input_size, self.input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            # n = counter of faces 
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.                    
                    resized_face = self._isotropically_resize_image(face, self.input_size)
                    resized_face = self._make_square_image(resized_face)

                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
            
            if n > 0:
                # put the array of cropped resized faces x into a tensor
                x = torch.tensor(x, device=self.gpu).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                # Normalize each faces with the mean and standard deviation from the set of images used to train the resnet
                for i in range(len(x)):
                    x[i] = self._normalize_X(x[i] / 255.)
                
                #for data in x:

                #batch_size = x[0].shape[0]
                x = x.to(self.gpu)

                binary_label = 1 if label == "FAKE" else 0
                y_true = torch.empty(batch_size).fill_(binary_label)
                y_true = y_true.to(self.gpu).float()
                
                optimizer.zero_grad()

                y_pred = model(x)
                y_pred = y_pred.squeeze()
                
                loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
                loss.backward()
                optimizer.step()
                
                batch_bce = loss.item()

                return model, batch_bce

        print("no face found")

        return model, 0

if __name__ == "__main__":
    tools.algo.execute(Algo())
