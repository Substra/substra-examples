"""
Script to generate data to be registered to Substra
deepfake-detection example
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import cv2
from shutil import copyfile

def get_meta_from_json(path):
    df = pd.read_json(path)
    df = df.T
    return df

def load_data_DFDC(data_path):
    DFDC_FOLDER = data_path
    print(f"Loading DFDC data from {DFDC_FOLDER}")

    #load data
    DATA_FOLDER = 'train_sample_videos'

    print(f"# of files in data folder: {len(os.listdir(os.path.join(DFDC_FOLDER, DATA_FOLDER)))}")
    
    # check files type

    train_list = list(os.listdir(os.path.join(DFDC_FOLDER, DATA_FOLDER)))
    ext_dict = []
    for file in train_list:
        file_ext = file.split('.')[1]
        if (file_ext not in ext_dict):
            ext_dict.append(file_ext)

    for file_ext in ext_dict:
        print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")   

    #check json file
    json_file = [file for file in train_list if  file.endswith('json')][0]
    print(f"JSON file: {json_file}")

    #load metadata and stored data
    meta_df = get_meta_from_json(os.path.join(DFDC_FOLDER, DATA_FOLDER, json_file))
    meta = np.array(list(meta_df.index))
    meta_labels = np.array(list(meta_df.label))

    storage = np.array([os.path.join(DFDC_FOLDER, DATA_FOLDER, file) for file in train_list if  file.endswith('mp4')])
    #storage = np.array([os.path.join(file) for file in train_list if  file.endswith('mp4')])

    print(f"# of files in metadata: {meta.shape[0]}, # of videos: {storage.shape[0]}")


    ## option 1 (deprecated): put entire files in arrays
    """
    print(f"videos: {storage}")
    
    print(os.path.isfile(os.path.join(DFDC_FOLDER, DATA_FOLDER, storage[0])))
    data = []
    for file in storage: 
        print(file)
        video = read_video(os.path.join(DFDC_FOLDER, DATA_FOLDER, file)) 
        #data.append(video)
        full_path = os.path.join(DFDC_FOLDER,"tmp", file)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        np.save(full_path, video) #uncompressed file -> 1 Go for a video of 2-10 mo in mp4
    """
    ## option 2 : put paths as features in arrays 
    data = storage

    labels = meta_labels 

    print("Spliting data in train/test sets...")
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20)

    print("# of train data points: ", data_train.shape[0])
    print("# of test data points: ", data_test.shape[0])
 
    return (data_train, labels_train), (data_test, labels_test)

file_path = os.path.dirname(__file__)
root_path = os.path.join(file_path, "..")
data_path = os.path.join(root_path, 'data')
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')
assets_path = os.path.join(root_path, "assets")

# the data, split between train and test sets
# (train features, train labels), (test features, test labels)
(x_train, y_train), (x_test, y_test) = load_data_DFDC(os.path.join('data', "DFDC"))

print("Data will be generated in : ", os.path.abspath(assets_path))
# number of data samples for the train and test sets
N_TRAIN_DATA_SAMPLES = 80 #80 => 4 videos/sample
N_TEST_DATA_SAMPLES = 20

train_test_configs = [
    {
        'features': x_train,
        'labels': y_train,
        'n_samples': N_TRAIN_DATA_SAMPLES,
        'data_samples_root': os.path.join(assets_path, 'train_data_samples'),
        'data_samples_content_x': [],
        'data_samples_content_y': [],
    },
    {
        'features': x_test,
        'labels': y_test,
        'n_samples': N_TEST_DATA_SAMPLES,
        'data_samples_root': os.path.join(assets_path, 'test_data_samples'),
        'data_samples_content_x': [],
        'data_samples_content_y': [],
    },
]

# generate data samples
for conf in train_test_configs:
    kf = KFold(n_splits=conf['n_samples'])
    splits_x = kf.split(conf['features'])
    splits_y = kf.split(conf['labels'])

    for _, index in splits_x:
        conf['data_samples_content_x'].append(conf['features'][index])
    for _, index in splits_y:
        conf['data_samples_content_y'].append(conf['labels'][index])

# save data samples
for conf in train_test_configs:
    """
    for i, data_sample in enumerate(conf['data_samples_content_x']):
        filepath = os.path.join(conf['data_samples_root'], f'data_sample_{i}/features/x_data_sample_{i}.npy')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath,data_sample)
    """
    for i, data_sample in enumerate(conf['data_samples_content_x']):
        
        filepath = os.path.join(conf['data_samples_root'], f'data_sample_{i}/features/x_data_sample_{i}_')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        for file in data_sample:
            copyfile(file, str(filepath+os.path.basename(file)))

    for i, data_sample in enumerate(conf['data_samples_content_y']):
        filepath = os.path.join(conf['data_samples_root'], f'data_sample_{i}/labels/y_data_sample_{i}.npy')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath,data_sample)


"""
#function used in option 1
def read_video(path):
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    print(buf.shape)
    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    print("releasing")
    cap.release()
    return buf
"""