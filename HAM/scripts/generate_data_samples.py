import os
import shutil
import numpy as np
import pandas as pd

from glob import glob
from sklearn.model_selection import KFold, train_test_split

root_path = os.path.dirname(__file__)
asset_de_path = os.path.join(root_path, '../HAM10000_DE/assets/')
asset_ds_path = os.path.join(root_path, '../HAM10000_DS/assets/')

# load dataset
data_path = './data'
source = pd.read_csv(os.path.join(data_path,'HAM10000_metadata.csv'))
metadata = source.copy()

print(metadata['dx'].value_counts())

num_classes = len(metadata['dx'].unique())
metadata['dx_idx'] = pd.Categorical(metadata['dx']).codes

image_path = os.path.join(data_path, 'images')
all_images_path = glob(os.path.join(image_path, '*.jpg'))

# Retrieve file name from image_id
get_path = {os.path.splitext(os.path.basename(x))[0]: x for x in all_images_path}
metadata['image_path'] = metadata['image_id'].map(get_path)

## Exclude duplicated images from validation test

# Find duplicates
df_train = metadata.copy()
df_unduplicate = df_train.copy()
df_unduplicate = df_unduplicate.groupby(by='lesion_id').count()
df_unduplicate = df_unduplicate[df_unduplicate['image_id'] == 1].reset_index()

df_train['duplicated'] = 'duplicated'
for el in df_unduplicate['lesion_id']:
    df_train.loc[df_train['lesion_id'] == el, 'duplicated'] = 'unduplicated'

df_duplicate = df_train[df_train['duplicated'] == 'duplicated']
df_unduplicate = df_train[df_train['duplicated'] == 'unduplicated']

df_train, df_val = train_test_split(df_unduplicate, test_size=0.2, 
                                    #stratify=df_unduplicate['dx']
                                    )

df_train = pd.concat([df_duplicate, df_train])
df_train.reset_index()
df_val = df_val.copy().reset_index()

####################
#      TRAIN       #
####################
print('train')
N_TRAIN_DATA_SAMPLES = int(np.round(df_train.shape[0]/10.))
train_data_sample_content = []
kf = KFold(n_splits=N_TRAIN_DATA_SAMPLES, shuffle=True)
splits = kf.split(df_train)
for _, index in splits:
    train_data_sample_content.append(df_train.iloc[index])

# #save data samples for data engineer
print(asset_de_path)
train_data_path = os.path.join(os.path.abspath(asset_de_path), 'data')
if os.path.isdir(train_data_path):
    shutil.rmtree(train_data_path)
for i, data_sample in enumerate(train_data_sample_content):
    folder_name = os.path.join(train_data_path, f'data_sample_{i}')
    docker_name = os.path.join('/sandbox/data', f'data_sample_{i}')
    os.makedirs(folder_name)
    for el in data_sample.iterrows():
        shutil.copy(el[1]['image_path'], folder_name)
        df_train.loc[df_train['image_id']==el[1]['image_id'], 'image_path'] = os.path.join(
            el[1]['image_id'] + '.jpg')
        df_train.loc[df_train['image_id']==el[1]['image_id'], 'folder'] = folder_name.split('/')[-1]
os.makedirs(os.path.join(train_data_path, f'csv'))
df_train.to_csv(os.path.join(train_data_path, 'csv/data_sample.csv'))

# ###################
# #      TEST       #
# ###################
print('test')
N_TEST_DATA_SAMPLES = int(np.round(df_val.shape[0]/10.))
test_data_sample_content = []
kf = KFold(n_splits=N_TEST_DATA_SAMPLES, shuffle=True)
splits = kf.split(df_val)
for _, index in splits:
    test_data_sample_content.append(df_val.iloc[index])

# #save data samples for data scientist (testset)
test_data_path = os.path.join(os.path.abspath(asset_ds_path), 'data')
if os.path.isdir(test_data_path):
    shutil.rmtree(test_data_path)
for i, data_sample in enumerate(test_data_sample_content):
    folder_name = os.path.join(test_data_path, f'data_sample_{i}')
    docker_name = os.path.join('/sandbox/data', f'data_sample_{i}')
    os.makedirs(folder_name)
    for el in data_sample.iterrows():
        shutil.copy(el[1]['image_path'], folder_name)
        df_val.loc[df_val['image_id']==el[1]['image_id'], 'image_path'] = os.path.join(
            # docker_name, 
            # folder_name,
            el[1]['image_id'] + '.jpg')
        df_val.loc[df_val['image_id']==el[1]['image_id'], 'folder'] = folder_name.split('/')[-1]
os.makedirs(os.path.join(test_data_path, f'csv'))
df_val.to_csv(os.path.join(test_data_path, 'csv/data_sample.csv'))

