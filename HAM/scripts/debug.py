import json
import logging
import os
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace

from tqdm import tqdm

import substra

default_stream_handler = logging.StreamHandler()
substra_logger = logging.getLogger('substra')
substra_logger.addHandler(default_stream_handler)

client = substra.Client(debug=True)

@contextmanager
def progress_bar(length):
    """Provide progress bar for for loops"""
    pg = tqdm(total=length)
    progress_handler = logging.StreamHandler(SimpleNamespace(write=lambda x: pg.write(x, end='')))
    substra_logger.removeHandler(default_stream_handler)
    substra_logger.addHandler(progress_handler)
    try:
        yield pg
    finally:
        pg.close()
        substra_logger.removeHandler(progress_handler)
        substra_logger.addHandler(default_stream_handler)


current_directory = os.path.dirname(__file__)
assets_directory = os.path.join(current_directory, '../data_owner/assets')

########################################################
#       Add train dataset
########################################################

DATASET = {
    'name': 'ham_train',
    'type': 'image',
    'data_opener': os.path.join(assets_directory, 'dataset/opener.py'),
    'description': os.path.join(assets_directory, 'dataset/description.md'),
    'permissions': {
        'public': False,
        'authorized_ids': [],
    },
}

print('Adding dataset...')
dataset_key = client.add_dataset(DATASET)
assert dataset_key, 'Missing data manager key'

TRAIN_DATA_SAMPLES_PATHS = [
    os.path.join(assets_directory, 'data', path)
    for path in os.listdir(os.path.join(assets_directory, 'data'))]


print('Adding dataset train...')
train_data_sample_keys = []

print(TRAIN_DATA_SAMPLES_PATHS)

for path in TRAIN_DATA_SAMPLES_PATHS:
    print(path)
    data_sample_key = client.add_data_sample({
        'data_manager_keys': [dataset_key],
        'test_only': False,
        'path': path,
    }, local=True)
    train_data_sample_keys.append(data_sample_key)

print('Associating data samples with dataset...')
client.link_dataset_with_data_samples(
    dataset_key,
    train_data_sample_keys
)

########################################################
#       Add test dataset & objective
########################################################

assets_directory = os.path.join(current_directory, '../data_scientist/assets')

DATASET = {
    'name': 'ham_test',
    'type': 'image',
    'data_opener': os.path.join(assets_directory, 'dataset/opener.py'),
    'description': os.path.join(assets_directory, 'dataset/description.md'),
    'permissions': {
        'public': False,
        'authorized_ids': [],
    },
}

# Add dataset
print('Adding dataset...')
dataset_key_test = client.add_dataset(DATASET)
assert dataset_key_test, 'Missing data manager key'


TEST_DATA_SAMPLES_PATHS = [
    os.path.join(assets_directory, 'data', path)
    for path in os.listdir(os.path.join(assets_directory, 'data'))]


# Add test sample
print('Adding dataset test...')
test_data_sample_keys = []

for path in TEST_DATA_SAMPLES_PATHS:
    print(path)
    data_sample_key = client.add_data_sample({
        'data_manager_keys': [dataset_key_test],
        'test_only': True,
        'path': path,
    }, local=True)

    test_data_sample_keys.append(data_sample_key)

print('Associating data samples with dataset...')
client.link_dataset_with_data_samples(
    dataset_key,
    test_data_sample_keys,
)

# Create objective
OBJECTIVE ={
    'name': 'Detecting skin cancer / HAM10000',
    'description': os.path.join(assets_directory, 'objective/description.md'),
    'metrics_name': 'Accuracy',
    'metrics': os.path.join(assets_directory, 'objective/metrics.zip'),
    'permissions': {
        'public': False,
        'authorized_ids': [],
    },
}

METRICS_DOCKERFILE_FILES = [
    os.path.join(assets_directory, 'objective/metrics.py'),
    os.path.join(assets_directory, 'objective/Dockerfile')
]

# Create objective archive
archive_path = OBJECTIVE['metrics']
with zipfile.ZipFile(archive_path, 'w') as z:
    for filepath in METRICS_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))

# Add objective
print('Adding objective ...')
objective_key = client.add_objective({
    'name': OBJECTIVE['name'],
    'description': OBJECTIVE['description'],
    'metrics_name': OBJECTIVE['metrics_name'],
    'metrics': OBJECTIVE['metrics'],
    'test_data_sample_keys': test_data_sample_keys,
    'test_data_manager_key': dataset_key_test,
    'permissions': OBJECTIVE['permissions'],
})
assert objective_key, 'Missing objective key'


########################################################
#       Add algo
########################################################

ALGO = {
    'name': 'HAM10000 resnet 2',
    'description': os.path.join(assets_directory, 'algo/description.md'),
    'permissions': {
        'public': False,
        'authorized_ids': [],
    }
}

ALGO_DOCKERFILE_FILES = [
    os.path.join(assets_directory, 'algo/algo.py'),
    os.path.join(assets_directory, 'algo/Dockerfile'),
    os.path.join(assets_directory, 'algo/resnet50-19c8e357.pth'),
]

# Build archive
archive_path = os.path.join(current_directory, 'algo_resnet.zip')
with zipfile.ZipFile(archive_path, 'w') as z:
    for filepath in ALGO_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))
ALGO['file'] = archive_path


# Add algo
print('Adding algo...')
algo_key = client.add_algo({
    'name': ALGO['name'],
    'file': ALGO['file'],
    'description': ALGO['description'],
    'permissions': ALGO['permissions'],
})

# Add traintuple
print('Registering traintuple...')
traintuple_key = client.add_traintuple({
    'algo_key': algo_key,
    'data_manager_key': dataset_key,
    'train_data_sample_keys': train_data_sample_keys,
})
assert traintuple_key, 'Missing traintuple key'


# Add testtuple
print('Adding testtuple...')
testtuple_key = client.add_testtuple({
    'objective_key': objective_key,
    'traintuple_key': traintuple_key,
})
assert testtuple_key, 'Missing testtuple key'




