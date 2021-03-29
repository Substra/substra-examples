import os
import json
import logging
import zipfile

import substra

default_stream_handler = logging.StreamHandler()
substra_logger = logging.getLogger('substra')
substra_logger.addHandler(default_stream_handler)

client = substra.Client.from_config_file(profile_name="node-1")

current_directory = os.path.dirname(__file__)
assets_directory = os.path.join(current_directory, '../assets')

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

# Save assets keys
assets_keys = {
    'dataset_key': dataset_key,
    'train_data_sample_keys': train_data_sample_keys,
}
assets_keys_path = os.path.join(current_directory, '../train_assets_keys.json')
with open(assets_keys_path, 'w') as f:
    json.dump(assets_keys, f, indent=2)

print(f'Assets keys have been saved to {os.path.abspath(assets_keys_path)}')