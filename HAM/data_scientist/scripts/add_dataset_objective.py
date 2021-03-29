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
dataset_key = client.add_dataset(DATASET)
assert dataset_key, 'Missing data manager key'


TEST_DATA_SAMPLES_PATHS = [
    os.path.join(assets_directory, 'data', path)
    for path in os.listdir(os.path.join(assets_directory, 'data'))]


# Add test sample
print('Adding dataset test...')
test_data_sample_keys = []

for path in TEST_DATA_SAMPLES_PATHS:
    print(path)
    data_sample_key = client.add_data_sample({
        'data_manager_keys': [dataset_key],
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
    'test_data_manager_key': dataset_key,
    'permissions': OBJECTIVE['permissions'],
})
assert objective_key, 'Missing objective key'

# client.link_dataset_with_objective(dataset_key, objective_key)

# Save assets keys
assets_keys = {
    'dataset_key': dataset_key,
    'objective_key': objective_key,
    'test_data_sample_keys': test_data_sample_keys,
}
assets_keys_path = os.path.join(current_directory, '../test_assets_keys.json')
with open(assets_keys_path, 'w') as f:
    json.dump(assets_keys, f, indent=2)

print(f'Assets keys have been saved to {os.path.abspath(assets_keys_path)}')