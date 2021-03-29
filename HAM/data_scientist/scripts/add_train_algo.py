import json
import os
import zipfile

import substra

current_directory = os.path.dirname(__file__)
assets_directory = os.path.join(current_directory, '../assets')

client = substra.Client.from_config_file(profile_name="node-1")

ALGO_KEYS_JSON_FILENAME = 'algo_keys.json'
TEST_KEYS_JSON_FILENAME = 'test_assets_keys.json'

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

# Load keys of train dataset
dataset_list = client.list_dataset()
for dataset in dataset_list:
    if dataset.name == 'ham_train':
        dataset_key = dataset.key

train_data_sample_keys = []
train_data_samples = []
for train_data_sample in client.list_data_sample():
    if dataset_key in train_data_sample.data_manager_keys:
        train_data_samples.append(train_data_sample)
        train_data_sample_keys.append(train_data_sample.key)

train_assets_keys = {
    'dataset_key': dataset_key,
    'train_data_sample_keys': train_data_sample_keys,
}
print(train_assets_keys)

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
    'data_manager_key': train_assets_keys['dataset_key'],
    'train_data_sample_keys': train_assets_keys['train_data_sample_keys'],
})
assert traintuple_key, 'Missing traintuple key'

# Load keys of test dataset and objective
assets_keys_path =os.path.join(current_directory, '../test_assets_keys.json')
with open(assets_keys_path, 'r') as f:
    test_assets_keys = json.load(f)
print(test_assets_keys)
# Link dataset with objective
#client.link_dataset_with_objective(test_assets_keys['dataset_key'], test_assets_keys['objective_key'])

# Add testtuple
print('Adding testtuple...')
testtuple_key = client.add_testtuple({
    'objective_key': test_assets_keys['objective_key'],
    'traintuple_key': traintuple_key,
})
assert testtuple_key, 'Missing testtuple key'

# Save keys in json
#TODO : assets keys completed with train and test?? 
assets_keys = {}
assets_keys['algo_resnet'] = {
    'algo_key': algo_key,
    'traintuple': traintuple_key,
    'testtuple': testtuple_key,
}
assets_keys_path = os.path.join(current_directory, '../assets_keys.json')
with open(assets_keys_path, 'w') as f:
    json.dump(assets_keys, f, indent=2)

print(f'Assets keys have been saved to {os.path.abspath(assets_keys_path)}')
print('\nRun the following commands to track the status of the tuples:')
print(f'    substra get traintuple {traintuple_key} --profile node-1')
print(f'    substra get testtuple {testtuple_key} --profile node-1')
