import json
import os
import zipfile

import substra

current_directory = os.path.dirname(__file__)
assets_directory = os.path.join(current_directory, '../assets')
algo_directory = os.path.join(assets_directory, 'algo_inference')

client = substra.Client(profile_name="node-1")

ALGO_KEYS_JSON_FILENAME = 'algo_inference_keys.json'

ALGO = {
    'name': 'deepfake-detection: inference',
    'description': os.path.join(algo_directory, 'description.md'),
    'permissions': {
        'public': False,
        'authorized_ids': []
    },
}
ALGO_DOCKERFILE_FILES = [
        os.path.join('algo.py'),
        os.path.join('Dockerfile'),
        os.path.join('blazeface-pytorch/anchors.npy'),
        os.path.join('blazeface-pytorch/blazeface.pth'),
        os.path.join('blazeface-pytorch/blazeface.py'),
        os.path.join('deepfakes-inference-demo/helpers/face_extract_1.py'),
        os.path.join('deepfakes-inference-demo/helpers/read_video_1.py'),
        os.path.join('deepfakes-inference-demo/resnext50_32x4d-7cdf4587.pth'),
]

########################################################
#       Build archive
########################################################

archive_path = os.path.join(current_directory, 'algo_inference.zip')
with zipfile.ZipFile(archive_path, 'w') as z:
    for filepath in ALGO_DOCKERFILE_FILES:
        z.write(os.path.join(algo_directory, filepath), arcname=filepath)
ALGO['file'] = archive_path

########################################################
#       Load keys for dataset and objective
########################################################

assets_keys_path = os.path.join(current_directory, '../assets_keys.json')
with open(assets_keys_path, 'r') as f:
    assets_keys = json.load(f)

########################################################
#         Add algo
########################################################

print('Adding algo...')
algo_key = client.add_algo({
    'name': ALGO['name'],
    'file': ALGO['file'],
    'description': ALGO['description'],
    'permissions': ALGO['permissions'],
}, exist_ok=True)['pkhash']

########################################################
#         Add traintuple
########################################################

print('Registering traintuple...')
traintuple = client.add_traintuple({
    'algo_key': algo_key,
    'data_manager_key': assets_keys['dataset_key'],
    'train_data_sample_keys': assets_keys['train_data_sample_keys']
}, exist_ok=True)
traintuple_key = traintuple.get('key') or traintuple.get('pkhash')
assert traintuple_key, 'Missing traintuple key'

########################################################
#         Add testtuple
########################################################
print('Registering testtuple...')
testtuple = client.add_testtuple({
    'objective_key': assets_keys['objective_key'],
    'traintuple_key': traintuple_key
}, exist_ok=True)
testtuple_key = testtuple.get('key') or testtuple.get('pkhash')
assert testtuple_key, 'Missing testtuple key'

########################################################
#         Save keys in json
########################################################

assets_keys['algo_inference'] = {
    'algo_key': algo_key,
    'traintuple_key': traintuple_key,
    'testtuple_key': testtuple_key,
}
with open(assets_keys_path, 'w') as f:
    json.dump(assets_keys, f, indent=2)

print(f'Assets keys have been saved to {os.path.abspath(assets_keys_path)}')
print('\nRun the following commands to track the status of the tuples:')
print(f'    substra get traintuple {traintuple_key} --profile node-1')
print(f'    substra get testtuple {testtuple_key} --profile node-1')