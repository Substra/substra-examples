# A Substra example for Deepfakes Detection

*This example is a Substra implementation of a deepfake detector.
The Algo is based on the [inference demo Kaggle notebook](https://www.kaggle.com/humananalog/inference-demo) and use the [DFDC dataset from Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge).
The structure of the example is inspired from [Substra's Titanic Example](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic/)*

## Prerequisites

In order to run this example, you'll need to:

* use Python 3
* have [Docker](https://www.docker.com/) installed
* [install the `substra` cli](https://github.com/SubstraFoundation/substra#install) (supported version: 0.6.0)
* [install the `substratools` library](https://github.com/substrafoundation/substra-tools) (supported version: 0.6.0)
* [pull the `substra-tools` docker images](https://github.com/substrafoundation/substra-tools#pull-from-private-docker-registry)
* have access to a Substra installation ([configure your host to a public node ip](https://doc.substra.ai/getting_started/installation/local_install_skaffold.html#network) or [install Substra on your machine](https://doc.substra.ai/getting_started/installation/local_install_skaffold.html))
* create a substra profile to define the substra network to target, for instance:

```sh
substra config --profile node-1 http://substra-backend.node-1.com
substra login --profile node-1 --username node-1 --password 'p@$swr0d44'
```

* checkout this repository

All commands in this example are run from the `deepfake-detection` folder.

## Data preparation

### Download the data

The first step will be to download the data from the [Kaggle challenge source](https://www.kaggle.com/c/deepfake-detection-challenge/data)

* Sign-up or login to [Kaggle](https://www.kaggle.com/) and accept the [competitions rules](https://www.kaggle.com/c/deepfake-detection-challenge/rules).
* Download the data samples (4Go) manually (`Download All` at the bottom of the [data section](https://www.kaggle.com/c/deepfake-detection-challenge/data)), or install & configure the [Kaggle API](https://github.com/Kaggle/kaggle-api) and execute the following command:

  ```sh
  kaggle competitions download -c deepfake-detection-challenge
  ```

* Extract the zip file and copy-paste the 'train_sample_videos' folder in the data/DFDC folder of the example.

```sh
mkdir -p data/DFDC
unzip deepfake-detection-challenge.zip 'train_sample_videos/*' -d data/DFDC
rm deepfake-detection-challenge.zip
```

### Generate data samples

The second step will be to generate train and test data samples from the [Kaggle challenge source](https://www.kaggle.com/c/deepfake-detection-challenge/data).
To generate the data samples, run:

```sh
pip install -r scripts/requirements.txt
python scripts/generate_data_samples.py
```

This will create two sub-folders in the `assets` folder:

* `train_data_samples` contains train data features (paths of the videos) and labels as numpy array files
* `test_data_samples` contains test data features (paths of the videos) and labels as numpy array files

## Writing the objective and data manager

Both objective and data manager will need a proper markdown description, you can check them out in their respective
folders. Notice that the data manager's description includes a formal description of the data structure.

Notice also that the `metrics.py` and `opener.py` module both rely on classes imported from the `substratools` module.
These classes provide a simple yet rigid structure that will make algorithms pretty easy to write.

## Writing a simple algorithm

You'll find under `assets/algo_inference` an implementation of the `inference` model from the [inference demo Kaggle notebook](https://www.kaggle.com/humananalog/inference-demo). Like the metrics and opener scripts, it relies on a
class imported from `substratools` that greatly simplifies the writing process. You'll notice that it handles not only
the train and predict tasks but also a lot of data preprocessing.

## Testing our assets

### Using asset command line interfaces

You can first test each assets with the `substratools` CLI, by running specific ML tasks in your local Python environment.

#### Training task

```sh
#for a quicker test, you can change --data-samples-path to a specific data sample, (e.g. assets/train_data_samples/data_sample_0)

#train your model with the train_data
python assets/algo_inference/algo.py train \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data_samples \
  --output-model-path assets/model/model \
  --log-path assets/logs/train.log

#predict the labels of train_data with your previously trained model
python assets/algo_inference/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data_samples \
  --output-predictions-path assets/pred-train.csv \
  --models-path assets/model/ \
  --log-path assets/logs/train_predict.log \
  model

#calculate the score of your model on train_data predictions
python assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data_samples \
  --input-predictions-path assets/pred-train.csv \
  --output-perf-path assets/perf-train.json \
  --log-path assets/logs/train_metrics.log
  
 ```

#### Testing task

```sh
#predict the labels of test_data with your previously trained model
python assets/algo_inference/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data_samples \
  --output-predictions-path assets/pred-test.csv \
  --models-path assets/model/ \
  --log-path assets/logs/test_predict.log \
  model

#calculate the score of your model on test_data predictions
python assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data_samples \
  --input-predictions-path assets/pred-test.csv \
  --output-perf-path assets/perf-test.json \
  --log-path assets/logs/test_metrics.log
```

### Using substra cli

Before pushing our assets to the platform, we need to make sure they work well. To do so, we can run them locally in a
Docker container. This way, if the training fails, we can access the logs and debug our code.

To test the assets, we'll use `substra run-local`, passing it paths to our algorithm of course, but also the opener,
the metrics and to the data samples we want to use. It will launch a training task on the train data, a prediction task on the test data and return the accuracy score.

```sh
substra run-local assets/algo_inference \
  --train-opener=assets/dataset/opener.py \
  --test-opener=assets/dataset/opener.py \
  --metrics=assets/objective/ \
  --train-data-samples=assets/train_data_samples \
  --test-data-samples=assets/test_data_samples
```

At the end of this step, you'll find in the newly created `sandbox/model` folder a `model` file that contains your
trained model. There is also a `sandbox/pred_train` folder that contains both the predictions made by the model on
train data and the associated performance.

#### Debugging

It's more than probable that your code won't run perfectly the first time. Since runs happen in dockers, you can't
debug using prints. Instead, you should use the `logging` module from python. All logs can then be consulted at the end
of the run in  `sandbox/model/log_model.log`.

## Adding the assets to substra

### Adding the objective, dataset and data samples to substra

A script has been written that adds objective, data manager and data samples to substra. It uses the `substra` python
sdk to perform actions. It's main goal is to create assets, get their keys and use these keys in the creation of other
assets.

To run it:

```sh
pip install -r scripts/requirements.txt
python scripts/add_dataset_objective.py
```

This script just generated an `assets_keys.json` file in the `deepfake-detection` folder. This file contains the keys of all assets
we've just created and organizes the keys of the train data samples in folds. This file will be used as input when
adding an algorithm so that we can automatically launch all training and testing tasks.

### Adding the algorithm and training it

The script `add_train_algo_inference.py` pushes our simple algo to substra and then uses the `assets_keys.json` file
we just generated to train it against the dataset and objective we previously set up. It will then update the
`assets_keys.json` file with the newly created assets keys (algo, traintuple and testtuple)

To run it:

```sh
python scripts/add_train_algo_inference.py
```

It will end by providing a couple of commands you can use to track the progress of the train and test tuples as well
as the associated scores. Alternatively, you can browse the frontend to look up progress and scores.
