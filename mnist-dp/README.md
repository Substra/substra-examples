# Mnist with Differential Privacy

*This example is a Substra implementation of on the [Classification_Privacy tutorial](https://github.com/tensorflow/privacy/blob/master/tutorials/Classification_Privacy.ipynb) from [Tensorflow_Privacy](https://github.com/tensorflow/privacy). The structure of this example is inspired from [Substra's Titanic Example](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic/)*

> [Differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) (DP) is a framework for measuring the privacy guarantees provided by an algorithm. Through the lens of differential privacy, we can design machine learning algorithms that responsibly train models on private data. Learning with differential privacy provides provable guarantees of privacy, mitigating the risk of exposing sensitive training data in machine learning. Intuitively, a model trained with differential privacy should not be affected by any single training example, or small set of training examples, in its data set. This mitigates the risk of exposing sensitive training data in ML.
>  
> The basic idea of this approach, called differentially private stochastic gradient descent (DP-SGD), is to modify the gradients used in stochastic gradient descent (SGD), which lies at the core of almost all deep learning algorithms. Models trained with DP-SGD provide provable differential privacy guarantees for their input data. There are two modifications made to the vanilla SGD algorithm:
>  
> 1. First, the sensitivity of each gradient needs to be bounded. In other words, we need to limit how much each individual training point sampled in a minibatch can influence gradient computations and the resulting updates applied to model parameters. This can be done by *clipping* each gradient computed on each training point.
> 2. *Random noise* is sampled and added to the clipped gradients to make it statistically impossible to know whether or not a particular data point was included in the training dataset by comparing the updates SGD applies when it operates with or without this particular data point in the training dataset.
>  
> This tutorial uses [tf.keras](https://www.tensorflow.org/guide/keras) to train a convolutional neural network (CNN) to recognize handwritten digits with the DP-SGD optimizer provided by the TensorFlow Privacy library. TensorFlow Privacy provides code that wraps an existing TensorFlow optimizer to create a variant that implements DP-SGD.  
> &mdash; [The TensorFlow Authors][1]

[1]: https://github.com/tensorflow/privacy/blob/master/tutorials/Classification_Privacy.ipynb

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

All commands in this example are run from the `mnist` folder.

## Data preparation

The first step will be to generate train and test data from keras.datasets.mnist

To generate the data, run:

```sh
pip install --upgrade pip
pip install -r scripts/requirements.txt
python scripts/generate_data.py
```

This will create two sub-folders in the `assets` folder:

* `train_data` contains train data features and labels as numpy array files
* `test_data` contains test data features and labels as numpy array files

## Writing the objective and data manager

Both objective and data manager will need a proper markdown description, you can check them out in their respective
folders. Notice that the data manager's description includes a formal description of the data structure.

Notice also that the `metrics.py` and `opener.py` module both rely on classes imported from the `substratools` module.
These classes provide a simple yet rigid structure that will make algorithms pretty easy to write.

## Writing a simple algorithm

You'll find under `assets/algo_cnn_dp` an implementation of the cnn model in the [Classification_Privacy tutorial](https://github.com/tensorflow/privacy/blob/master/tutorials/Classification_Privacy.ipynb). Like the metrics and opener scripts, it relies on a
class imported from `substratools` that greatly simplifies the writing process. You'll notice that it handles not only
the train and predict tasks but also a lot of data preprocessing.

This algorithm measure the differential privacy guarantee after training the model:
You will see in the console the value Epsilon (ϵ) - This is the privacy budget. It measures the strength of the privacy guarantee by bounding how much the probability of a particular model output can vary by including (or excluding) a single training point. A smaller value for ϵ implies a better privacy guarantee. However, the ϵ value is only an upper bound and a large value could still mean good privacy in practice.

This value depends on:

1. The total number of points in the training data, `n`.
2. The `batch_size`.
3. The `noise_multiplier`.
4. The number of `epochs` of training.

## Testing our assets

### Using asset command line interfaces

You can first test each assets with the `substratools` CLI, by running specific ML tasks in your local Python environment.

#### Training task

```sh
# train your model with the train_data
python assets/algo_cnn_dp/algo.py train \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data \
  --output-model-path assets/model/model \
  --log-path assets/logs/train.log

# predict the labels of train_data with your previously trained model
python assets/algo_cnn_dp/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data \
  --output-predictions-path assets/pred-train.npy \
  --models-path assets/model/ \
  --log-path assets/logs/train_predict.log \
  model

# calculate the score of your model on train_data predictions
python assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data \
  --input-predictions-path assets/pred-train.npy \
  --output-perf-path assets/perf-train.json \
  --log-path assets/logs/train_metrics.log
```

#### Testing task

```sh
# predict the labels of test_data with your previously trained model
python assets/algo_cnn_dp/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data \
  --output-predictions-path assets/pred-test.npy \
  --models-path assets/model/ \
  --log-path assets/logs/test_predict.log \
  model

# calculate the score of your model on test_data predictions
python assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data \
  --input-predictions-path assets/pred-test.npy \
  --output-perf-path assets/perf-test.json \
  --log-path assets/logs/test_metrics.log
```

### Using substra cli

Before pushing our assets to the platform, we need to make sure they work well. To do so, we can run them locally in a
Docker container. This way, if the training fails, we can access the logs and debug our code.

To test the assets, we'll use `substra run-local`, passing it paths to our algorithm of course, but also the opener,
the metrics and to the data samples we want to use. It will launch a training task on the train data, a prediction task on the test data and return the accuracy score.

```sh
substra run-local assets/algo_cnn_dp \
  --train-opener=assets/dataset/opener.py \
  --test-opener=assets/dataset/opener.py \
  --metrics=assets/objective/ \
  --train-data-samples=assets/train_data \
  --test-data-samples=assets/test_data
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

This script just generated an `assets_keys.json` file in the `mnist` folder. This file contains the keys of all assets
we've just created and organizes the keys of the train data samples in folds. This file will be used as input when
adding an algorithm so that we can automatically launch all training and testing tasks.

### Adding the algorithm and training it

The script `add_train_algo_cnn_dp.py` pushes our simple algo to substra and then uses the `assets_keys.json` file
we just generated to train it against the dataset and objective we previously set up. It will then update the
`assets_keys.json` file with the newly created assets keys (algo, traintuple and testtuple)

To run it:

```sh
python scripts/add_train_algo_cnn_dp.py
```

It will end by providing a couple of commands you can use to track the progress of the train and test tuples as well
as the associated scores. Alternatively, you can browse the frontend to look up progress and scores.

