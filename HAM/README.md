# A Substra example for Skin cancer detection

*This example is a Substra implementation of a skin cancer detector.
The Algo is based on the [notebook using a resnet50](https://www.kaggle.com/xinruizhuang/skin-lesion-classification-acc-90-pytorch) and use the [HAM10000 dataset from Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000).
The structure of the example is inspired from [Substra's Titanic Example](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic/). The main difference is that train and test datasets are not owned by the same identity.*

## Prerequisites

In order to run this example, you'll need to:

* use Python 3
* have [Docker](https://www.docker.com/) installed
* [install the `substra` cli](https://github.com/SubstraFoundation/substra#install) (supported version: 0.6.0)
* [install the `substratools` library](https://github.com/substrafoundation/substra-tools) (supported version: 0.6.0)*
* have access to a Substra installation ([configure your host to a public node ip](https://doc.substra.ai/getting_started/installation/local_install_skaffold.html#network) or [install Substra on your machine](https://doc.substra.ai/getting_started/installation/local_install_skaffold.html)). Check that this installation is [compatible](https://github.com/SubstraFoundation/substra#compatibility-table) with your CLI version.
* create a substra profile to define the substra network to target, for instance:

```sh
substra config --profile node-1 http://substra-backend.node-1.com
substra login --profile node-1 --username node-1 --password 'p@$swr0d44'
```

* checkout this repository

All commands in this example are run from the `HAM10000` folder.

## Data preparation

### Download the data

The first step will be to download the data from the [Kaggle challenge source](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

* Sign-up or login to [Kaggle](https://www.kaggle.com/).
* Download the data samples (4Go) manually (`Download All` at the bottom of the [data section](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)), or install & configure the [Kaggle API](https://github.com/Kaggle/kaggle-api) and execute the following command:

```sh
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
```

* Extract the zip file in the `data` folder of the example.

```sh
mkdir -p data/images
unzip skin-cancer-mnist-ham10000.zip 'HAM10000_images_part_*' -d data/images
unzip skin-cancer-mnist-ham10000.zip 'HAM10000_metadata.csv' -d data
cp -r data/images/HAM10000_images_part_1/ data/images/ && rm -r data/images/HAM10000_images_part_1
cp -r data/images/HAM10000_images_part_2/ data/images/ && rm -r data/images/HAM10000_images_part_2
rm skin-cancer-mnist-ham10000.zip
```

### Generate data samples

The second step will be to generate train and test data samples.
To generate the data samples, run:

```sh
pip install -r scripts/requirements.txt
python scripts/generate_data_samples.py
```

This will create one sub-folders `data` in the `data_onwer/assets` for train data and in `data_scientist/assets` for test data. Each `data` forder contains a folder called `csv` where labels and images path are stored and `data_samples` folders where images are saved.

## Writing the objective and data manager

Both objective and data manager will need a proper markdown description, you can check them out in their respective
folders. Notice that the data manager's description includes a formal description of the data structure.

Notice also that the `metrics.py` and `opener.py` module both rely on classes imported from the `substratools` module.
These classes provide a simple yet rigid structure that will make algorithms pretty easy to write.

## Writing a simple algorithm

You'll find under `data_scientist/assets/algo` an implementation of the `detection` model. Like the metrics and opener scripts, it relies on a
class imported from `substratools` that greatly simplifies the writing process. You'll notice that it handles not only
the train and predict tasks but also a lot of data preprocessing.

This example uses transfer learning. You need to download the pretrained weights of the model [here](https://download.pytorch.org/models/resnet50-19c8e357.pth) and save it in the algo assets folder. 

## Testing our assets

### Debug script
You can run the script `scripts/debug.py` to test your assets. The algorithm will run on a docker, you should have docker installed and launched on your computer. 

```sh
python scripts/debug.py
```

## Adding the assets to substra

### Adding the train dataset from the data owner perspective

A script has been written that adds data manager and data samples to substra. It uses the `substra` python
sdk to perform actions. It's main goal is to create assets, get their keys and use these keys in the creation of other
assets.

To run it:

```sh
pip install -r scripts/requirements.txt
python data_owner/scripts/add_dataset.py
```

This script just generated an `assets_keys.json` file in the `data_owner` folder. This file contains the keys of all assets
we've just created and organizes the keys of the train data samples in folds. This file will be used as input when
adding an algorithm so that we can automatically launch all training and testing tasks.

### Adding the test dataset and objective from the data scientist perspective

A script has been written that adds data manager and data samples to substra. It uses the `substra` python
sdk to perform actions. It's main goal is to create assets, get their keys and use these keys in the creation of other
assets.

To run it:

```sh
python data_scientist/scripts/add_dataset_objective.py
```

This script just generated an `test_assets_keys.json` file in the `data_scientist` folder. This file contains the keys of all assets
we've just created and organizes the keys of the train data samples in folds. This file will be used as input when
adding an algorithm so that we can automatically launch all training and testing tasks.

### Adding the algorithm and training it

The script `add_train_algo.py` pushes our simple algo to substra and then uses the `assets_keys.json` file
we just generated to train it against the dataset and objective we previously set up. It will then update the
`assets_keys.json` file with the newly created assets keys (algo, traintuple and testtuple)

To run it:

```sh
python data_scientist/scripts/add_train_algo.py
```

:warning: Be careful, in the `add_train_algo` script, the keys of the train data samples are founded with the name of the dataset. If you reupload the train dataset, you must change the name of the dataset and in the `add_train_algo` script too. 

It will end by providing a couple of commands you can use to track the progress of the train and test tuples as well
as the associated scores. Alternatively, you can browse the frontend to look up progress and scores.





