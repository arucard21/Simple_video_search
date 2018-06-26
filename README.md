# Simple Video Search
A simple system that allows searching for videos from the Youtube-8M dataset based on their similarity to a provided video.

## Preparation
In order to make the system work, 2 additional directories have to be added to the root of the application:

* `dataset`: This should contain the Youtube-8M dataset (see Minimal Dataset section below)
* `logistic-model`: This should contain the training model. This can be generated using the Youtube-8M code (using `train.py` and `eval.py`)

These directories could not be added to the repo because they are simply too large.

### LSH Forest
In order to provide the search results quickly, we can use LSH (we use a variant called LSH Forest) to hash the dataset beforehand and quickly go through it when needed. Since this only needs to be done once for the dataset, and it's quite resource-intensive to do, we store these hashes using the Python `pickle` module. Of course we load this stored data when the application starts up. 

This data should be stored in the `lsh_forest.pkl` file in the root of the application. The repo already contains this file but it only has hashes for a fraction of the data available in the Youtube-8M dataset (only the data from the `train00.tfrecord` and `validate00.tfrecord` files). You can update this file with more hashes by running `python similarity.py`. This may take a long time though, depending on your available resources. You can also just overwrite this file with another one containing more hashes, if you already have one available. This new file will be loaded next time the application is started.

In order to use this functionality, you need to add query parameter `useForest=True` to the URL, e.g. `http://localhost:5000/?useForest=True`.

### Minimal Dataset
The system allows iterating through the TFRecord files in the dataset though due to performance issues it currently only looks through the `train00.tfrecord` and `validate00.tfrecord` files. This means it can only look through a fraction of the dataset to provide results using this method. This can be modified to look through more files. Of course the additional records would have to be made available in the `dataset` directory.

Alternatively, the system can use the hashes from `lsh_forest.pkl`, in which case it doesn't need to use the TFRecords from the dataset at all. Here, the system actually provides search results from the entire dataset, meaning all `train*.tfrecord` and `validate*.tfrecord` files. 

However, for inferring the labels for each new video, the system still needs all `validate*.tfrecord` files. So a minimal dataset for this system would need to consist of all `validate*.tfrecord` files and the `train00.tfrecord` file.

## Usage
In order to start the system, just run `python webapp.py`. 

This will load the hashes (might take some time depending on the size of the file) and start the web application which you can access at `http://localhost:5000` (or `http://localhost:5000/?useForest=True` to use the loaded hashes)
