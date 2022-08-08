Guide to formatting a local dataset
==========================

Since the datasets we download will not be standardised, we need to do the formatting ourselves for any datasets we want to store locally. This guide will explain how to do so.

I have created the folder `dummy_local_dataset/` to serve as an example template.

File structure
-------------

Each dataset will be stored in it's own folder using a descriptive name (e.g. `dummy_local_dataset/`). In this folder, there must be:

- A csv file named `train.csv`. This will store the data for training.

- A csv file named `evaluation.csv`. This will store the data for validation. 

- A csv file named `test.csv`. This will store the data for testing.


Dataset labels
---------------------------

Each csv file must contain exactly ***two*** columns.

The first line in each file must be hold the names of the columns which must be named:

- `text` for the column containing the sentences.
- `label` for the column containing the integers representing the label given.

