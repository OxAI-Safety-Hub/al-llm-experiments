Guide to Dual Labelling
==========================

One measure we want for our experiments is how agreeable the labelling of that experiment was. To do this, after an experiment has run, we want another human to label all of the added data, and then compare these results.

How to run
-------------

To run the script, you must pass the following parameters:

- `--run-id` : The run id of the experiment you wish to dual label.

- `--score-ambiguities` : Call this flag if the ambiguities should have to match for the labels to be considered consistent.

You can then call the script from the console. For example, this will dual label the data from the experiment with run id "make_example_dataset", and the ambiguities matter.

```
python scripts/labelling/dual_label.py --run-id make_example_dataset --score-ambiguities
```


The results
---------------------------

The new labels, along with the old are stored to weights and biases as a new artifact. The results are also stored. They are saved as a dictionary of the form:


- `num_labels` : The number of labels given in this process.

- `labelling_consistency` : The proportion of sentences that both humans labelled the same. 

Guide to Labelling Training
==========================

Another measure we want is how similar our own labelling is to that of the original labels in the dataset. We also want to train ourselves to get better at this before running experiments. To do this we load a subset of the dataset and compare our own labels against it.

How to run
-------------

To run the script, you must pass the following parameters:

- `--dataset-name` : The name of the dataset you wish to train yourself on.

- `--num-labels` : The size of the subset to take from the dataset.

- `--seed` : The seed to use for shuffling the dataset to create the subset to label. ***Default=42***

You can then call the script from the console. For example, this will make a subset of size 10 from the rotten_tomatoes dataset for you to label.

```
python scripts/labelling/label_training.py --dataset-name rotten_tomatoes --num-labels 10
```


The results
---------------------------

The results of this process are logged to weights and biases in the 'Labelling-Training' project. The metrics recorded are:

- `consistency` : The proportion of sentences that you labelled correctly.

