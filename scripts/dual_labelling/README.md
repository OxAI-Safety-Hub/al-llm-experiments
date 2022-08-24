Guide to Dual Labelling
==========================

One measure we want for our experiments is how agreeable the labelling of that experiment was. To do this, after an experiment has run, we want another human to label all of the added data, and then compare these results.

How to run
-------------

To run the script, you must pass the following parameters:

- `--run-id` : The run id of the experiment you wish to dual label.

- `--score-ambiguities` : Should the ambiguities have to match for the labels to be considered consistent? ***Default = True***

You can then call the script from the console. For example, this will dual label the data from the experiment with run id "make_example_dataset".

```
python scripts/dual_labelling/dual_label.py --run-id make_example_dataset --score-ambiguities True
```


The results
---------------------------

The new labels, along with the old are stored to weights and biases as a new artifact. The results are also stored. They are saved as a dictionary of the form:


- `num_labels` : The number of labels given in this process.

- `labelling_consistency` : The percentage of sentences that both humans labelled the same. 