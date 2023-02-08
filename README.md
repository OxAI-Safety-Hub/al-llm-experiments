# Active learning with a generative large language model

## Installation

- Clone the repository:
```
git clone git@github.com:OxAI-Safety-Hub/al-llm-experiments.git
```
- Optional (but recommended): create a new virtual environment.
- Install the requirements:
```
pip install -r requirements.txt
```


## Running experiments

- The main script for running experiments is [`scripts/run_experiment.py`](/scripts/run_experiment.py).
- First install the `al_llm` package locally in editable mode:
```
pip install -e .
```
- Pass the `--help` flag to see the list of options:
```
python /scripts/run_experiment.py --help
```
- The first argument is the run ID. A good convention for run ID is:
```
{DATASET}_{CLASSIFIER}_{SAMPLE_GENERATOR}_{ACQUISTION_FUNCTION}_{NUMBER}
```
using abbreviations. For example, the second pool-based experiment with the Rotten Tomatoes dataset, which uses a plain classifier and max uncertainty acquisition function might be called:
```
rt_plain_pool_mu_2
```
If there are special features you can add them at the end, before the number.
- By default the experiment runs in the 'Experiments' project. To change this specify the `--project-name` option.
- To select the GPU, use something like `--cuda-device 'cuda:1'`. The default is to use the 0th device.
- By default, we run a single experiment for one seed. To run multiple of the same experiment over different seeds, add the `--multiple-seeds` flag.
- In terms of configuring the experiment parameters, you'll most likely want to play around with the following options:
```
--dataset-name
--classifier-base-model
--use-tapted-classifier
--sample-generator-base-model
--use-tapted-sample-generator
--sample-generator-temperature
--sample-generator-top-k
--acquisition-function
```
But other options may be interesting.


## Documentation

The following guides are located in the [docs](/docs) folder.
- [Guide to contributing code](/docs/contributing-code.md).
- [Instructions on setting up Docker](/docs/using-docker.md).
- [Instructions on using with Colab](/docs/colab.md).
- [Guide to adding a new dataset](/docs/new-dataset.md).
