Guide to TAPTing a sample generator
==========================

For training, we are using a helper script called [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py). This script can be run via the command prompt but requires many parameters, so to simplify the process, we will instead only need to run our own script `tapt_trainer.py` which uses this helper script.

Awkwardly, since this process requires different install versions of torch and transformers, it uses it's own virtual environment.

How to run
-------------

In AWS, load `tmux`, and then navigate to this directory

```
cd scripts/tapt/
```

Then run the script, specifying the parameters. Below is an example of this which trains the gpt2 model on the rotten_tomatoes dataset with a batch size of 4, over the course of 8 epochs.

```
python tapt_trainer --model-name gpt2 --dataset-name rotten_tomatoes --batch-size 4 --num-epochs 8
```

The parameters
---------------------------

There are various parameters you can define in line after the call to the function as seen above. These are:


- `--model-name` : The name of the huggingface transformer to use as a base model.

- `--dataset-name` : The name or path of the dataset to train on.

- `--batch-size` : The batch size for training and evaluation. ***Default=4***

- `--num-epochs` : The number of epochs to use in the train loop. ***Default=3***

- `--seed` : The seed to use for this process. ***Default=327532***
