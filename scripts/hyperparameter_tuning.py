import numpy
import random

from sklearn.model_selection import ParameterGrid

from al_llm import Experiment, Parameters

# The different hyperparameters to test
param_grid = {
    "dataset_name": ["rotten_tomatoes"],
    "num_iterations": [1,5,10],
    "refresh_every": [1,3,10],
    "batch_size": [4],
    "num_epochs_update": [1,3,10],
    "num_epochs_afresh": [1,3,10],
    "num_samples": [50],
    "sample_pool_size": list(numpy.logspace(1,3,3)),
    "train_dataset_size": list(numpy.logspace(1,3,3)),
    "classifier_base_model": ["gpt2"],
    "acquisition_function": ["max_uncertainty"],
    "sample_generator_base_model": ["pool"],
}

# An interator over the configurations of hyperparameters
param_iter = ParameterGrid(param_grid)

# The best set of hyperparameters
best_params = {}
best_eval_loss = 1

# Set the number of trials to run
num_trials = 10

# Get a list of random samples to try from the param_iter
combinations = random.sample(list(param_iter), num_trials)

# Use a counter to differentiate between trials in WandB
counter = 0

# Run the experiment for each sampled combination of parameters
for combination in combinations:

    # Create a unique run_id for this trial
    run_id = f"hparams_tuning_trial_{counter}"

    # Print the run_id and the Parameters
    print(f"Run ID: {run_id}\nParameters: {combination}\n\n")

    # Set up the parameters for the experiment
    parameters = Parameters(
        dataset_name=combination["dataset_name"],
        num_iterations=combination["num_iterations"],
        refresh_every=combination["refresh_every"],
        batch_size=combination["batch_size"],
        num_epochs_update=combination["num_epochs_update"],
        num_epochs_afresh=combination["num_epochs_afresh"],
        num_samples=combination["num_samples"],
        sample_pool_size=combination["sample_pool_size"],
        learning_rate=combination["learning_rate"],
        train_dataset_size=combination["train_dataset_size"],
        classifier_base_model=combination["classifier_base_model"],
        acquisition_function=combination["acquisition_function"],
        sample_generator_base_model=combination["sample_generator_base_model"],
    )

    # Make the experiment and run it
    args = Experiment.make_experiment(parameters=parameters, run_id=run_id)
    experiment = Experiment(**args)
    experiment.run_full()

    # Increment the counter
    counter = counter + 1
