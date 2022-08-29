import warnings
import random

from sklearn.model_selection import ParameterGrid

from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS

# The different hyperparameters to test
param_grid = {
    "dataset_name": ["rotten_tomatoes"],
    "num_iterations": [1, 5, 10],
    "refresh_every": [1, 3, 10],
    "batch_size": [4, 8, 16],
    "num_epochs_update": [1, 3, 6],
    "num_epochs_afresh": [1, 3, 6],
    "num_samples": [50],
    "sample_pool_size": [50, 100, 1000],  # Should never be less than 'num_samples'
    "train_dataset_size": [100],
    "classifier_base_model": ["gpt2"],
    "acquisition_function": ["max_uncertainty"],
    "sample_generator_base_model": ["pool"],
}

# An interator over the configurations of hyperparameters
param_iter = ParameterGrid(param_grid)

# Set the number of trials to run
num_trials = 10

# Get a list of random samples to try from the param_iter
combinations = random.sample(list(param_iter), num_trials)

# Use a counter to differentiate between trials in WandB
counter = 0

# Also define a horizontal line for neater printing
horizontal = "=" * 70 + "\n"

# Run the experiment for each sampled combination of parameters
for counter, combination in enumerate(combinations):

    # Ensure the code won't break due to human error in parameter grid
    #   inputs. Better to skip an experiment than crash the whole program
    if combination["num_samples"] <= combination["sample_pool_size"]:

        # Create a unique run_id for this trial
        run_id = f"hparams_tuning_trial_{counter}"

        # Print the run_id and the Parameters
        text = "\n"
        text += horizontal
        text += f"Run ID: {run_id}\n\n"
        text += f"Parameters: {combination}\n"
        text += horizontal
        print(text)

        # Set up the parameters for the experiment
        parameters = Parameters(**combination)

        # Make the experiment and run it
        args = Experiment.make_experiment(parameters=parameters, run_id=run_id, project_name=WANDB_PROJECTS["sandbox"])
        experiment = Experiment(**args)
        experiment.run()

    # If the test needs to be skipped, log a warning to the user
    else:
        warnings.warn(
            "Failed to run one of the hyperparameter tuning tests (num_samples > sample_pool_size)."
        )
