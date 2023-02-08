import warnings

from sklearn.model_selection import ParameterGrid

from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS, EXPERIMENT_SEEDS

GPU_NUMBER = 1
SPLIT_BETWEEN_GPUS = True
RUN_LETTER = "p"
PROJECT_NAME = WANDB_PROJECTS["hyperparameter_tuning"]

# The different hyperparameters to test
param_grid = {
    "dataset_name": ["rotten_tomatoes"],
    "num_iterations": [200],
    "refresh_every": [100],
    "batch_size": [16],
    "eval_batch_size": [128],
    "num_epochs_update": [3],
    "num_epochs_afresh": [10],
    "num_samples": [10],
    "sample_pool_size": [500000],  # Should never be less than 'num_samples'
    "train_dataset_size": [10],
    "classifier_base_model": ["gpt2"],
    "acquisition_function": ["max_uncertainty"],
    "sample_generator_base_model": ["pool"],
    "use_tapted_classifier": [True],
    "seed": EXPERIMENT_SEEDS,
    "cuda_device": [f"cuda:{GPU_NUMBER}"],
    "save_classifier_every": [-1],
}

# An interator over the configurations of hyperparameters
param_iter = ParameterGrid(param_grid)

# If splitting between GPUs, select the appropriate half of the combinations
combinations = list(param_iter)
if SPLIT_BETWEEN_GPUS:
    if GPU_NUMBER == 0:
        combinations = combinations[: len(combinations) // 2]
    else:
        combinations = combinations[len(combinations) // 2 :]

# Run the experiment for each sampled combination of parameters
for counter, combination in enumerate(combinations):
    # Ensure the code won't break due to human error in parameter grid
    #   inputs. Better to skip an experiment than crash the whole program
    if combination["num_samples"] > combination["sample_pool_size"]:
        warnings.warn(
            "Failed to run one of the hyperparameter tuning tests (num_samples > sample_pool_size)."
        )
        continue

    # Create a unique run_id for this trial
    run_id = f"hpt_rt_{RUN_LETTER}_{GPU_NUMBER}_{counter}"

    # Print the run_id and the Parameters
    print()
    print()
    print("=" * 79)
    title = f"| HYPERPARAMETER EXPERIMENT | Run ID: {run_id}"
    title += (" " * (78 - len(title))) + "|"
    print(title)
    print("=" * 79)
    print()
    print()

    # Set up the parameters for the experiment
    parameters = Parameters(**combination)

    # Make the experiment and run it
    args = Experiment.make_experiment(
        parameters=parameters,
        run_id=run_id,
        project_name=PROJECT_NAME,
    )
    experiment = Experiment(**args)
    experiment.run()
