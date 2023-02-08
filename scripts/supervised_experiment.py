from sklearn.model_selection import ParameterGrid

from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS, EXPERIMENT_SEEDS

PROJECT_NAME = WANDB_PROJECTS["experiment"]
SPLIT_BETWEEN_GPUS = True
GPU_NUMBER = 0
RUN_LETTER = "a"

param_grid = {
    "dataset_name": ["rotten_tomatoes"],
    "num_epochs_afresh": [20],
    "train_dataset_size": [32, 64, 128, 256, 512, 1024, 2048, 4096],
    "classifier_base_model": ["gpt2"],
    "use_tapted_classifier": [True, False],
    "supervised": [True],
    "cuda_device": [f"cuda:{GPU_NUMBER}"],
    "eval_every": [1],
    "seed": EXPERIMENT_SEEDS,
}

# The configurations of hyperparameters
combinations = list(ParameterGrid(param_grid))

# If splitting between GPUs, select the appropriate half of the combinations
if SPLIT_BETWEEN_GPUS:
    if GPU_NUMBER == 0:
        combinations = combinations[: len(combinations) // 2]
    else:
        combinations = combinations[len(combinations) // 2 :]

# Run the experiment for each sampled combination of parameters
for counter, combination in enumerate(combinations):
    # Create a unique run_id for this trial
    run_id = f"rt_supervised_{RUN_LETTER}_{GPU_NUMBER}_{counter}"

    # Print the run_id and the Parameters
    print()
    print()
    print("=" * 79)
    title = f"| SUPERVISED EXPERIMENT | Run ID: {run_id}"
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
