from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sklearn.model_selection import ParameterGrid

from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS, EXPERIMENT_SEEDS

# Set up the arg parser
parser = ArgumentParser(
    description="Run the supervised MMH experiments",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

# Add various arguments
parser.add_argument("run_letter", type=str, help="The letter to use for the W&B run")
parser.add_argument(
    "--combo-groups",
    type=int,
    default=1,
    help="Into how many groups to split the experiment combinations",
)
parser.add_argument(
    "--combo-num",
    type=int,
    default=0,
    help="Which combo group to run this time",
)
parser.add_argument(
    "--num-skip",
    type=int,
    default=0,
    help="The number of initial combos to skip. Useful to resume a group",
)
parser.add_argument(
    "--gpu-num", type=int, default=0, help="The (0-indexed) GPU number to use"
)
parser.add_argument(
    "--project-name",
    type=str,
    default=WANDB_PROJECTS["experiment"],
    help="The W&B project to use",
)

# Get the arguments
cmd_args = parser.parse_args()

# The different hyperparameters to test
param_grid = {
    "dataset_name": ["rotten_tomatoes"],
    "num_iterations": [10],
    "batch_size": [2],
    "eval_batch_size": [2],
    "num_samples": [32],
    "sample_pool_size": [32],  # Should never be less than 'num_samples'
    "train_dataset_size": [32],
    "classifier_base_model": ["gpt2"],
    "acquisition_function": ["none"],
    "sample_generator_base_model": ["bert"],
    "use_tapted_classifier": [False],
    "seed": EXPERIMENT_SEEDS,
    "cuda_device": [f"cuda:{cmd_args.gpu_num}"],
    "use_mmh_sample_generator": [True],
    "use_suggested_labels": [True],
    "save_classifier_every": [-1],
    "mmh_num_steps": [5, 10, 20, 50],
    "mmh_mask_probability": [0, 0.05, 0.1, 0.15],
}

# An interator over the configurations of hyperparameters
param_iter = ParameterGrid(param_grid)

# Enumerate these to keep track of them
combinations = enumerate(param_iter)

# Filter to combos
combinations = filter(
    lambda x: x[0] % cmd_args.combo_groups == cmd_args.combo_num, combinations
)
combinations = list(combinations)[cmd_args.num_skip :]

# Keep track of the results of the runs
run_results = []
for i in range(len(combinations)):
    run_results.append("SKIPPED")

try:

    # Run the experiment for each sampled combination of parameters
    for i, combo in combinations:

        # Set the status of the current run to failed until proven otherwise
        run_results[i] = "FAILED"

        # Create a unique run_id for this trial
        run_id = f"mh_supervised_{cmd_args.run_letter}_{i}"

        # Print the run_id and the Parameters
        print()
        print()
        print("=" * 79)
        title = f"| SUPERVISED MMH EXPERIMENT | Run ID: {run_id}"
        title += (" " * (78 - len(title))) + "|"
        print(title)
        print("=" * 79)
        print()
        print()

        # Set up the parameters for the experiment
        parameters = Parameters(**combo)

        # Make the experiment and run it
        args = Experiment.make_experiment(
            parameters=parameters,
            run_id=run_id,
            project_name=cmd_args.project_name,
        )
        experiment = Experiment(**args)
        experiment.run()

        run_results[i] = "SUCCEEDED"

finally:

    # Print a summary of the experiment results
    print()
    print()
    print("=" * 79)
    title = f"| SUMMARY | GROUP {cmd_args.combo_num}/{cmd_args.combo_groups}"
    title += (" " * (78 - len(title))) + "|"
    print(title)
    print("=" * 79)
    for result, (i, combo) in zip(run_results, combinations):
        print()
        print(f"COMBO {i}")
        print(combo)
        print(result)
