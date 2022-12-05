from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random

from sklearn.model_selection import ParameterGrid

from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS, EXPERIMENT_SEEDS

# The random seed to use when shuffling the experiments
SHUFFLE_SEED = 45897

# Set up the arg parser
parser = ArgumentParser(
    description="Run pool-based experiments",
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
    "--no-shuffle-experiments",
    dest="shuffle",
    action="store_false",
    default=False,
    help="Don't shuffle the experiment order",
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
parser.add_argument(
    "--tag",
    type=str,
    default="pool-based",
    help="The tag to  W&B project to use",
)

# Get the arguments
cmd_args = parser.parse_args()

# The different hyperparameters to test
param_grid = {
    "dataset_name": ["rotten_tomatoes", "wiki_toxic", "pubmed_20k_rct", "trec6"],
    "num_iterations": [15],
    "batch_size": [2],
    "eval_batch_size": [2],
    "num_samples": [10],
    "sample_pool_size": [256],  # Should never be less than 'num_samples'
    "train_dataset_size": [32, 256, 1024],
    "classifier_base_model": ["gpt2"],
    "acquisition_function": ["random", "max_uncertainty"],
    "sample_generator_base_model": ["pool"],
    "use_tapted_classifier": [True],
    "seed": EXPERIMENT_SEEDS,
    "cuda_device": [f"cuda:{cmd_args.gpu_num}"],
    "save_classifier_every": [-1],
}

# An iterator over the configurations of hyperparameters
param_iter = ParameterGrid(param_grid)

# Shuffle the experiments if required
if cmd_args.shuffle:
    random.seed(SHUFFLE_SEED)
    param_iter = list(param_iter)
    random.shuffle(param_iter)

# Enumerate these to keep track of them
combinations = enumerate(param_iter)

# Filter to combos
combinations = filter(
    lambda x: x[0] % cmd_args.combo_groups == cmd_args.combo_num, combinations
)
combinations = list(combinations)[cmd_args.num_skip :]

# Keep track of the results of the runs
run_results = []
for combo_num in range(len(combinations)):
    run_results.append("NOT RUN")

try:

    # Run the experiment for each sampled combination of parameters
    for i, (combo_index, combo) in enumerate(combinations):

        # Set the status of the current run to failed until proven otherwise
        run_results[i] = "FAILED"

        # Create a unique run_id for this trial
        run_id = f"pool_based_{cmd_args.run_letter}_{combo_index}"

        # Print the run_id and the Parameters
        print()
        print()
        print("=" * 79)
        title = f"| POOL-BASED EXPERIMENT | Run ID: {run_id}"
        title += (" " * (78 - len(title))) + "|"
        print(title)
        print("=" * 79)
        print()
        print()

        # Set up the parameters for the experiment
        parameters = Parameters(**combo)

        # Make the experiment and run it
        tags = [cmd_args.tag] if cmd_args.tag != "" else []
        args = Experiment.make_experiment(
            parameters=parameters,
            run_id=run_id,
            project_name=cmd_args.project_name,
            tags=tags,
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
    for result, (combo_num, combo) in zip(run_results, combinations):
        print()
        print(f"COMBO {combo_num}")
        print(combo)
        print(result)
