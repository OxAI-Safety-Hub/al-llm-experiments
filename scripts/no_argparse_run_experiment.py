from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS, EXPERIMENT_SEEDS

# Default W&B run identification
PROJECT_NAME = WANDB_PROJECTS["sandbox"]
MULTUPLE_SEEDS = False
run_id = str(input(
    "What should the wadnb run ID be? A good convention is: " +
    "{DATASET}_{CLASSIFIER}_{SAMPLE_GENERATOR}_{ACQUISTION_FUNCTION}_{NUMBER}"
))

# Set the seeds to work with
if MULTUPLE_SEEDS:
    seeds = EXPERIMENT_SEEDS
else:
    seeds = [EXPERIMENT_SEEDS[0]]

parameters = Parameters(
    dataset_name="rotten_tomatoes",
    classifier_base_model="gpt2",
    sample_generator_base_model="gpt2",
    acquisition_function="max_uncertainty",
    batch_size=1,
    eval_batch_size=1,
    seed=seeds[0],
    cuda_device="cuda:0"
)

for i, seed in enumerate(seeds):
    run_id = f"{run_id}_{seed}"

    print()
    print()
    print("=" * 79)
    title = f"| EXPERIMENT: {i} | seed: {seed} | Run ID: {run_id} "
    title += (" " * max(0, 78 - len(title))) + "|"
    print(title)
    print("=" * 79)
    print()
    print()

    # Set the seed
    parameters["seed"] = seed

    # Make the experiment
    args = Experiment.make_experiment(
        parameters=parameters,
        run_id=run_id,
        project_name=PROJECT_NAME,
    )
    experiment = Experiment(**args)

    # Run it
    experiment.run()
