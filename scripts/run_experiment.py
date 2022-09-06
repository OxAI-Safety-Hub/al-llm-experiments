from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS

# Default W&B run identification
PROJECT_NAME = WANDB_PROJECTS["experiment"]

# Override the defaults in the `Parameters` class
DEFAULTS = dict(
    dataset_name="rotten_tomatoes",
    classifier_base_model="gpt2",
    sample_generator_base_model="gpt2",
)

# Set up the arg parser
parser = ArgumentParser(
    description="Run an AL LLM experiment",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

# Add arguments for the W&B run
parser.add_argument("run-id", type=str, help="The W&B run ID to use")
parser.add_argument(
    "--project-name", type=str, default=PROJECT_NAME, help="The W&B project to use."
)

# Add the parameters to the parser
Parameters.add_to_arg_parser(parser, DEFAULTS)

# Get the arguments
cmd_args = parser.parse_args()

# Get a Parameters object from these
parameters = Parameters.from_argparse_namespace(cmd_args)

# Make the experiment
args = Experiment.make_experiment(
    parameters=parameters,
    run_id=cmd_args.run_id,
    project_name=cmd_args.project_name,
)
experiment = Experiment(**args)

# Run it
experiment.run()
