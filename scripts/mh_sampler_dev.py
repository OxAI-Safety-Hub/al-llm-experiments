from argparse import ArgumentParser

from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS

# Set up the arg parser
parser = ArgumentParser(
    description="Test the MH sampler",
)

# Add arguments for the W&B run
parser.add_argument("run_num", type=str, help="The number to use for the W&B run")

# Add MMH generation parameters to the parser
Parameters.add_to_arg_parser(
    parser, included_parameters=["mmh_num_steps", "mmh_mask_probability"]
)

# Get the arguments
cmd_args = parser.parse_args()

# Set up the parameters
parameters = Parameters(
    dataset_name="rotten_tomatoes",
    sample_generator_base_model="bert",
    classifier_base_model="gpt2",
    use_tapted_classifier=True,
    num_epochs_afresh=1,
    use_mmh_sample_generator=True,
)

# Make the experiment
args = Experiment.make_experiment(
    parameters,
    WANDB_PROJECTS["sandbox"],
    f"mh_sampler_test_{cmd_args.run_num}",
)
experiment = Experiment(**args)
