from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS

# Override the defaults in the `Parameters` class
DEFAULTS = dict(
    dataset_name="rotten_tomatoes",
    sample_generator_base_model="bert",
    classifier_base_model="gpt2",
    use_tapted_classifier=True,
    num_epochs_afresh=1,
    use_mmh_sample_generator=True,
    mmh_num_steps=1,
    num_samples=10,
)

# Set up the arg parser
parser = ArgumentParser(
    description="Test the MH sampler",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

# Add arguments for the W&B run
parser.add_argument("run_num", type=str, help="The number to use for the W&B run")

# Add MMH generation parameters to the parser
Parameters.add_to_arg_parser(
    parser,
    included_parameters=["mmh_num_steps", "mmh_mask_probability", "num_samples"],
    defaults=DEFAULTS,
)

# Get the arguments
cmd_args = parser.parse_args()

# Set up the parameters
parameters = Parameters.from_argparse_namespace(cmd_args, defaults=DEFAULTS)

# Make the experiment
args = Experiment.make_experiment(
    parameters,
    WANDB_PROJECTS["sandbox"],
    f"mh_sampler_test_{cmd_args.run_num}",
)
experiment = Experiment(**args)

# Generate some samples
samples = experiment.sample_generator._generate_sample_pool(parameters["num_samples"])
