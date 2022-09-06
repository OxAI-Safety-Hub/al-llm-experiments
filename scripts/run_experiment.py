from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS

# W&B run identification
RUN_ID = "experiment"
PROJECT_NAME = WANDB_PROJECTS["experiments"]

# Make sure we've specified the RUN_ID
assert RUN_ID != "experiment"

# Experiment parameters
parameters = Parameters(
    dataset_name="rotten_tomatoes",
    classifier_base_model="gpt2",
    sample_generator_base_model="gpt2",
    use_tapted_classifier=False,
    use_tapted_sample_generator=False,
)

# Make the experiment
args = Experiment.make_experiment(
    parameters=parameters,
    run_id=RUN_ID,
    project_name=PROJECT_NAME,
)
experiment = Experiment(**args)

# Run it
experiment.run()
