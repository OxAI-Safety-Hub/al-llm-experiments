from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS

parameters = Parameters()
dummy_args = Experiment.make_experiment(parameters, WANDB_PROJECTS["sandbox"], "dummy")
experiment = Experiment(**dummy_args)
experiment.run()
