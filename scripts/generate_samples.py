from al_llm import Experiment, Parameters
from al_llm.constants import WANDB_PROJECTS

TEMP = 1
NUMBER = 1
SEED = 68526

parameters = Parameters(
    dataset_name="rotten_tomatoes",
    sample_generator_base_model="gpt2",
    use_tapted_sample_generator=True,
    sample_generator_temperature=TEMP,
    num_samples=1000,
    acquisition_function="none",
    num_iterations=2,
    num_epochs_update=1,
    cuda_device="cuda:0",
    seed=68526,
)

args = Experiment.make_experiment(
    parameters,
    WANDB_PROJECTS["experiment"],
    f"rt_samples_tapt_temp{int(TEMP*100)}_{NUMBER}_{SEED}",
)
experiment = Experiment(**args)
experiment.run()
