from al_llm.parameters import Parameters
from al_llm.experiment import Experiment
from al_llm.constants import WANDB_PROJECTS
from al_llm.utils import UnlabelledSamples


def test_pool_simulator_interface():

    # Make a dummy experiment using PoolSimulatorInterface
    parameters = Parameters(
        sample_generator_base_model="pool", full_loop=True, is_running_pytests=True
    )
    args = Experiment.make_experiment(parameters, WANDB_PROJECTS["sandbox"], "test")
    experiment = Experiment(**args)

    # Run it, to make sure there are no errors
    experiment.run()

    # Check that the prompt method returns the correct labels
    dataset_train = args["dataset_container"].dataset_remainder
    initial_dataset_slice = dataset_train.with_format(None)
    samples = UnlabelledSamples(initial_dataset_slice["text"][:10])
    samples.suggested_labels = initial_dataset_slice["labels"][:10]
    prompt_labels, _ = args["interface"].prompt(samples)
    assert prompt_labels == samples.suggested_labels
