from al_llm.parameters import Parameters
from al_llm.experiment import Experiment


def test_pool_simulator_interface():

    # Make a dummy experiment using PoolSimulatorInterface
    parameters = Parameters(sample_generator="PoolSampleGenerator")
    args = Experiment.make_experiment(parameters, "test", is_running_pytests=True)
    experiment = Experiment(**args)

    # Run it, to make sure there are no errors
    experiment.run_full()

    # Check that the prompt method returns the correct labels
    dataset_train = args["dataset_container"].dataset_remainder
    initial_dataset_slice = dataset_train.with_format(None)
    samples = initial_dataset_slice["text"][:10]
    labels = initial_dataset_slice["labels"][:10]
    assert args["interface"].prompt(samples) == labels