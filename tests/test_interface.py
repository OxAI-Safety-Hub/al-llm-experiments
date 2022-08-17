from al_llm import parameters
from al_llm.parameters import Parameters
from al_llm.experiment import Experiment


def test_pool_simulator_interface():

    # Make a dummy experiment using PoolSimulatorInterface
    parameters = Parameters(sample_generator="PoolSampleGenerator")
    args = Experiment.make_experiment(parameters, "test")
    experiment = Experiment(**args)

    # Run it, to make sure there are no errors
    # experiment.run_full()

    # Check that the prompt method returns the correct labels
    # initial_dataset_slice = data_handler.dataset_train[:10]
    # initial_dataset_slice.set_format("pandas")
    # samples = list(initial_dataset_slice[:]["text"])
    # labels = list(initial_dataset_slice[:]["labels"])
    # assert interface.prompt(samples) == labels
