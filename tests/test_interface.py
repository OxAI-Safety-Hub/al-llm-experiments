from al_llm.data_handler import DummyDataHandler
from al_llm.classifier import DummyClassifier
from al_llm.sample_generator import DummySampleGenerator
from al_llm.interface import PoolSimulatorInterface
from al_llm.experiment import Experiment


def test_pool_simulator_interface():

    # Make a dummy experiment using PoolSimulatorInterface
    parameters = {"is_dummy": True}
    categories = {0: "Valid sentence", 1: "Invalid sentence"}
    classifier = DummyClassifier(parameters)
    data_handler = DummyDataHandler(classifier, categories, parameters)
    sample_generator = DummySampleGenerator(parameters)
    interface = PoolSimulatorInterface(categories, data_handler)
    experiment = Experiment(
        data_handler, categories, classifier, sample_generator, interface, parameters
    )

    # Run it, to make sure there are no errors
    experiment.run()

    # Check that the prompt method returns the correct labels
    initial_dataset_slice = data_handler.dataset_train[:10]
    initial_dataset_slice.set_format("pandas")
    samples = list(initial_dataset_slice[:]["text"])
    labels = list(initial_dataset_slice[:]["labels"])
    assert interface.prompt(samples) == labels
