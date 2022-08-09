from al_llm.acquisition_function import (
    DummyAcquisitionFunction,
    RandomAcquisitionFunction,
)
from al_llm.sample_generator import DummySampleGenerator


def _basic_acquisition_function_test(acquisition_function_cls):

    # The parameters to use for this test
    parameters = {"num_samples": 5, "num_oversamples": 20}

    # Make the instances
    acquisition_function = acquisition_function_cls(parameters)

    # Generate some sentences then select them using the acquisition function
    oversamples = [str(i) for i in range(parameters["num_oversamples"])]
    samples = acquisition_function.select(oversamples)

    # Make sure the selection is a sublist
    for sample in samples:
        assert sample in oversamples

    # Make sure the selection has the correct size
    assert len(samples) == parameters["num_samples"]


def test_random_acquisition_function():
    _basic_acquisition_function_test(RandomAcquisitionFunction)


def test_dummy_acquisition_function():
    _basic_acquisition_function_test(DummyAcquisitionFunction)
