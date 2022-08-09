import random

from al_llm.acquisition_function import (
    DummyAcquisitionFunction,
    RandomAcquisitionFunction,
    MaxUncertaintyAcquisitionFunction,
)
from al_llm.classifier import DummyClassifier


class LengthUncertaintyClassifier(DummyClassifier):
    """A classifier where the uncertainty is the length of the sentence"""

    def calculate_uncertainties(self, samples: list) -> list:
        return [len(sample) for sample in samples]


def _basic_acquisition_function_test(acquisition_function_cls):

    # The parameters to use for this test
    parameters = {"num_samples": 5, "num_oversamples": 20}

    # Make the instances
    if acquisition_function_cls == MaxUncertaintyAcquisitionFunction:
        classifier = DummyClassifier(parameters)
        acquisition_function = acquisition_function_cls(parameters, classifier)
    else:
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


def test_max_uncertainty_function():

    # Do a basic test with this sampler
    _basic_acquisition_function_test(MaxUncertaintyAcquisitionFunction)

    # Some basic parameters
    num_samples = 5
    num_oversamples = 20
    parameters = {"num_samples": num_samples, "num_oversamples": num_oversamples}

    # Set up the acquisition function
    classifier = LengthUncertaintyClassifier(parameters)
    acquisition_function = MaxUncertaintyAcquisitionFunction(parameters, classifier)

    # Generate some samples with increasing length
    samples = ["a" * i for i in range(num_oversamples)]

    # Shuffle these to make it harder
    random.seed(3535)
    random.shuffle(samples)

    # The selected samples should be the longest ones
    target_samples = [
        "a" * i for i in range(num_oversamples - num_samples, num_oversamples)
    ]
    assert acquisition_function.select(samples) == target_samples
