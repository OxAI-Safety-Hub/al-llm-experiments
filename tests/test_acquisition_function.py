import random

import wandb

from al_llm.parameters import Parameters
from al_llm.acquisition_function import (
    DummyAF,
    RandomAF,
    MaxUncertaintyAF,
)
from al_llm.classifier import DummyClassifier
from al_llm.dataset_container import DummyDatasetContainer
from al_llm.utils import UnlabelledSamples


wandb_run = wandb.init(project="test", entity="test", mode="disabled")


class LengthUncertaintyClassifier(DummyClassifier):
    """A classifier where the uncertainty is the length of the sentence"""

    def calculate_uncertainties(self, samples: list) -> list:
        return [len(sample) for sample in samples]


def _basic_acquisition_function_test(acquisition_function_cls):

    # The parameters to use for this test
    parameters = Parameters(num_samples=5, sample_pool_size=20)

    # Make the instances
    if acquisition_function_cls == MaxUncertaintyAF:
        dataset_container = DummyDatasetContainer(parameters)
        classifier = DummyClassifier(parameters, dataset_container, wandb_run)
        acquisition_function = acquisition_function_cls(parameters, classifier)
    else:
        acquisition_function = acquisition_function_cls(parameters)

    # Generate some sentences then select them using the acquisition function
    sample_pool = UnlabelledSamples(
        [str(i) for i in range(parameters["sample_pool_size"])]
    )
    samples = acquisition_function.select(sample_pool)

    # Make sure the selection is a sublist
    for sample in samples:
        assert sample in sample_pool

    # Make sure the selection has the correct size
    assert len(samples) == parameters["num_samples"]


def test_random_acquisition_function():
    _basic_acquisition_function_test(RandomAF)


def test_dummy_acquisition_function():
    _basic_acquisition_function_test(DummyAF)


def test_max_uncertainty_function():

    # Do a basic test with this sampler
    _basic_acquisition_function_test(MaxUncertaintyAF)

    # Some basic parameters
    num_samples = 5
    sample_pool_size = 20
    parameters = Parameters(num_samples=num_samples, sample_pool_size=sample_pool_size)

    # Set up the acquisition function
    dataset_container = DummyDatasetContainer(parameters)
    classifier = LengthUncertaintyClassifier(parameters, dataset_container, wandb_run)
    acquisition_function = MaxUncertaintyAF(parameters, classifier)

    # Generate some samples with increasing length
    sample_pool = UnlabelledSamples(["a" * i for i in range(sample_pool_size)])

    # Shuffle these to make it harder
    random.seed(3535)
    random.shuffle(sample_pool)

    # The selected samples should be the longest ones
    target_samples = [
        "a" * i for i in range(sample_pool_size - num_samples, sample_pool_size)
    ]
    assert acquisition_function.select(sample_pool) == target_samples
