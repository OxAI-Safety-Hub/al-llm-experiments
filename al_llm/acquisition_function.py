# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

import random

from al_llm.classifier import Classifier, UncertaintyMixin


class AcquisitionFunction(ABC):
    """Base acquisition function

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters

    @abstractmethod
    def select(self, sample_pool: list, num_samples: int = -1) -> list:
        """Apply the acquisition function

        Parameters
        ----------
        sample_pool : list
            The list of sentences from which to sample
        num_samples : int, default=-1
            The number of samples to select. The default value of -1 means
            that `parameters["num_samples"]` is used.

        Returns
        -------
        selected_samples : list
            A sublist of `samples` of size `num_samples` selected according
            the to acquisition function.
        """
        pass

    def _get_validated_num_samples(
        self, sample_pool: list, num_samples: int = -1
    ) -> int:
        """Determine and validate the number of samples to take

        The value of -1 means that `parameters["num_samples"]` is used.
        """

        if num_samples == -1:
            num_samples = self.parameters["num_samples"]

        if num_samples > len(sample_pool):
            raise ValueError("Size of `samples` is smaller than `num_samples`")

        return num_samples


class DummyAcquisitionFunction(AcquisitionFunction):
    """A dummy acquisition function, which selects the first slice of samples

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    """

    def select(self, sample_pool: list, num_samples: int = -1) -> list:
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)
        return sample_pool[:num_samples]


class RandomAcquisitionFunction(AcquisitionFunction):
    """An acquisition function which selects randomly

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    """

    def select(self, sample_pool: list, num_samples: int = -1) -> list:
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)
        return random.sample(sample_pool, num_samples)


class MaxUncertaintyAcquisitionFunction(AcquisitionFunction):
    """An acquisition function which selects for the highest uncertainty

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: dict, classifier: Classifier):
        super().__init__(parameters)
        if not isinstance(classifier, UncertaintyMixin):
            raise TypeError("`classifier` must implement uncertainty measuring")
        self.classifier = classifier

    def select(self, sample_pool: list, num_samples: int = -1) -> list:

        # Process and validate `num_samples`
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)

        # Compute the uncertainty values of each of the samples
        uncertainties = self.classifier.calculate_uncertainties(sample_pool)

        # Compute the index array which would sort these in ascending order
        argsorted = sorted(range(len(sample_pool)), key=lambda i: uncertainties[i])

        # Take the values of `samples` at the last `num_samples` indices
        uncertain_samples = [sample_pool[i] for i in argsorted[-num_samples:]]

        return uncertain_samples
