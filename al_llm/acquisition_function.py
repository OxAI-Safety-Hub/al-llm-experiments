# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

import random


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
    def select(self, samples: list, num_samples: int = -1) -> list:
        """Apply the acquisition function

        Parameters
        ----------
        samples : list
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

    def _process_num_samples(self, samples: list, num_samples: int = -1) -> int:
        """Determine and validate the number of samples to take

        The value of -1 means that `parameters["num_samples"]` is used.
        """

        if num_samples == -1:
            num_samples = self.parameters["num_samples"]

        if num_samples > len(samples):
            raise ValueError("Size of `samples` is smaller than `num_samples`")

        return num_samples


class DummyAcquisitionFunction(AcquisitionFunction):
    """A dummy acquisition function, which selects the first slice of samples

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    """

    def select(self, samples: list, num_samples: int = -1) -> list:
        """Apply the acquisition function

        Parameters
        ----------
        samples : list
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

        num_samples = self._process_num_samples(samples, num_samples)

        return samples[:num_samples]


class RandomAcquisitionFunction(AcquisitionFunction):
    """An acquisition function which select randomly

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    """

    def select(self, samples: list, num_samples: int = -1) -> list:
        """Apply the acquisition function

        Parameters
        ----------
        samples : list
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

        num_samples = self._process_num_samples(samples, num_samples)

        return random.sample(samples, num_samples)
