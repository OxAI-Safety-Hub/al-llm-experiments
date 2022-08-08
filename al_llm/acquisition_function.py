# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod


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


class DummyAcquisitionFunction(AcquisitionFunction):
    """A dummy acquisition function, which select the first slice of samples

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
        if num_samples == -1:
            num_samples = self.parameters["num_samples"]

        if num_samples > len(samples):
            raise ValueError("Size of `samples` is smaller than `num_samples`")

        return samples[:num_samples]
