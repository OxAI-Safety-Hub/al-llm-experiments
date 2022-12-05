# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

import random

from al_llm.classifier import Classifier, UncertaintyMixin
from al_llm.parameters import Parameters
from al_llm.utils import UnlabelledSamples


class AcquisitionFunction(ABC):
    """Base acquisition function

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    @abstractmethod
    def select(
        self, sample_pool: UnlabelledSamples, num_samples: int = -1
    ) -> UnlabelledSamples:
        """Apply the acquisition function

        Parameters
        ----------
        sample_pool : UnlabelledSamples
            The list of sentences from which to sample
        num_samples : int, default=-1
            The number of samples to select. The default value of -1 means
            that `parameters["num_samples"]` is used.

        Returns
        -------
        selected_samples : UnlabelledSamples
            A sublist of `samples` of size `num_samples` selected according
            the to acquisition function.
        """
        pass

    def _get_validated_num_samples(
        self, sample_pool: UnlabelledSamples, num_samples: int = -1
    ) -> int:
        """Determine and validate the number of samples to take

        The value of -1 means that `parameters["num_samples"]` is used.
        """

        if num_samples == -1:
            num_samples = self.parameters["num_samples"]

        if num_samples > len(sample_pool):
            raise ValueError(
                f"The size of `sample_pool` must be at least that of `num_samples`"
                f" (currently {num_samples} > {len(sample_pool)}"
            )

        return num_samples


class DummyAF(AcquisitionFunction):
    """A dummy acquisition function, which selects the first slice of samples

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    """

    def select(
        self, sample_pool: UnlabelledSamples, num_samples: int = -1
    ) -> UnlabelledSamples:
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)
        return sample_pool[:num_samples]


class RandomAF(AcquisitionFunction):
    """An acquisition function which selects randomly

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    """

    def select(
        self, sample_pool: UnlabelledSamples, num_samples: int = -1
    ) -> UnlabelledSamples:

        # Get the indices of the random samples
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)
        random_indices = random.sample(range(len(sample_pool)), num_samples)

        # Select these samples by indexing using the list
        random_samples = sample_pool[random_indices]

        # Return these as an `UnlabelledSamples` object
        return random_samples


class MaxUncertaintyAF(AcquisitionFunction):
    """An acquisition function which selects for the highest uncertainty

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: Parameters, classifier: Classifier):
        super().__init__(parameters)
        if not isinstance(classifier, UncertaintyMixin):
            raise TypeError("`classifier` must implement uncertainty measuring")
        self.classifier = classifier

    def select(
        self, sample_pool: UnlabelledSamples, num_samples: int = -1
    ) -> UnlabelledSamples:

        # Process and validate `num_samples`
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)

        # Compute the uncertainty values of each of the samples
        uncertainties = self.classifier.calculate_uncertainties(sample_pool)

        # Compute the index array which would sort these in ascending order
        argsorted = sorted(range(len(sample_pool)), key=lambda i: uncertainties[i])

        # Take the values of `samples` at the last `num_samples` indices
        uncertain_samples = sample_pool[argsorted[-num_samples:]]

        return uncertain_samples
