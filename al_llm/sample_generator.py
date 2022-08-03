# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

from random import randrange


class SampleGenerator(ABC):
    """Base sample generator

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters

    @abstractmethod
    def generate(self) -> list:
        """Generate new samples for querying

        Returns
        -------
        samples : list
            A list of samples which are to be labelled
        """
        return []


class DummySampleGenerator(SampleGenerator):
    """Dummy sample generator, which generates random stuff"""

    def generate(self) -> list:
        alphabet = "abcdefghijklmnopqrstuvwxyz         "
        length = randrange(5, 30)
        sample_nums = [randrange(len(alphabet)) for i in range(length)]
        sample_chars = map(lambda x: alphabet[x], sample_nums)
        sample = "".join(sample_chars)
        return [sample]


class PlainBertSampleGenerator(SampleGenerator):
    """Plain bert sample generator, which just generates real sentences"""

    def generate(self) -> list:
        alphabet = "abcdefghijklmnopqrstuvwxyz         "
        length = randrange(5, 30)
        sample_nums = [randrange(len(alphabet)) for i in range(length)]
        sample_chars = map(lambda x: alphabet[x], sample_nums)
        sample = "".join(sample_chars)
        return [sample]
