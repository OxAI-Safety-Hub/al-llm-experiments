# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

from random import randrange


class SampleGenerator(ABC):
    """Base sample generator"""

    @abstractmethod
    def generate(self):
        """Generate new samples for querying

        Returns
        -------
        samples : list
            A list of samples which are to be labelled
        """
        return []


class DummySampleGenerator(SampleGenerator):
    """Dummy sample generator, which generates random stuff"""

    def generate(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz         "
        length = randrange(5, 30)
        sample_nums = [randrange(len(alphabet)) for i in range(length)]
        sample_chars = map(lambda x: alphabet[x], sample_nums)
        sample = "".join(sample_chars)
        return [sample]
