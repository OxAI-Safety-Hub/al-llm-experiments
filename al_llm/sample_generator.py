# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

from random import randrange

from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from al_llm.acquisition_function import AcquisitionFunction
from al_llm.parameters import Parameters


class SampleGenerator(ABC):
    """Base sample generator

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    def __init__(
        self,
        parameters: Parameters,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        self.parameters = parameters
        self.acquisition_function = acquisition_function

    def generate(self) -> list:
        """Generate new samples for querying

        Returns
        -------
        samples : list
            A list of samples which are to be labelled of length `num_samples`
            as defined in the experiment parameters
        """

        if self.acquisition_function is None:

            # With no acquisition function, just generate samples without
            # filtering
            return self._generate_sample_pool(self.parameters["num_samples"])

        else:

            # With an acquisition function, generate the samples, then filer
            sample_pool = self._generate_sample_pool(
                self.parameters["sample_pool_size"]
            )
            return self.acquisition_function.select(sample_pool)

    @abstractmethod
    def _generate_sample_pool(self, pool_size: int) -> list:
        """Generate a pool of samples, from which to select

        Parameters
        ----------
        pool_size: int
            The number of samples to generate

        Returns
        -------
        samples : list
            A list of samples
        """
        return []


class DummySampleGenerator(SampleGenerator):
    """Dummy sample generator, which generates random stuff

    It generates `parameters["num_samples"]` samples. If an acquisition
    function is supplied, it first generates `parameters["sample_pool_size"]`
    then uses the function to select from these.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    def _generate_sample_pool(self, pool_size: int) -> list:

        alphabet = "abcdefghijklmnopqrstuvwxyz         "

        # Generate the samples by sampling from the alphabet
        sample_pool = []
        for sample_index in range(pool_size):
            length = randrange(5, 30)
            sample_nums = [randrange(len(alphabet)) for i in range(length)]
            sample_chars = map(lambda x: alphabet[x], sample_nums)
            sample = "".join(sample_chars)
            sample_pool.append(sample)

        return sample_pool


class PlainGPT2SampleGenerator(SampleGenerator):
    """Plain GPT-2 sample generator, which just generates real sentences

    It generates `parameters["num_samples"]` samples. If an acquisition
    function is supplied, it first generates `parameters["sample_pool_size"]`
    then uses the function to select from these.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    max_length : int, default=30
        The maximum length of the sentences generated.
    """

    def __init__(
        self,
        parameters: Parameters,
        acquisition_function: Optional[AcquisitionFunction] = None,
        max_length: int = 30,
    ):

        super().__init__(parameters, acquisition_function)

        self.max_length = max_length

        # Loads the GPT-2 model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def _generate_sample_pool(self, pool_size: int) -> list:

        # Uses `pipeline` to generate real sentences
        generator = pipeline(
            task="text-generation", model=self.model, tokenizer=self.tokenizer
        )
        sentence_dicts = generator(
            "",
            max_length=self.max_length,
            num_return_sequences=pool_size,
        )
        sample_pool = [d["generated_text"] for d in sentence_dicts]

        return sample_pool
