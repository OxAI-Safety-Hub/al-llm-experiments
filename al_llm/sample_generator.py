# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

from random import randrange

from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from acquisition_function import AcquisitionFunction


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
            A list of samples which are to be labelled of length `num_samples`
            as defined in the experiment parameters
        """
        return []


class DummySampleGenerator(SampleGenerator):
    """Dummy sample generator, which generates random stuff

    It generates `parameters["num_samples"]` samples. If an acquisition
    function is supplied, it first generates `parameters["num_oversamples"]`
    then uses the function to select from these.

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    def __init__(
        self,
        parameters: dict,
        acquisition_function: Optional[AcquisitionFunction],
    ):
        super().__init__(parameters)
        self.acquisition_function = acquisition_function

    def generate(self) -> list:

        alphabet = "abcdefghijklmnopqrstuvwxyz         "

        # The number of sentences to generate first
        if self.acquisition_function is None:
            num_sentences_first = self.parameters["num_samples"]
        else:
            num_sentences_first = self.parameters["num_oversamples"]

        # Generate the samples by sampling from the alphabet
        samples = []
        for sampleIndex in range(num_sentences_first):
            length = randrange(5, 30)
            sample_nums = [randrange(len(alphabet)) for i in range(length)]
            sample_chars = map(lambda x: alphabet[x], sample_nums)
            sample = "".join(sample_chars)
            samples.append(sample)

        # Select from these, if using an acquisition function
        if self.acquisition_function is not None:
            samples = self.acquisition_function.select(samples)

        return samples


class PlainGPT2SampleGenerator(SampleGenerator):
    """Plain GPT-2 sample generator, which just generates real sentences

    It generates `parameters["num_samples"]` samples. If an acquisition
    function is supplied, it first generates `parameters["num_oversamples"]`
    then uses the function to select from these.

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    max_length : int, default=30
        The maximum length of the sentences generated.
    """

    def __init__(
        self,
        parameters: dict,
        acquisition_function: Optional[AcquisitionFunction],
        max_length: int = 30,
    ):

        super().__init__(parameters)

        self.max_length = max_length
        self.acquisition_function = acquisition_function

        # Loads the GPT-2 model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def generate(self) -> list:
        """Use GTP-2 to generate new samples for querying

        Returns
        -------
        samples : list
            A list of samples which are to be labelled of length `num_samples`
            as defined in the experiment parameters
        """

        # The number of sentences to generate first
        if self.acquisition_function is None:
            num_sentences_first = self.parameters["num_samples"]
        else:
            num_sentences_first = self.parameters["num_oversamples"]

        # Uses `pipeline` to generate real sentences
        generator = pipeline(
            task="text-generation", model=self.model, tokenizer=self.tokenizer
        )
        sentence_dicts = generator(
            "",
            max_length=self.max_length,
            num_return_sequences=num_sentences_first,
        )
        samples = [d["generated_text"] for d in sentence_dicts]

        # Select from these, if using an acquisition function
        if self.acquisition_function is not None:
            samples = self.acquisition_function.select(samples)

        return samples
