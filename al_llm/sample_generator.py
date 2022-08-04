# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

from random import randrange

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


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
    """Dummy sample generator, which generates random stuff"""

    def generate(self) -> list:
        alphabet = "abcdefghijklmnopqrstuvwxyz         "
        samples = []
        for sampleIndex in range(self.parameters["num_samples"]):
            length = randrange(5, 30)
            sample_nums = [randrange(len(alphabet)) for i in range(length)]
            sample_chars = map(lambda x: alphabet[x], sample_nums)
            sample = "".join(sample_chars)
            samples.append(sample)
        return samples


class PlainGPT2SampleGenerator(SampleGenerator):
    """Plain GPT-2 sample generator, which just generates real sentences"""

    def __init__(self, parameters: dict, max_length: int = 30):
        super().__init__(parameters)
        self.max_length = max_length
        # Loads the GPT-2 model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def generate(self) -> list:
        # Uses `pipeline` to generate real sentences
        generator = pipeline(
            task="text-generation", model=self.model, tokenizer=self.tokenizer
        )
        sentence_dicts = generator(
            "",
            max_length=self.max_length,
            num_return_sequences=self.parameters["num_samples"],
        )
        sentences = [d["generated_text"] for d in sentence_dicts]
        return sentences
