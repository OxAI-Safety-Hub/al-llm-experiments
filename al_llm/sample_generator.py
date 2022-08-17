# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from random import randrange
from typing import Optional
import configparser

import torch
import tempfile
import os
import wandb

from transformers import pipeline, AutoModelForCausalLM

from al_llm.acquisition_function import AcquisitionFunction
from al_llm.dataset_container import DatasetContainer
from al_llm.parameters import Parameters


# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


class SampleGenerator(ABC):
    """Base sample generator

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    acquisition_function : AcquisitionFunction, optional
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
    acquisition_function : AcquisitionFunction, optional
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
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    max_length : int, default=30
        The maximum length of the sentences generated.
    """

    MODEL_NAME = "gpt2"

    def __init__(
        self,
        parameters: Parameters,
        acquisition_function: Optional[AcquisitionFunction] = None,
        max_length: int = 30,
    ):

        super().__init__(parameters, acquisition_function)

        self.max_length = max_length

        # Set the device to use
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Create a pipeline for text generation
        self.generator = pipeline(
            task="text-generation", model=self.MODEL_NAME, device=self.device
        )

    def _generate_sample_pool(self, pool_size: int) -> list:

        # Use the pipeline to generate real sentences
        sentence_dicts = self.generator(
            "",
            max_length=self.max_length,
            num_return_sequences=pool_size,
        )
        sample_pool = [d["generated_text"] for d in sentence_dicts]

        return sample_pool


class PoolSampleGenerator(SampleGenerator):
    """Generate samples by sampling from the remainder dataset
    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    acquisition_function : AcquisitionFunction
        The acquisition function to use.
    dataset_container : DatasetContainer
        The dataset container for the experiment, which holds the remainder
        dataset
    """

    def __init__(
        self,
        parameters: Parameters,
        acquisition_function: AcquisitionFunction,
        dataset_container: DatasetContainer,
    ):
        super().__init__(parameters, acquisition_function)

        # Get the list of sentences in the remainder dataset, as a list
        remainder_python = dataset_container.dataset_remainder.with_format(None)
        text_column_name = config["Data Handling"]["TextColumnName"]
        self.remainder_sentences = remainder_python[text_column_name]

    def generate(self) -> list:

        # Filter the acquisition function through the set of sentences in the
        # remainder dataset
        sample_pool = self._generate_sample_pool()
        return self.acquisition_function.select(sample_pool)

    def _generate_sample_pool(self) -> list:
        return self.remainder_sentences


class TAPTSampleGenerator(SampleGenerator, ABC):
    """Base TAPT sample generator

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    max_length : int, default=30
        The maximum length of the sentences generated.
    """

    MODEL_NAME = ""

    def __init__(
        self,
        parameters: Parameters,
        wandb_run: wandb.sdk.wandb_run.Run,
        acquisition_function: Optional[AcquisitionFunction] = None,
        max_length: int = 30,
    ):

        super().__init__(parameters, acquisition_function)

        self.wandb_run = wandb_run
        self.max_length = max_length

        # Set the device to use
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Loads the pretrained model from wandb
        self.load_tapted_model()

        # Create a pipeline for text generation
        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.MODEL_NAME,
            device=self.device,
        )

    def load_tapted_model(self):
        """Loads the pretrained model from wandb

        Sets `self.model` to this pretrained model after loading
        """

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # download the model into this directory from wandb
            artifact_name = self.MODEL_NAME + "---" + self.parameters["dataset_name"]
            artifact_path_components = (
                config["Wandb"]["Entity"],
                config["Wandb"]["Project"],
                artifact_name + ":latest",
            )
            artifact_path = "/".join(artifact_path_components)
            artifact = self.wandb_run.use_artifact(
                artifact_path,
                type=config["TAPT Generator Loading"]["TAPTGeneratorType"],
            )
            artifact.download(tmpdirname)

            # load model from this directory
            file_path = os.path.join(
                tmpdirname, config["TAPT Generator Loading"]["ModelFileName"]
            )
            self.model = AutoModelForCausalLM.from_pretrained(file_path)

    def _generate_sample_pool(self, pool_size: int) -> list:

        # Use the pipeline to generate real sentences
        sentence_dicts = self.generator(
            "",
            max_length=self.max_length,
            num_return_sequences=pool_size,
        )
        sample_pool = [d["generated_text"] for d in sentence_dicts]

        return sample_pool


class TAPTdistilGPT2SampleGenerator(TAPTSampleGenerator):
    """Tapted distilGPT-2 sample generator, which generates real sentences

    It generates `parameters["num_samples"]` samples. If an acquisition
    function is supplied, it first generates `parameters["sample_pool_size"]`
    then uses the function to select from these.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    max_length : int, default=30
        The maximum length of the sentences generated.
    """

    MODEL_NAME = "distilgpt2"


class TAPTGPT2SampleGenerator(TAPTSampleGenerator):
    """Tapted GPT-2 sample generator, which generates real sentences

    It generates `parameters["num_samples"]` samples. If an acquisition
    function is supplied, it first generates `parameters["sample_pool_size"]`
    then uses the function to select from these.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    max_length : int, default=30
        The maximum length of the sentences generated.
    """

    MODEL_NAME = "gpt2"
