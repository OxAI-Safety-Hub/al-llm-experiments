# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from random import randrange, sample
from typing import Optional, Any

import torch

import wandb

from transformers import pipeline

from al_llm.acquisition_function import AcquisitionFunction
from al_llm.dataset_container import DatasetContainer
from al_llm.parameters import Parameters
from al_llm.utils.artifacts import load_tapted_model
from al_llm.constants import TEXT_COLUMN_NAME


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

        print()
        print("Generating samples...")

        if self.acquisition_function is None:

            # With no acquisition function, just generate samples without
            # filtering
            return self._generate_sample_pool(self.parameters["num_samples"])

        else:

            # With an acquisition function, first generate the samples
            sample_pool = self._generate_sample_pool(
                self.parameters["sample_pool_size"]
            )

            # Then select through the acquisition function
            print()
            print("Selecting using the acquisition function...")
            filtered = self.acquisition_function.select(sample_pool)

            return filtered

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


class PipelineGeneratorMixin(ABC):
    """A mixin for sample generators which use the pipeline() for generation."""

    def _make_pipeline_generator(self, task: str, model: Any, tokenizer: str, **kwargs):
        """Created the generator using pipeline()

        Parameters
        ----------
        task: str
            The pipeline task (e.g. "text-generation")
        model: Any
            The name or reference to the model the pipeline should use.
        tokenizer: str
            The name of the tokenizer the pipeline should use.
        """

        # Set the device to use
        if torch.cuda.is_available():
            device = torch.device(self.parameters["cuda_device"])
        else:
            device = torch.device("cpu")

        # Create a pipeline for text generation
        self.generator = pipeline(
            task=task,
            model=model,
            device=device,
            tokenizer=tokenizer,
            **kwargs,
        )

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

        # Use the pipeline to generate real sentences
        sentence_dicts = self.generator(
            "",
            max_length=self.max_length,
            num_return_sequences=pool_size,
        )
        sample_pool = [d["generated_text"] for d in sentence_dicts]

        return sample_pool


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


class PlainGPT2SampleGenerator(PipelineGeneratorMixin, SampleGenerator):
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

        # Setup the pipeline generator
        self._make_pipeline_generator(
            "text-generation",
            self.MODEL_NAME,
            self.MODEL_NAME,
            temperature=parameters["sample_generator_temperature"],
            top_k=parameters["sample_generator_top_k"],
        )


class PoolSampleGenerator(SampleGenerator):
    """Generate samples by sampling from the remainder dataset.

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
        text_column_name = TEXT_COLUMN_NAME
        self.remainder_sentences = remainder_python[text_column_name]

    def generate(self) -> list:

        # Filter the acquisition function through the set of sentences in the
        # simulated pool taken from the remainder dataset
        sample_pool = self._generate_sample_pool()
        print()
        print("Selecting samples using the acquisition function...")
        return self.acquisition_function.select(sample_pool)

    def _generate_sample_pool(self) -> list:

        # Take `sample_pool_size` random samples from `remainder_sentences`, or
        # as many as you can take up to the length of `remainder_sentences`
        simulated_pool = sample(
            self.remainder_sentences,
            min(len(self.remainder_sentences), self.parameters["sample_pool_size"]),
        )
        return simulated_pool


class TAPTSampleGenerator(PipelineGeneratorMixin, SampleGenerator, ABC):
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

        # Loads the pretrained model from wandb
        self._load_tapted_model()

        # Setup the pipeline generator
        self._make_pipeline_generator(
            "text-generation",
            self.model,
            self.MODEL_NAME,
            temperature=parameters["sample_generator_temperature"],
            top_k=parameters["sample_generator_top_k"],
        )

    def _load_tapted_model(self):
        """Loads the pretrained model from wandb

        Sets `self.model` to this pretrained model after loading
        """

        # load model and training args from wandb
        model, training_args = load_tapted_model(
            self.wandb_run,
            self.MODEL_NAME,
            self.parameters["dataset_name"],
            "sample_generator",
        )
        self.model = model
        self.training_parameters = training_args

    def get_training_parameters(self) -> dict:
        """Get the parameters used for training this model

        Returns
        -------
        self.training_parameters : dict
            The TAPT training parameters
        """
        return self.training_parameters


class TAPTDistilGPT2SampleGenerator(TAPTSampleGenerator):
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
