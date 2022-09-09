# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from random import randrange, sample
from typing import Optional, Any

import torch

import wandb

from transformers import pipeline
from transformers import LogitsProcessor, TopKLogitsWarper, LogitsProcessorList

from al_llm.acquisition_function import AcquisitionFunction
from al_llm.dataset_container import DatasetContainer
from al_llm.parameters import Parameters
from al_llm.classifier import HuggingFaceClassifier
from al_llm.utils.artifacts import load_tapted_model
from al_llm.constants import TEXT_COLUMN_NAME


class TopKLogitsProcessor(TopKLogitsWarper, LogitsProcessor):
    """LogitsProcessor that performs top-k filtering

    Restricts to the k highest probability elements.

    Obtained by converting `TopKLogitsWarper` into a `LogitsProcessor`

    Parameters
    ----------
    top_k : int
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering.
    filter_value : float, optional, default=-`float("Inf")`
        All filtered values will be set to this float value.
    min_tokens_to_keep : int, optional, default=1
        Minimum number of tokens that cannot be filtered.
    """

    pass


class UncertaintyLogitsProcessor(LogitsProcessor):
    """A logits processor which incorporates uncertainties from a classifier

    Adds to each probability a weighted uncertainty estimate for the sentence
    generated up to this point together with the corresponding token.

    Given the probability value `p` of a particular token (the output of the
    generative model) and the uncertainty value `u` of the sentence generated
    so far plus this token, the new probability value is:
        p + uncertainty_weighting * u

    Given that we actually work with a logit value `v` and not a probability,
    the actual calculation, producing the new logit value, is:
        logsumexp(v, uncertainty_weighting * u)

    Parameters
    ----------
    classifier : HuggingFaceClassifier
        The classifier for which to compute the uncertainties
    uncertainty_weighting : float
        The weighting to use when adding the uncertainty to the logit value
    filter_value : float, default=`-float("Inf")`
        The value used in the `scores` by previous processors to indicate that
        we shouldn't consider that token.
    """

    def __init__(
        self,
        classifier: HuggingFaceClassifier,
        uncertainty_weighting: float,
        filter_value: float = -float("Inf"),
    ):
        super().__init__()

        self.classifier = classifier
        self.uncertainty_weighting = uncertainty_weighting
        self.filter_value = filter_value

    @torch.no_grad()
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        # The mask for the uncertainties we actually want to compute
        mask = scores != self.filter_value

        # Filter the input IDs by the mask
        filtered_input_ids = input_ids[mask]

        # Compute the uncertainties of the input sequences for the classifier
        filtered_uncertainties = self.classifier.calculate_uncertainties_tokenized(
            filtered_input_ids
        )

        # Add these to the scores, weighting appropriately
        filtered_weighted_uncertainties = (
            self.uncertainty_weighting * filtered_uncertainties
        )
        to_sum = torch.stack([scores, filtered_weighted_uncertainties], dim=0)
        scores[mask] = torch.logsumexp(to_sum, dim=0)

        return scores


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


class TokenByTokenSampleGenerator(SampleGenerator, ABC):
    """Base class to generate a sentence token-by-token to maximise uncertainty

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    classifier : HuggingFaceClassifier
        The classifier used in the current experiment, for which we maximise
        uncertainty.
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    GENERATOR_MODEL_NAME = ""

    def __init__(
        self,
        parameters: Parameters,
        classifier: HuggingFaceClassifier,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        super().__init__(parameters, acquisition_function)

        self.classifier = classifier

    def _generate_sample_pool(self, pool_size: int) -> list:

        # The logits processor which filters out the top k tokens before adding
        # the uncertainties
        pre_top_k_logits_processor = TopKLogitsProcessor(
            self.parameters["tbt_pre_top_k"]
        )

        # The logits processor which adds the uncertainty values
        uncertainty_logits_processor = UncertaintyLogitsProcessor(
            self.classifier, self.parameters["tbt_uncertainty_weighting"]
        )

        # Combine these two into a list of the logits processors
        logits_processor = LogitsProcessorList(
            [pre_top_k_logits_processor, uncertainty_logits_processor]
        )

        # Use the Hugging Face generation utility to generate samples. This
        # does most of the hard work for us in terms of interacting with the
        # model. We use a custom logits processor to add the uncertainty
        # values coming from the classifier
        samples_tokenized = self.classifier.model.generate(
            temperature=self.parameters["sample_generator_temperature"],
            top_k=self.parameters["sample_generator_top_k"],
            logits_processor=logits_processor,
            renormalize_logits=True,
            num_return_sequences=pool_size,
        )

        # Detokenize the samples to produce the final sentences
        samples = self.classifier.tokenizer.batch_decode(
            samples_tokenized, skip_special_tokens=True
        )

        return samples


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
