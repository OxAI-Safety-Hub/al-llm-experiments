# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from random import randrange, sample
from typing import Optional, Any

import torch
import torch.nn.functional as F

import wandb

from transformers import (
    pipeline,
    LogitsProcessor,
    TopKLogitsWarper,
    LogitsProcessorList,
    AutoModelForCausalLM,
)

from tqdm import tqdm

from al_llm.acquisition_function import AcquisitionFunction
from al_llm.dataset_container import DatasetContainer
from al_llm.parameters import Parameters
from al_llm.classifier import HuggingFaceClassifier
from al_llm.utils.artifacts import load_tapted_model
from al_llm.constants import TEXT_COLUMN_NAME


class TqdmHolder:
    """Helper class to holds the current tqdm instance"""

    def __init__(self):
        self.tqdm_bar: Optional[tqdm] = None


class TopKLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that performs top-k filtering

    Restricts to the k highest probability elements.

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

    def __init__(self, top_k: int, filter_value: float = -float("Inf")):
        self.top_k = top_k
        self.filter_value = filter_value

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        # Compute the indices of the top k scores
        indices_to_keep = torch.topk(scores, self.top_k, dim=1).indices

        # Create a new tensor from `scores`, where all of the values to remove
        # get value `self.filter_value`
        filtered_sores = torch.ones_like(scores) * self.filter_value
        dim_0_indices = (
            torch.arange(scores.shape[0]).repeat(self.top_k, 1).transpose(0, 1)
        )
        filtered_sores[dim_0_indices, indices_to_keep] = scores[
            dim_0_indices, indices_to_keep
        ]

        return filtered_sores


class UncertaintyLogitsProcessor(LogitsProcessor):
    """A logits processor which incorporates uncertainties from a classifier

    Adds to each probability a weighted uncertainty estimate for the sentence
    generated up to this point together with the corresponding token.

    Given the probability value `p` of a particular token (the output of the
    generative model) and the uncertainty value `u` of the sentence generated
    so far plus this token, the new probability value is:
        p + uncertainty_weighting * u

    Given that we actually work with a logit values L and not probabilities
    the actual calculation, producing the new logit values L' given uncertainty
    values U, is:
        L' = log(softmax(L) + uncertainty_weighting * U)

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

        # Get the device used to hold the tensors
        device = input_ids.device

        # The number of input sequences
        num_inputs = input_ids.shape[0]

        # The number of tokens
        num_tokens = scores.shape[1]

        # The mask for the tokens for which we actually want to compute the
        # uncertainties
        scores_mask = scores != self.filter_value

        # Determine the amounts of filtered scores per input
        count_per_input = torch.count_nonzero(scores_mask, dim=1)
        unique_counts = torch.unique(count_per_input)

        # We need that all scores have the same number of filtered values
        if len(unique_counts) != 1:
            raise ValueError(
                f"Parameter `scores` must have the same number"
                f" of filtered values per input. Got {count_per_input.tolist()}"
            )

        # Get the number of filtered scores
        num_filtered_scores = unique_counts.item()

        # Create `num_filtered_scores` copies of each input sequence
        # This creates a tensor of dimension:
        #     num_inputs x num_filtered_scores x {sequence length}
        inputs_repeated = input_ids.repeat(num_filtered_scores, 1, 1).transpose(0, 1)

        # Get the token ids of each of the filtered scores
        score_indices = torch.arange(num_tokens, device=device).repeat(num_inputs, 1)
        filtered_token_ids = score_indices[scores_mask].reshape(
            (num_inputs, num_filtered_scores, 1)
        )

        # Add these token IDs at the end of the repeated inputs, to get a
        # tensor of dimension:
        #     num_inputs x num_filtered_scores x ({sequence length} + 1)
        # which contains all the sequences for which we want to compute the
        # uncertainty
        sequences_block = torch.cat((inputs_repeated, filtered_token_ids), dim=2)

        # Serialise these into a tensor of dimension:
        #     (num_inputs * num_filtered_scores) x ({sequence length} + 1)
        sequences_serialised = torch.flatten(sequences_block, 0, 1)

        # Compute the uncertainties of the input sequences for the classifier
        uncertainties_serialised = self.classifier.calculate_uncertainties_tokenized(
            sequences_serialised, print_output=False
        )

        # Insert the weighted uncertainty values in the appropriate places in a:
        #     num_inputs x num_tokens
        # tensor, so be added to the scores.
        uncertainties_located = torch.zeros_like(scores)
        uncertainties_located[scores_mask] = uncertainties_serialised

        # Compute the probabilities corresponding to the scores using softmax
        probabilities = F.softmax(scores, dim=1)

        # Add the weighted uncertainties to the probabilities
        new_outputs = probabilities + self.uncertainty_weighting * uncertainties_located

        # The new scores are the log of these
        return torch.log(new_outputs)


class TqdmStepLogitsProcessor(LogitsProcessor):
    """A LogitsProcessor which steps a tqdm progress bar

    A bit of a hack to make Hugging Face's generation output a progress bar.
    This is the only way to insert code into the generation loop.

    Parameters
    ----------
    tqdm_bar : tqdm
        The tqdm instance to update
    """

    def __init__(self, tqdm_holder: TqdmHolder):
        super().__init__()
        self.tqdm_holder = tqdm_holder

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        # Update the bar if it exists
        if self.tqdm_holder.tqdm_bar is not None:
            self.tqdm_holder.tqdm_bar.update()

        # Return the scores unchanged
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

    def _make_logits_processor(self) -> LogitsProcessorList:
        """Make a logits processor for use in generation

        Returns
        -------
        logits_processor : LogitsProcessorList
            The list of logits processors to use
        """

        # Create a tqdm holder instance, to communicate with the tqdm updater
        self._tqdm_holder = TqdmHolder()

        # Add the tqdm updater as a 'logits processor'
        tqdm_stepper = TqdmStepLogitsProcessor(self._tqdm_holder)

        # Make this into a list
        logits_processor = LogitsProcessorList([tqdm_stepper])

        return logits_processor

    def _make_pipeline_generator(self, task: str, model: Any, tokenizer: str, **kwargs):
        """Create the generator using pipeline()

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
            logits_processor = (logits_processor,)

        # Get the logits preprocessor to use
        logits_processor = self._make_logits_processor()

        # Create a pipeline for text generation
        self.generator = pipeline(
            task=task,
            model=model,
            device=device,
            tokenizer=tokenizer,
            logits_processor=logits_processor,
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

        # Create a tqdm progress bar to track the generation progress. We
        # assume that we go to the max length; if not, the bar will just end
        # early
        with tqdm(total=self.parameters["sample_generator_max_length"] - 1) as tqdm_bar:

            self._tqdm_holder.tqdm_bar = tqdm_bar

            # Use the pipeline to generate real sentences
            sentence_dicts = self.generator(
                "",
                max_length=self.parameters["sample_generator_max_length"],
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
    """

    MODEL_NAME = "gpt2"

    def __init__(
        self,
        parameters: Parameters,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ):

        super().__init__(parameters, acquisition_function)

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


class TokenByTokenSampleGenerator(PipelineGeneratorMixin, SampleGenerator, ABC):
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

        # Load the base sample generator model
        self._load_generator_model()

        # Setup the pipeline generator
        # Use the Hugging Face generation utility to generate samples. This
        # does most of the hard work for us in terms of interacting with the
        # model. We use a custom logits processor to add the uncertainty
        # values coming from the classifier
        self._make_pipeline_generator(
            "text-generation",
            self.generator_model,
            self.GENERATOR_MODEL_NAME,
            temperature=self.parameters["sample_generator_temperature"],
            top_k=self.parameters["sample_generator_top_k"],
            renormalize_logits=True,
            do_sample=True,
        )

    def _load_generator_model(self):
        """Load the model used as a sentence generator"""

        # Load a text generation verion of the model
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            self.GENERATOR_MODEL_NAME
        )

        # Set the padding token to EOS for open generation
        self.generator_model.config.pad_token_id = (
            self.generator_model.config.eos_token_id
        )

    def _make_logits_processor(self) -> LogitsProcessorList:

        logits_processor = super()._make_logits_processor()

        # The logits processor which filters out the top k tokens before adding
        # the uncertainties
        pre_top_k_logits_processor = TopKLogitsProcessor(
            self.parameters["tbt_pre_top_k"]
        )

        # The logits processor which adds the uncertainty values
        uncertainty_logits_processor = UncertaintyLogitsProcessor(
            self.classifier, self.parameters["tbt_uncertainty_weighting"]
        )

        # Combine these two to the beginning of the processors list
        logits_processor.insert(0, pre_top_k_logits_processor)
        logits_processor.insert(1, uncertainty_logits_processor)

        return logits_processor


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
    """

    MODEL_NAME = ""

    def __init__(
        self,
        parameters: Parameters,
        wandb_run: wandb.sdk.wandb_run.Run,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ):

        super().__init__(parameters, acquisition_function)

        self.wandb_run = wandb_run

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
    """

    MODEL_NAME = "gpt2"


class PlainGPT2TokenByTokenSampleGenerator(TokenByTokenSampleGenerator):
    """GPT-2 token-by-token generator to maximise uncertainty

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

    GENERATOR_MODEL_NAME = "gpt2"
