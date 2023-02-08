# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
import random
from typing import Optional, Any

import torch

import wandb

from transformers import (
    PreTrainedTokenizer,
    pipeline,
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from transformers.pipelines import PIPELINE_REGISTRY

from tqdm import tqdm

from al_llm.acquisition_function import AcquisitionFunction
from al_llm.dataset_container import DatasetContainer
from al_llm.data_handler import DataHandler
from al_llm.parameters import Parameters
from al_llm.classifier import HuggingFaceClassifier
from al_llm.utils import UnlabelledSamples
from al_llm.utils.artifacts import load_tapted_model
from al_llm.utils.generation import (
    TopKLogitsProcessor,
    UncertaintyLogitsProcessor,
    MaskedMHSamplerPipeline,
)
from al_llm.constants import TEXT_COLUMN_NAME, LABEL_COLUMN_NAME

PIPELINE_REGISTRY.register_pipeline(
    "mmh-text-generation",
    pipeline_class=MaskedMHSamplerPipeline,
)


class TqdmHolder:
    """Helper class to hold the current tqdm instance"""

    def __init__(self):
        self.tqdm_bar: Optional[tqdm] = None


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
    dataset_container : DatasetContainer
        The dataset container for the experiment, which holds the remainder
        dataset
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        self.parameters = parameters
        self.dataset_container = dataset_container
        self.wandb_run = wandb_run
        self.acquisition_function = acquisition_function

    def generate(self) -> UnlabelledSamples:
        """Generate new samples for querying

        Returns
        -------
        samples : UnlabelledSamples
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
    def _generate_sample_pool(self, pool_size: int) -> UnlabelledSamples:
        """Generate a pool of samples, from which to select

        Parameters
        ----------
        pool_size: int
            The number of samples to generate

        Returns
        -------
        sample_pool : UnlabelledSamples
            The sample pool from which to select
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
    dataset_container : DatasetContainer
        The dataset container for the experiment, which holds the remainder
        dataset
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    def _generate_sample_pool(self, pool_size: int) -> UnlabelledSamples:
        alphabet = "abcdefghijklmnopqrstuvwxyz         "

        # Generate the samples by sampling from the alphabet
        sample_pool = UnlabelledSamples()
        for sample_index in range(pool_size):
            length = random.randrange(5, 30)
            sample_nums = [random.randrange(len(alphabet)) for i in range(length)]
            sample_chars = map(lambda x: alphabet[x], sample_nums)
            sample = "".join(sample_chars)
            sample_pool.append(sample)

        return sample_pool


class PoolSampleGenerator(SampleGenerator):
    """Generate samples by sampling from the remainder dataset.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for the experiment, which holds the remainder
        dataset
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction
        The acquisition function to use.
    """

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        acquisition_function: AcquisitionFunction,
    ):
        super().__init__(parameters, dataset_container, wandb_run, acquisition_function)

        # Get the list of sentences in the remainder dataset, as a list
        remainder_python = dataset_container.dataset_remainder.with_format(None)
        self.remainder_sentences = remainder_python[TEXT_COLUMN_NAME]
        self.remainder_labels = remainder_python[LABEL_COLUMN_NAME]

    def generate(self) -> UnlabelledSamples:
        # Filter the acquisition function through the set of sentences in the
        # simulated pool taken from the remainder dataset
        sample_pool = self._generate_sample_pool()
        print()
        print("Selecting samples using the acquisition function...")
        return self.acquisition_function.select(sample_pool)

    def _generate_sample_pool(self) -> UnlabelledSamples:
        # Take `sample_pool_size` random samples from `remainder_sentences`, or
        # as many as you can take up to the length of `remainder_sentences`
        pool_indices = random.sample(
            range(len(self.remainder_sentences)),
            min(len(self.remainder_sentences), self.parameters["sample_pool_size"]),
        )

        # Build an `UnlabelledSamples` object, attaching the dataset labels as
        # the 'suggested labels'
        simulated_pool = UnlabelledSamples(
            [self.remainder_sentences[i] for i in pool_indices]
        )
        simulated_pool.suggested_labels = [
            self.remainder_labels[i] for i in pool_indices
        ]

        return simulated_pool


class ReplaySampleGenerator(SampleGenerator):
    """Retrieve samples from a previous run

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for the experiment, which holds the remainder
        dataset
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    data_handler : DataHandler
        The data handler for the experiment
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use.
    """

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        data_handler: DataHandler,
    ):
        super().__init__(parameters, dataset_container, wandb_run, None)
        self.data_handler = data_handler

        # The current index in the replay dataset extension
        self._iteration = 0

    def generate(self) -> UnlabelledSamples:
        # Announce what we're doing
        print()
        print("Getting samples from the replayed run...")

        # Get the next batch of samples from the replayed run
        samples = self.data_handler.get_replay_samples(self._iteration)

        # Update the index
        self._iteration += 1

        return samples

    def _generate_sample_pool(self, pool_size: int) -> UnlabelledSamples:
        return []


class HuggingFaceSampleGenerator(SampleGenerator, ABC):
    """Base class for sample generators using Hugging Face models

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for the current experiment.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    GENERATOR_MODEL_NAME = ""

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        super().__init__(parameters, dataset_container, wandb_run, acquisition_function)

        # Set the max sentence length to generate, in tokens
        if self.parameters["sample_generator_max_length"] == -1:
            self._max_length = dataset_container.TOKENIZED_LENGTH_UPPER_QUARTILE
        else:
            self._max_length = self.parameters["sample_generator_max_length"]

        # Load the model we'll use for generating
        self._load_generator_model()

        # Setup the pipeline generator
        self._make_pipeline_generator(self.generator_model, self.GENERATOR_MODEL_NAME)

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

    def _make_pipeline_generator(self, model: Any, tokenizer: str, **kwargs):
        """Create the generator using pipeline()

        Parameters
        ----------
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

        # Get the logits preprocessor to use
        logits_processor = self._make_logits_processor()

        # Create a pipeline for text generation
        self.generator = pipeline(
            task="text-generation",
            model=model,
            device=device,
            tokenizer=tokenizer,
            temperature=self.parameters["sample_generator_temperature"],
            top_k=self.parameters["sample_generator_top_k"],
            logits_processor=logits_processor,
            **kwargs,
        )

    def _generate_sample_pool(self, pool_size: int) -> UnlabelledSamples:
        """Generate a pool of samples, from which to select

        Parameters
        ----------
        pool_size: int
            The number of samples to generate

        Returns
        -------
        sample_pool : UnlabelledSamples
            The sample pool from which to select
        """

        # Create a tqdm progress bar to track the generation progress. We
        # assume that we go to the max length; if not, the bar will just end
        # early
        with tqdm(total=self._max_length - 1) as tqdm_bar:
            self._tqdm_holder.tqdm_bar = tqdm_bar

            # Use the pipeline to generate real sentences
            sentence_dicts = self.generator(
                "",
                max_length=self._max_length,
                num_return_sequences=pool_size,
            )

        sample_pool = UnlabelledSamples([d["generated_text"] for d in sentence_dicts])

        return sample_pool


class TokenByTokenSampleGenerator(HuggingFaceSampleGenerator, ABC):
    """Base class to generate a sentence token-by-token to maximise uncertainty

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    classifier : HuggingFaceClassifier
        The classifier used in the current experiment, for which we maximise
        uncertainty.
    dataset_container : DatasetContainer
        The dataset container for the current experiment.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    def __init__(
        self,
        parameters: Parameters,
        classifier: HuggingFaceClassifier,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        super().__init__(parameters, dataset_container, wandb_run, acquisition_function)

        self.classifier = classifier

    def _make_pipeline_generator(self, model: Any, tokenizer: str, **kwargs):
        # Make the default pipeline with two extra arguments
        kwargs["renormalize_logits"] = True
        kwargs["do_sample"] = True
        super()._make_pipeline_generator(model, tokenizer, **kwargs)

    def _make_logits_processor(self) -> LogitsProcessorList:
        logits_processor = super()._make_logits_processor()

        # The logits processor which filters out the top k tokens before adding
        # the uncertainties
        pre_top_k_logits_processor = TopKLogitsProcessor(
            self.parameters["tbt_pre_top_k"]
        )

        # The logits processor which adds the uncertainty values
        uncertainty_logits_processor = UncertaintyLogitsProcessor(
            self.parameters, self.classifier, self._max_length
        )

        # Combine these two to the beginning of the processors list
        logits_processor.insert(0, pre_top_k_logits_processor)
        logits_processor.insert(1, uncertainty_logits_processor)

        return logits_processor


class MaskedMHSampleGenerator(HuggingFaceSampleGenerator, ABC):
    """Generate samples using the masked Metropolis-Hastings sampler

    This generator uses a masked language model (MLM) and generates samples
    using a modified Metropolis-Hastings Markov Chain Monte Carlo sampler.

    The algorithm proceeds as follows.

    1. Start with a set of `parameters["sample_pool_size"]` initial samples
       taken from the unlabelled set.
    2. Repeat the following three steps for `parameters["mmh_num_steps"]`.
    3. In each sample, randomly mask `parameters["mmh_mask_proportion"]` of
       the tokens, and sample tokens to replace them based on the MLM's
       probability distribution.
    4. Accept these new tokens with probability given by the resulting
       classifier uncertainty.
    5. If they are not accepted, return the tokens back the way they were.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    classifier : HuggingFaceClassifier
        The classifier used in the current experiment, for which we maximise
        uncertainty.
    dataset_container : DatasetContainer
        The dataset container for the current experiment.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    def __init__(
        self,
        parameters: Parameters,
        classifier: HuggingFaceClassifier,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        super().__init__(parameters, dataset_container, wandb_run, acquisition_function)

        self.classifier = classifier

        # Get the list of sentences in the remainder dataset, as a list
        remainder_python = dataset_container.dataset_remainder.with_format(None)
        self.remainder_sentences = remainder_python[TEXT_COLUMN_NAME]
        self.remainder_labels = remainder_python[LABEL_COLUMN_NAME]

    def _load_generator_model(self):
        """Load the model used as a sentence generator"""

        # Load a text generation verion of the model
        self.generator_model = AutoModelForMaskedLM.from_pretrained(
            self.GENERATOR_MODEL_NAME
        )

    def _make_logits_warper(self) -> LogitsProcessorList:
        """Make a logits warper for use in generation

        Returns
        -------
        logits_warper : LogitsProcessorList
            The list of logits warpers to use
        """

        # Warper for chainging the temperature
        temperature_logits_warper = TemperatureLogitsWarper(
            self.parameters["sample_generator_temperature"]
        )

        # Warper for filtering the top k
        top_k_logits_warper = TopKLogitsWarper(
            self.parameters["sample_generator_top_k"]
        )

        # Make these into a list
        logits_warper = LogitsProcessorList(
            [temperature_logits_warper, top_k_logits_warper]
        )

        return logits_warper

    def _make_pipeline_generator(self, model: Any, tokenizer: str, **kwargs):
        """Create the generator using pipeline()

        Parameters
        ----------
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

        # The scoring function to use in the sampling algorithm, which scores
        # using the classifier uncertainty
        def scoring_function(
            sample_ids: torch.Tensor, tokenizer: PreTrainedTokenizer
        ) -> torch.Tensor:
            # If we're using the sample classifer model as the generator, we
            # don't need to retokenize
            if self.GENERATOR_MODEL_NAME == self.classifier.MODEL_NAME:
                samples_tokenized = sample_ids

            # Otherwise, retokenize the input IDs
            else:
                samples = tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
                samples_tokenized = self.classifier.tokenize(
                    samples, return_tensors="pt"
                )["input_ids"]

            # Compute the uncertainties of the samples
            uncertainties = self.classifier.calculate_uncertainties_tokenized(
                samples_tokenized, output_probabilities=True, print_output=False
            )

            return uncertainties

        # Get the logits preprocessor to use
        logits_warper = self._make_logits_warper()

        # Create a pipeline for text generation
        self.generator = pipeline(
            task="mmh-text-generation",
            model=model,
            device=device,
            tokenizer=tokenizer,
            num_steps=self.parameters["mmh_num_steps"],
            scoring_function=scoring_function,
            mask_probability=self.parameters["mmh_mask_probability"],
            logits_warper=logits_warper,
            **kwargs,
        )

    def _generate_sample_pool(self, pool_size: int) -> UnlabelledSamples:
        """Generate a pool of samples, from which to select

        Parameters
        ----------
        pool_size: int
            The number of samples to generate

        Returns
        -------
        sample_pool : UnlabelledSamples
            The sample pool from which to select
        """

        # Select `pool_size` samples from the unlabelled pool
        pool_indices = random.sample(
            range(len(self.remainder_sentences)),
            min(len(self.remainder_sentences), self.parameters["sample_pool_size"]),
        )
        initial_samples = [self.remainder_sentences[i] for i in pool_indices]

        # Run the Masked MH algorithm starting with these
        sample_pool = self.generator(
            initial_samples, batch_size=self.parameters["eval_batch_size"]
        )
        sample_pool = UnlabelledSamples(sample_pool)

        # Add the original labels as suggested labels for each sample
        sample_pool.suggested_labels = [self.remainder_labels[i] for i in pool_indices]

        return sample_pool


class TAPTMixin(ABC):
    """Mixin to provide methods for TAPTed sample generators"""

    def _load_generator_model(self):
        """Loads the pretrained model from wandb

        Sets `self.model` to this pretrained model after loading
        """

        # load model and training args from wandb
        model, training_args = load_tapted_model(
            self.wandb_run,
            self.GENERATOR_MODEL_NAME,
            self.parameters["dataset_name"],
            "sample_generator",
            tapted_model_version=self.parameters["tapted_model_version"],
        )
        self.generator_model = model
        self.training_parameters = training_args

        # Set the padding token to EOS for open generation
        self.generator_model.config.pad_token_id = (
            self.generator_model.config.eos_token_id
        )

    def get_training_parameters(self) -> dict:
        """Get the parameters used for training this model

        Returns
        -------
        self.training_parameters : dict
            The TAPT training parameters
        """
        return self.training_parameters


class PlainGPT2SampleGenerator(HuggingFaceSampleGenerator):
    """Plain GPT-2 sample generator, which just generates real sentences

    It generates `parameters["num_samples"]` samples. If an acquisition
    function is supplied, it first generates `parameters["sample_pool_size"]`
    then uses the function to select from these.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for the current experiment.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    GENERATOR_MODEL_NAME = "gpt2"


class TAPTDistilGPT2SampleGenerator(TAPTMixin, HuggingFaceSampleGenerator):
    """Tapted distilGPT-2 sample generator, which generates real sentences

    It generates `parameters["num_samples"]` samples. If an acquisition
    function is supplied, it first generates `parameters["sample_pool_size"]`
    then uses the function to select from these.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for the current experiment.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    GENERATOR_MODEL_NAME = "distilgpt2"


class TAPTGPT2SampleGenerator(TAPTMixin, HuggingFaceSampleGenerator):
    """Tapted GPT-2 sample generator, which generates real sentences

    It generates `parameters["num_samples"]` samples. If an acquisition
    function is supplied, it first generates `parameters["sample_pool_size"]`
    then uses the function to select from these.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for the current experiment.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : acquisition_function.AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    GENERATOR_MODEL_NAME = "gpt2"


class PlainGPT2TokenByTokenSampleGenerator(TokenByTokenSampleGenerator):
    """GPT-2 token-by-token generator to maximise uncertainty

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    classifier : HuggingFaceClassifier
        The classifier used in the current experiment, for which we maximise
        uncertainty.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    GENERATOR_MODEL_NAME = "gpt2"


class TAPTGPT2TokenByTokenSampleGenerator(TAPTMixin, TokenByTokenSampleGenerator):
    """TAPTed GPT-2 token-by-token generator to maximise uncertainty

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    classifier : HuggingFaceClassifier
        The classifier used in the current experiment, for which we maximise
        uncertainty.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    GENERATOR_MODEL_NAME = "gpt2"


class PlainBERTMaskedMHSampleGenerator(MaskedMHSampleGenerator):
    """BERT Masked Metropolis-Hastings sample generator

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    classifier : HuggingFaceClassifier
        The classifier used in the current experiment, for which we maximise
        uncertainty.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    GENERATOR_MODEL_NAME = "bert-base-cased"


class TAPTBERTMaskedMHSampleGenerator(TAPTMixin, MaskedMHSampleGenerator):
    """TAPTed BERT Masked Metropolis-Hastings sample generator

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    classifier : HuggingFaceClassifier
        The classifier used in the current experiment, for which we maximise
        uncertainty.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    acquisition_function : AcquisitionFunction, optional
        The acquisition function to use, if any. By default we simply generate
        a number of samples with no selection procedure.
    """

    GENERATOR_MODEL_NAME = "bert-base-cased"
