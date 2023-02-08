from typing import Optional, Union, Callable, List

import torch
import torch.nn.functional as F

from transformers import LogitsProcessor, LogitsProcessorList, Pipeline

from tqdm import tqdm

from al_llm.classifier import HuggingFaceClassifier
from al_llm.parameters import Parameters


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
    parameters : Parameters
        The parameters for the current experiment
    classifier : HuggingFaceClassifier
        The classifier for which to compute the uncertainties
    max_length : int
        The maximum length of sentence which will be generated.
    filter_value : float, default=`-float("Inf")`
        The value used in the `scores` by previous processors to indicate that
        we shouldn't consider that token.
    """

    def __init__(
        self,
        parameters: Parameters,
        classifier: HuggingFaceClassifier,
        max_length: int,
        filter_value: float = -float("Inf"),
    ):
        super().__init__()

        self.parameters = parameters
        self.classifier = classifier
        self.max_length = max_length
        self.filter_value = filter_value

    @torch.no_grad()
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Get the device used to hold the tensors
        device = input_ids.device

        # The number of input sequences
        num_inputs = input_ids.shape[0]

        # The length of the sequences generated so far
        sequnence_len = input_ids.shape[1]

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
        #     num_inputs x num_filtered_scores x sequnence_len
        inputs_repeated = input_ids.repeat(num_filtered_scores, 1, 1).transpose(0, 1)

        # Get the token ids of each of the filtered scores
        score_indices = torch.arange(num_tokens, device=device).repeat(num_inputs, 1)
        filtered_token_ids = score_indices[scores_mask].reshape(
            (num_inputs, num_filtered_scores, 1)
        )

        # Add these token IDs at the end of the repeated inputs, to get a
        # tensor of dimension:
        #     num_inputs x num_filtered_scores x (sequnence_len + 1)
        # which contains all the sequences for which we want to compute the
        # uncertainty
        sequences_block = torch.cat((inputs_repeated, filtered_token_ids), dim=2)

        # Serialise these into a tensor of dimension:
        #     (num_inputs * num_filtered_scores) x (sequnence_len + 1)
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

        # Compute the weighting for the uncertainty values
        weighting = self.parameters["tbt_uncertainty_weighting"]
        if self.parameters["tbt_uncertainty_scheduler"] == "linear":
            weighting = max(0, min(1, sequnence_len / self.max_length)) * weighting

        # Add the weighted uncertainties to the probabilities
        new_outputs = probabilities + weighting * uncertainties_located

        # The new scores are the log of these
        return torch.log(new_outputs)


class MaskedMHSamplerPipeline(Pipeline):
    """Pipeline for generating sentences using masked Metropolis-Hastings

    This sampler generates sentences for a Masked-Language Model (MLM) using a
    variant of the Metropolis-Hastings Markov Chain Monte Carlo (MCMC)
    sampler. The aim is to generate sentences which are simultaneously
    high-probability according to the language model, and give high values
    according to some scoring function.

    More precisely, we consider the MLM distribution over sentences as a
    prior, and perform a Bayesian update using a distribution given by the
    scoring function. We then aim to sample from this distribution.

    The algorithm works as follows.

    1. Start with a set of initial samples (e.g. taken from the unlabelled set
       of sentences).
    2. Repeat the following three steps for `num_steps`.
    3. In each sample, randomly mask some of the tokens, and sample tokens to
       replace them based on the MLM's probability distribution.
    4. Accept these new tokens with probability given by the value of the
       scoring function.
    5. If they are not accepted, return the tokens back the way they were.
    """

    def _sanitize_parameters(
        self,
        num_steps: Optional[int] = None,
        scoring_function: Optional[Callable] = None,
        mask_probability: Optional[float] = None,
        logits_processor: Optional[Union[LogitsProcessorList, list]] = None,
        logits_warper: Optional[Union[LogitsProcessorList, list]] = None,
        **tokenizer_kwargs,
    ):
        """Sanitize any excess parameters to `__init__` or `__call__`

        Parameters
        ----------
        num_steps : int, optional
            The number of steps for which to run the algorithm
        scoring_function : callable, optional
            Scoring function used to evaluate new generated samples
        mask_probability : float, optional
            The probability that any individual token will be masked
        logits_processor : LogitsProcessorList or list, optional
            The logits processor to use on the logits outputed at each stage
            by the model
        logits_warper : LogitsProcessorList or list, optional
            The logits warper to use on the logits outputed at each stage by
            the model
        """

        # Set the initial kwargs for the three key methods
        preprocess_kwargs = tokenizer_kwargs
        forward_kwargs = {}
        postprocess_kwargs = {}

        if num_steps is not None:
            forward_kwargs["num_steps"] = num_steps

        if scoring_function is not None:
            forward_kwargs["scoring_function"] = scoring_function

        if mask_probability is not None:
            if mask_probability <= 0 or mask_probability > 1:
                raise ValueError(
                    f"Parameter `mask_probability` should lie in the interval "
                    f"(0,1]; got {mask_probability}"
                )
            forward_kwargs["mask_probability"] = mask_probability

        if logits_processor is not None:
            if isinstance(logits_processor, list):
                logits_processor = LogitsProcessorList(logits_processor)
            forward_kwargs["logits_processor"] = logits_processor

        if logits_warper is not None:
            if isinstance(logits_warper, list):
                logits_warper = LogitsProcessorList(logits_warper)
            forward_kwargs["logits_warper"] = logits_warper

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs, **tokenizer_kwargs):
        # Tokenize the inputs
        if isinstance(inputs, dict):
            return self.tokenizer(
                **inputs, padding="max_length", truncation=True, **tokenizer_kwargs
            )
        else:
            return self.tokenizer(
                inputs, padding="max_length", truncation=True, **tokenizer_kwargs
            )

    @torch.no_grad()
    def _forward(
        self,
        samples: dict,
        num_steps: int,
        scoring_function: Callable,
        mask_probability: float,
        logits_processor: LogitsProcessorList = LogitsProcessorList([]),
        logits_warper: LogitsProcessorList = LogitsProcessorList([]),
    ) -> torch.Tensor:
        """Run the MH sampler on the preprocessed inputs

        Parameters
        ----------
        num_steps : int
            The number of steps for which to run the algorithm
        scoring_function : callable
            Scoring function used to evaluate new generated samples. Should
            take as input a tensor of shape (batch_size, sequence_length) of
            samples and the pipeline tokenizer `tokenizer`, and output a
            tensor of shape (batch_size) consisting of the scores for each
            element in the batch.
        mask_probability : float
            The probability that any individual token will be masked
        logits_processor : LogitsProcessorList or list,
        default=LogitsProcessorList([])
            The logits processor to use on the logits outputed at each stage
            by the model
        logits_warper : LogitsProcessorList or list,
        default=LogitsProcessorList([])
            The logits warper to use on the logits outputed at each stage by
            the model

        Returns
        -------
        sample_ids : torch.Tensor
            The token ids of the newly generated samples.
        """

        # The token ids of the samples
        sample_ids = samples["input_ids"]
        if not isinstance(sample_ids, torch.Tensor):
            sample_ids = torch.tensor(sample_ids)

        # Add a batch dimension, if we only have a single sample
        if sample_ids.ndim == 1:
            sample_ids = torch.unsqueeze(sample_ids, dim=0)

        # The number of samples we have (batch size)
        batch_size = sample_ids.shape[0]

        # The length of the sequences (should be the model max length)
        sequence_length = sample_ids.shape[1]

        print("Generating samples using masked Metropolis-Hastings...")

        # Iterate through the MCMC process
        for i in tqdm(range(num_steps)):
            # To mask some of the tokens, First choose from all tokens
            # uniformly with probability `mask_probability`
            masking_mask = torch.rand_like(sample_ids, dtype=float) <= mask_probability

            # We now want to select those previous token in the original
            # sequence was not a padding token
            non_padding_mask = (
                (sample_ids != self.tokenizer.pad_token_id)
                & (sample_ids != self.tokenizer.cls_token_id)
                & (sample_ids != self.tokenizer.sep_token_id)
            )
            masking_mask = masking_mask & non_padding_mask

            # Replace all the tokens chosen to be masked with the mask token
            masked_sample_ids = torch.where(
                masking_mask, self.tokenizer.mask_token_id, sample_ids
            )

            # Move everything onto the correct device
            masked_sample_ids = masked_sample_ids.to(self.device)
            non_padding_mask = non_padding_mask.to(self.device)
            self.model.to(self.device)

            # Pass these through the model to get the logits over the masked
            # tokens
            output = self.model(
                input_ids=masked_sample_ids, attention_mask=non_padding_mask
            )
            logits = output.logits

            # Apply the logits processors and warpers
            logits = logits_processor(sample_ids, logits)
            logits = logits_warper(sample_ids, logits)

            # Compute the class probabilites by applying the softmax
            class_probs = F.softmax(logits, dim=-1)

            # Sample from the class probabilities
            class_probs_flat = torch.flatten(class_probs, start_dim=0, end_dim=1)
            sampled_tokens = torch.multinomial(class_probs_flat, num_samples=1)
            sampled_tokens = sampled_tokens.reshape((batch_size, sequence_length))

            # Put the tensors on the correct device
            masking_mask = masking_mask.to(self.device)
            sampled_tokens = sampled_tokens.to(self.device)
            sample_ids = sample_ids.to(self.device)

            # Replace the masked tokens using the sampled ones
            new_sample_ids = torch.where(masking_mask, sampled_tokens, sample_ids)

            # Compute the values of the scoring function for the new sample_ids
            new_sample_scores = scoring_function(new_sample_ids, self.tokenizer)

            # Keep the new samples with probability given by the scoring
            # function
            keep_indices = (
                torch.rand(batch_size, device=self.device) <= new_sample_scores
            )
            keep_indices = torch.unsqueeze(keep_indices, dim=1)
            sample_ids = torch.where(keep_indices, new_sample_ids, sample_ids)

        return sample_ids

    def postprocess(self, sample_ids: torch.Tensor) -> List[str]:
        """Detokenize the results of the algorithm

        Parameters
        ----------
        sample_ids : torch.Tensor
            The ouputs from running the MH sampler, which are tokenized
            samples.

        Returns
        -------
        list of str
            The detokenized samples
        """

        # Use `decode` or `batch_decode` depending on whether we have a batch
        # dimension
        if sample_ids.ndim == 1:
            return self.tokenizer.decode(sample_ids, skip_special_tokens=True)
        else:
            decoded = self.tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
            # If we only have one sample, we need to return the singleton item
            # to make it all work.
            if len(decoded) == 1:
                return decoded[0]
            else:
                return decoded
