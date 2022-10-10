import torch
import torch.nn.functional as F

from transformers import LogitsProcessor

from al_llm.classifier import HuggingFaceClassifier


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
