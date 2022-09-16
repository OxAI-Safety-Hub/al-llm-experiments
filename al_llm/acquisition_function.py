# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

import random
from typing import Tuple, List
from toma import toma
import torch
from tqdm.auto import tqdm

from al_llm.classifier import Classifier, UncertaintyMixin
from al_llm.parameters import Parameters
from al_llm.joint_entropy import DynamicJointEntropy


class AcquisitionFunction(ABC):
    """Base acquisition function

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    @abstractmethod
    def select(self, sample_pool: list, num_samples: int = -1) -> list:
        """Apply the acquisition function

        Parameters
        ----------
        sample_pool : list
            The list of sentences from which to sample
        num_samples : int, default=-1
            The number of samples to select. The default value of -1 means
            that `parameters["num_samples"]` is used.

        Returns
        -------
        selected_samples : list
            A sublist of `samples` of size `num_samples` selected according
            the to acquisition function.
        """
        pass

    def _get_validated_num_samples(
        self, sample_pool: list, num_samples: int = -1
    ) -> int:
        """Determine and validate the number of samples to take

        The value of -1 means that `parameters["num_samples"]` is used.
        """

        if num_samples == -1:
            num_samples = self.parameters["num_samples"]

        if num_samples > len(sample_pool):
            raise ValueError(
                f"The size of `sample_pool` must be at least that of `num_samples`"
                f" (currently {num_samples} > {len(sample_pool)}"
            )

        return num_samples


class DummyAF(AcquisitionFunction):
    """A dummy acquisition function, which selects the first slice of samples

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    """

    def select(self, sample_pool: list, num_samples: int = -1) -> list:
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)
        return sample_pool[:num_samples]


class RandomAF(AcquisitionFunction):
    """An acquisition function which selects randomly

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    """

    def select(self, sample_pool: list, num_samples: int = -1) -> list:
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)
        return random.sample(sample_pool, num_samples)


class MaxUncertaintyAF(AcquisitionFunction):
    """An acquisition function which selects for the highest uncertainty

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: Parameters, classifier: Classifier):
        super().__init__(parameters)
        if not isinstance(classifier, UncertaintyMixin):
            raise TypeError("`classifier` must implement uncertainty measuring")
        self.classifier = classifier

    def select(self, sample_pool: list, num_samples: int = -1) -> list:

        # Process and validate `num_samples`
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)

        # Compute the uncertainty values of each of the samples
        uncertainties = self.classifier.calculate_uncertainties(sample_pool)

        # Compute the index array which would sort these in ascending order
        argsorted = sorted(range(len(sample_pool)), key=lambda i: uncertainties[i])

        # Take the values of `samples` at the last `num_samples` indices
        uncertain_samples = [sample_pool[i] for i in argsorted[-num_samples:]]

        return uncertain_samples


class BatchBaldAF(AcquisitionFunction):
    """An acquisition function using the batch bald algorithm

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: Parameters, classifier: Classifier):
        super().__init__(parameters)
        self.classifier = classifier

    def select(self, sample_pool: list, num_samples: int = -1) -> list:

        max_training_samples = 150
        acquisition_batch_size = 5
        num_inference_samples = 100
        num_test_inference_samples = 5

        test_batch_size = 512
        batch_size = 64
        scoring_batch_size = 128
        training_iterations = 4096 * 6

        # Process and validate `num_samples`
        num_samples = self._get_validated_num_samples(sample_pool, num_samples)

        # Acquire pool predictions
        N = len(sample_pool)
        logits_N_K_C = torch.empty((N, num_inference_samples, 2), dtype=torch.double)

        with torch.no_grad():
            self.classifier._model.eval()

            for i in tqdm(
                range(len(sample_pool)), desc="Evaluating Acquisition Set", leave=False
            ):
                # Following code doesn't work. The sample pool needs to be tokenized 
                # and ideally put into a dataloader if we wanted to match the original
                # code.
                data = sample_pool[i].to(device=self.classifier.device)
                logits_N_K_C[i : i + 1].copy_(
                    self.classifier._model(data, num_inference_samples).double(),
                    non_blocking=True,
                )

        with torch.no_grad():
            scores, indices = self._get_batchbald_batch(
                logits_N_K_C,
                acquisition_batch_size,
                num_samples,
                dtype=torch.double,
                device=self.classifier.device,
            )

        new_samples = []
        for index in indices:
            new_samples.append(sample_pool[index])

    def _get_batchbald_batch(
        self,
        log_probs_N_K_C: torch.Tensor,
        batch_size: int,
        num_samples: int,
        dtype=None,
        device=None,
    ) -> Tuple[List[float], List[int]]:

        # len(sample_pool), num_inference_samples, num_classes
        N, K, C = log_probs_N_K_C.shape

        batch_size = min(batch_size, N)

        candidate_indices = []
        candidate_scores = []

        if batch_size == 0:
            return (candidate_scores, candidate_indices)

        conditional_entropies_N = self._compute_conditional_entropy(log_probs_N_K_C)

        batch_joint_entropy = DynamicJointEntropy(
            num_samples, batch_size - 1, K, C, dtype=dtype, device=device
        )

        # We always keep these on the CPU.
        scores_N = torch.empty(
            N, dtype=torch.double, pin_memory=torch.cuda.is_available()
        )

        for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
            if i > 0:
                latest_index = candidate_indices[-1]
                batch_joint_entropy.add_variables(
                    log_probs_N_K_C[latest_index : latest_index + 1]
                )

            shared_conditinal_entropies = conditional_entropies_N[
                candidate_indices
            ].sum()

            batch_joint_entropy.compute_batch(
                log_probs_N_K_C, output_entropies_B=scores_N
            )

            scores_N -= conditional_entropies_N + shared_conditinal_entropies
            scores_N[candidate_indices] = -float("inf")

            candidate_score, candidate_index = scores_N.max(dim=0)

            candidate_indices.append(candidate_index.item())
            candidate_scores.append(candidate_score.item())

        return (candidate_scores, candidate_indices)

    def _compute_conditional_entropy(
        self, log_probs_N_K_C: torch.Tensor
    ) -> torch.Tensor:
        N, K, C = log_probs_N_K_C.shape

        entropies_N = torch.empty(N, dtype=torch.double)

        pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

        @toma.execute.chunked(log_probs_N_K_C, 1024)
        def compute(log_probs_n_K_C, start: int, end: int):
            nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

            entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
            pbar.update(end - start)

        pbar.close()

        return entropies_N
