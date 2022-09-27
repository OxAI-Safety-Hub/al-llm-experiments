from typing import List, Optional

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from dataclasses import dataclass


@dataclass
class HuggingFaceClassifierEnsembleOutput(ModelOutput):
    """Output for an ensemble of Hugging Face classifiers

    Parameters
    ----------
    loss : torch.FloatTensor of shape (num_models)
        The loss value for each model
    class_probs : torch.FloatTensor of shape (batch_size, config.num_labels)
        The mean class probabilities
    """

    loss: torch.FloatTensor
    class_probs: torch.FloatTensor


class HuggingFaceClassifierEnsemble(nn.Module):
    """An ensemble of Hugging Face classifiers

    The `forward` method outputs a tensor of the mean class probabilities
    across the models.

    Parameters
    ----------
    models : list of PreTrainedModel
        The models which will make up the ensemble
    """

    def __init__(self, models: List[PreTrainedModel]):
        super().__init__()
        self.models = models

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> HuggingFaceClassifierEnsembleOutput:
        """Do a forward pass on each model and average the results

        Parameters
        ----------
        input_ids : torch.LongTensor of shape (batch_size, input_ids_length)
            The input IDs of the sequences to classify.
        attention_mask : torch.LongTensor of shape (batch_size, sequence_length)
            The attention mask, to avoid putting attention on padding tokens

        Returns
        -------
        output : HuggingFaceClassifierEnsembleOutput
            The results of doing the forward pass
        """
        pass

    def to(self, *args):
        pass

    def eval(self):
        for model in self.models:
            model.eval()

    def train(self):
        for model in self.models:
            model.train()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return super().parameters(recurse)