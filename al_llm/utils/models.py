from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from dataclasses import dataclass


@dataclass
class HuggingFaceClassifierEnsembleOutput(ModelOutput):
    """Output for an ensemble of Hugging Face classifiers

    Parameters
    ----------
    loss : torch.Tensor of shape (num_models)
        The loss value for each model
    class_probs : torch.Tensor of shape (batch_size, num_labels)
        The mean class probabilities
    """

    loss: torch.Tensor
    class_probs: torch.Tensor


class HuggingFaceClassifierEnsemble(nn.Module):
    """An ensemble of Hugging Face classifiers

    The `forward` method outputs a tensor of the mean class probabilities
    across the models.

    Parameters
    ----------
    models : list of PreTrainedModel
        The models which will make up the ensemble
    """

    def __init__(self, models: Iterable[PreTrainedModel]):
        super().__init__()

        # Store the models as a list
        self.models = list(models)

        # Store the number of models
        self.num_models = len(list(models))

        # Store the number of class labels, and make sure it's the same for
        # each model
        self.num_labels = self.models[0].config.num_labels
        for i, model in enumerate(models):
            if model.config.num_labels != self.num_labels:
                raise ValueError(
                    f"Expected all models to have the same number of class "
                    f"labels, but model 0 has {self.num_labels} while model "
                    f"{i} has {model.config.num_labels}."
                )

        # Add each model as an attribute. This registers them as submodules,
        # so that e.g. `to` works on them automatically, and we get nice
        # printing
        for i, model in enumerate(models):
            self.__setattr__(f"model{i}", model)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
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

        # Tensors of the outputs to be computed
        batch_size = input_ids.shape[0]
        loss = torch.zeros(self.num_models)
        class_probs = torch.zeros((batch_size, self.num_labels, self.num_models))

        for i, model in enumerate(self.models):

            # Compute model outputs using clones of the inputs, in case these
            # get modified by calling
            input_ids_clone = torch.clone(input_ids)
            if attention_mask is not None:
                attention_mask_clone = torch.clone(attention_mask)
            else:
                attention_mask_clone = None
            outputs = model(
                input_ids=input_ids_clone, attention_mask=attention_mask_clone
            )

            # Store the loss and prediction probabilities
            loss[i] = outputs.loss
            class_probs[:, :, i] = F.softmax(outputs.logits)

        # Take the mean of the class probabilities over all models
        class_probs = torch.mean(class_probs, dim=2)

        # Put the loss and class probabilities into a data structure
        output = HuggingFaceClassifierEnsembleOutput(loss=loss, class_probs=class_probs)

        return output
