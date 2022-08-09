# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Union, Any

import torch

import datasets

from transformers import AutoTokenizer


class Classifier(ABC):
    """Base classifier class

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters

    @abstractmethod
    def train_afresh(self, dataset: Union[datasets.Dataset, torch.utils.data.Dataset]):
        """Reset the classifier and fine-tune it anew on tokenised data

        Parameters
        ----------
        dataset : dataset.Dataset or torch.utils.data.Dataset
            The dataset with which to fine-tune
        """
        pass

    @abstractmethod
    def train_update(self, dataset: Union[datasets.Dataset, torch.utils.data.Dataset]):
        """Fine-tune the classifier on more data tokenized, without resetting

        Parameters
        ----------
        dataset : dataset.Dataset or torch.utils.data.Dataset
            The extra tokenized datapoints with which to fine-tune
        """
        pass

    @abstractmethod
    def tokenize(self, text: Union[str, list]) -> torch.Tensor:
        """Tokenize a string or batch of strings for this classifier

        Parameters
        ----------
        text : str or list
            The string or batch of strings to be tokenized

        Returns
        -------
        tokenized : torch.Tensor
            The result of tokenizing `text`
        """
        return None


class DummyClassifier(Classifier):
    """Dummy classifier, which does nothing"""

    def train_afresh(self, data: Any):
        pass

    def train_update(self, data: Any):
        pass

    def tokenize(self, text: Union[str, list]) -> torch.Tensor:
        if isinstance(text, str):
            return torch.zeros(1)
        elif isinstance(text, list):
            return torch.zeros((1, len(text)))
        else:
            raise TypeError(f"{text!r} is not a string or a list")


class DummyGPT2Classifier(Classifier):
    def __init__(self, parameters: dict):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.parameters = parameters

    def train_afresh(self, data: Any):
        pass

    def train_update(self, data: Any):
        pass

    def tokenize(self, text: Union[str, list]) -> torch.Tensor:
        return self.tokenizer(text)
