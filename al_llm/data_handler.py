# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

import torch
from torch.utils.data import TensorDataset

import datasets


class DataHandler(ABC):
    """Base class for loading and processing the data

    The data handler keeps track of both the raw dataset consisting of
    sentences and labels, and the tokenized version.

    Parameters
    ----------
    classifier : classifier.Classifier
        The classifier instance which will be using the data. We will use this
        to know how to tokenize the data.

    Attributes
    ----------
    dataset : dataset.Dataset
        The raw dataset consisting of labelled sentences, as a Hugging Face
        Dataset.
    tokenized_dataset : torch.utils.data.Dataset
        The tokenized dataset, as a PyTorch dataset.
    classifier : classifier.Classifier
        The classifier instance which will be using the data.
    """

    def __init__(self, classifier):
        self.dataset = None
        self.tokenized_dataset = None
        self.classifier = classifier

    def _tokenize(self, text):
        """Tokenize a string or batch of strings

        Parameters
        ----------
        text : str or list
            The string or batch of strings to be tokenised

        Returns
        -------
        tokenized : torch.Tensor
            The result of tokenizing `text`
        """

        return self.classifier.tokenize(text)

    @abstractmethod
    def new_labelled(self, samples, labels):
        """Add new labelled samples, returning a PyTorch dataset for them

        Parameters
        ----------
        samples : list
            The list of sample strings
        labels : list
            Labels for the samples

        Returns
        -------
        samples_dataset : torch.utils.data.Dataset
            A PyTorch dataset consisting of the newly added tokenized samples
            and their labels, ready for fine-tuning
        """
        return None


class DummyDataHandler(DataHandler):
    def new_labelled(self, samples, labels):
        return TensorDataset(torch.rand(100, 100))


class HuggingFaceDataHandler(DataHandler):
    """A data handler for Hugging Face Datasets

    The data handler keeps track of both the raw dataset consisting of
    sentences and labels, and the tokenized version.

    Parameters
    ----------
    dataset_name : str
        The name of the Hugging Face Dataset
    classifier : classifier.Classifier
        The classifier instance which will be using the data. We will use this
        to know how to tokenize the data.

    Attributes
    ----------
    dataset : dataset.Dataset
        The raw dataset consisting of labelled sentences, as a Hugging Face
        Dataset.
    tokenized_dataset : torch.utils.data.Dataset
        The tokenized dataset, as a PyTorch dataset.
    classifier : classifier.Classifier
        The classifier instance which will be using the data.
    """

    def __init__(self, dataset_name, classifier):
        pass

    def new_labelled(self, samples, labels):
        return TensorDataset(torch.rand(100, 100))
