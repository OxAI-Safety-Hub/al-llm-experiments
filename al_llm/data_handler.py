# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Union

import torch
from torch.utils.data import TensorDataset

import datasets

from al_llm.classifier import Classifier


class DataHandler(ABC):
    """Base class for loading and processing the data

    The data handler keeps track of both the raw dataset consisting of
    sentences and labels, and the tokenized version.

    Parameters
    ----------
    classifier : classifier.Classifier
        The classifier instance which will be using the data. We will use this
        to know how to tokenize the data.
    parameters : dict
        The dictionary of parameters for the present experiment

    Attributes
    ----------
    dataset_train : datasets.Dataset
        The raw dataset consisting of labelled sentences used for training, as
        a Hugging Face Dataset.
    dataset_validation : datasets.Dataset
        The raw dataset consisting of labelled sentences used for validation, as
        a Hugging Face Dataset.
    dataset_test : datasets.Dataset
        The raw dataset consisting of labelled sentences used for testing, as
        a Hugging Face Dataset.
    tokenized_train : torch.utils.data.Dataset
        The tokenized dataset for training, as a PyTorch dataset.
    tokenized_validation : torch.utils.data.Dataset
        The tokenized dataset for validation, as a PyTorch dataset.
    tokenized_test : torch.utils.data.Dataset
        The tokenized dataset for testing, as a PyTorch dataset.
    classifier : classifier.Classifier
        The classifier instance which will be using the data.
    """

    def __init__(self, classifier, parameters):
        self.dataset_train = None
        self.dataset_validation = None
        self.dataset_test = None
        self.tokenized_train = None
        self.tokenized_validation = None
        self.tokenized_test = None
        self.classifier = classifier
        self.parameters = parameters

    def _tokenize(self, text: Union[str, list]) -> torch.Tensor:
        """Tokenize a string or batch of strings

        Parameters
        ----------
        text : str or list
            The string or batch of strings to be tokenized

        Returns
        -------
        tokenized : torch.Tensor
            The result of tokenizing `text`
        """

        return self.classifier.tokenize(text)

    @abstractmethod
    def new_labelled(
        self, samples: list, labels: list
    ) -> Union[datasets.Dataset, torch.utils.data.Dataset]:
        """Add new labelled samples, returning a PyTorch dataset for them

        Parameters
        ----------
        samples : list
            The list of sample strings
        labels : list
            Labels for the samples

        Returns
        -------
        samples_dataset : datasets.Dataset or torch.utils.data.Dataset
            A PyTorch dataset consisting of the newly added tokenized samples
            and their labels, ready for fine-tuning
        """
        return None


class DummyDataHandler(DataHandler):
    def new_labelled(
        self, samples: list, labels: list
    ) -> Union[datasets.Dataset, torch.utils.data.Dataset]:
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
    validation_proportion : float, default=0.2
        Proportion of the training data to be used for validation, if it's not
        provided by the Hugging Face dataset.

    Attributes
    ----------
    dataset_train : datasets.Dataset
        The raw dataset consisting of labelled sentences used for training, as
        a Hugging Face Dataset.
    dataset_validation : datasets.Dataset
        The raw dataset consisting of labelled sentences used for validation, as
        a Hugging Face Dataset.
    dataset_test : datasets.Dataset
        The raw dataset consisting of labelled sentences used for testing, as
        a Hugging Face Dataset.
    tokenized_train : torch.utils.data.Dataset
        The tokenized dataset for training, as a PyTorch dataset.
    tokenized_validation : torch.utils.data.Dataset
        The tokenized dataset for validation, as a PyTorch dataset.
    tokenized_test : torch.utils.data.Dataset
        The tokenized dataset for testing, as a PyTorch dataset.
    classifier : classifier.Classifier
        The classifier instance which will be using the data.
    """

    def __init__(
        self,
        dataset_name: str,
        classifier: Classifier,
        parameters: dict,
        validation_proportion: float = 0.2,
    ):

        self.classifier = classifier
        self.parameters = parameters

        # Make sure that `validation_proportion` is in [0,1]
        if validation_proportion < 0 or validation_proportion > 1:
            raise ValueError("`validation_proportion` should be in [0,1]")

        # get dataset split names and ensure contains only train, test, or
        # validation
        split_names = datasets.get_dataset_split_names(dataset_name)
        for name in split_names:
            assert name in ["train", "test", "validation"]

        # load the training dataset from Hugging Face
        dataset_train_split = datasets.load_dataset(dataset_name, split="train")

        # if the Hugging Face dataset provides a validation set, load it
        if "validation" in split_names:
            self.dataset_validation = datasets.load_dataset(
                dataset_name,
                split="validation",
            )
            self.dataset_train = dataset_train_split

        # otherwise, use the `validation_proportion` to take some of the
        # training set for validation
        else:
            train_length = len(dataset_train_split)
            validation_length = train_length * validation_proportion

            # shuffle the training set before splitting it
            dataset_train_split = dataset_train_split.shuffle()

            # take the last `validation_proportion` elements for validation,
            # and use the remaining elements for training
            self.dataset_validation = dataset_train_split[-validation_length:]
            self.dataset_train = dataset_train_split[:-validation_length]

        # load the testing dataset from Hugging Face
        self.dataset_test = datasets.load_dataset(dataset_name, split="test")

        # slightly altered tokenizing function allows for easy use of
        # dataset `map` method
        def tokenize_function(examples):
            return self.classifier.tokenize(
                examples["text"], padding="max_length", truncation=True
            )

        # to get each tokenized dataset, first map `_tokenize_function` over each
        # of the raw datasets, setting batching to True for efficiency
        self.tokenized_train = self.dataset_train.map(tokenize_function, batched=True)
        self.tokenized_validation = self.dataset_validation.map(
            tokenize_function, batched=True
        )
        self.tokenized_test = self.dataset_test.map(tokenize_function, batched=True)

        # next, rename 'label' to 'labels' (expected by HuggingFace classifiers)
        self.tokenized_train = self.tokenized_train.rename_column("label", "labels")
        self.tokenized_validation = self.tokenized_validation.rename_column(
            "label", "labels"
        )
        self.tokenized_test = self.tokenized_test.rename_column("label", "labels")

        # finally, format all tokenized datasets as PyTorch datasets, keeping
        # only the necessary columns
        self.tokenized_train.set_format(
            "torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
        )
        self.tokenized_validation.set_format(
            "torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
        )
        self.tokenized_test.set_format(
            "torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
        )

    def new_labelled(
        self, samples: list, labels: list
    ) -> Union[datasets.Dataset, torch.utils.data.Dataset]:
        return TensorDataset(torch.rand(100, 100))
