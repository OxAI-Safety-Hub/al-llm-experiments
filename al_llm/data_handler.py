# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Union
import configparser

import torch
from torch.utils.data import TensorDataset

import datasets
import wandb

from al_llm.classifier import Classifier
from al_llm.parameters import Parameters


# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


class DataHandler(ABC):
    """Base class for loading and processing the data

    The data handler keeps track of both the raw dataset consisting of
    sentences and labels, and the tokenized version.

    Parameters
    ----------
    classifier : classifier.Classifier
        The classifier instance which will be using the data. We will use this
        to know how to tokenize the data.
    parameters : Parameters
        The dictionary of parameters for the present experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run

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
        classifier: Classifier,
        parameters: Parameters,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):
        self.dataset_train = None
        self.dataset_validation = None
        self.dataset_test = None
        self.tokenized_train = None
        self.tokenized_validation = None
        self.tokenized_test = None
        self.classifier = classifier
        self.parameters = parameters
        self.wandb_run = wandb_run

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

    def get_latest_tokenized_datapoints(
        self,
    ) -> Union[datasets.Dataset, torch.utils.data.Dataset]:
        """Get the most recently added datapoints, obtained from the human

        Returns
        -------
        tokenized_samples : datasets.Dataset or torch.utils.data.Dataset
            The latest datapoints
        """

        # return the last `num_samples`entries from `tokenized_train`
        # (because adding items puts them at the end of the dataset)
        samples_dict = self.tokenized_train[-self.parameters["num_samples"] :]
        tokenized_samples = datasets.Dataset.from_dict(samples_dict)
        tokenized_samples.set_format("torch")
        return tokenized_samples

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

        # store the number of new samples being processed
        num_samples = len(samples)

        # iterate over each sample-label pair
        for i in range(num_samples):

            # store the original sample text and label
            sample_text = samples[i]
            label = labels[i]

            # add this sample-label pair to the raw `dataset_train`
            self.dataset_train = self.dataset_train.add_item(
                {
                    config["Data Handling"]["TextColumnName"]: sample_text,
                    config["Data Handling"]["LabelColumnName"]: label,
                }
            )

            # tokenize the `sample_text` using the classifier's tokenizer
            sample_tokenized = self.classifier.tokenize(sample_text)

            # add this tokenized sample-label triple to `tokenized_train`
            self.tokenized_train = self.tokenized_train.add_item(
                {
                    "input_ids": sample_tokenized["input_ids"],
                    "attention_mask": sample_tokenized["attention_mask"],
                    config["Data Handling"]["LabelColumnName"]: label,
                }
            )

        # having added all the new samples, return the last `num_samples`
        # entries from `tokenized_train` (because adding items puts them at
        # the end of the dataset)
        samples_dict = self.tokenized_train[-num_samples:]
        return datasets.Dataset.from_dict(samples_dict)

    @abstractmethod
    def make_label_request(self, samples: list):
        """Make a request for labels for the samples from the human

        Parameters
        ----------
        samples : list
            The sample sentences for which to get the labels
        """
        pass

    @abstractmethod
    def save(self):
        """Save the current dataset"""
        pass


class DummyDataHandler(DataHandler):
    """A dummy data handler, which holds a dummy dataset"""

    def new_labelled(
        self, samples: list, labels: list
    ) -> Union[datasets.Dataset, torch.utils.data.Dataset]:
        return TensorDataset(torch.rand(100, 100))

    def get_latest_tokenized_datapoints(
        self,
    ) -> Union[datasets.Dataset, torch.utils.data.Dataset]:
        return TensorDataset(torch.rand(100, 20))

    def make_label_request(self, samples: list):
        pass

    def save(self):
        pass


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
    parameters : Parameters
        The dictionary of parameters for the present experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
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
        parameters: Parameters,
        wandb_run: wandb.sdk.wandb_run.Run,
        validation_proportion: float = 0.2,
    ):

        super().__init__(classifier, parameters, wandb_run)

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

            # make sure to convert to HuggingFace datasets, since splicing
            # operation above returns Python dicts
            self.dataset_validation = datasets.Dataset.from_dict(
                self.dataset_validation
            )
            self.dataset_train = datasets.Dataset.from_dict(self.dataset_train)

        # load the testing dataset from Hugging Face
        self.dataset_test = datasets.load_dataset(dataset_name, split="test")

        # next, rename 'label' to 'labels' (expected by some HuggingFace
        # classifiers - MORE RESEARCH NEEDED)
        self.dataset_train = self.dataset_train.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_test = self.dataset_test.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )

        # slightly altered tokenizing function allows for easy use of
        # dataset `map` method
        def tokenize_function(examples):
            return self.classifier.tokenize(
                examples[config["Data Handling"]["TextColumnName"]]
            )

        # to get each tokenized dataset, first map `tokenize_function` over each
        # of the raw datasets, setting batching to True for efficiency
        self.tokenized_train = self.dataset_train.map(tokenize_function, batched=True)
        self.tokenized_validation = self.dataset_validation.map(
            tokenize_function, batched=True
        )
        self.tokenized_test = self.dataset_test.map(tokenize_function, batched=True)

        # finally, format all tokenized datasets as PyTorch datasets, keeping
        # only the necessary columns
        self.tokenized_train.set_format(
            "torch",
            columns=[
                "input_ids",
                "attention_mask",
                config["Data Handling"]["LabelColumnName"],
            ],
        )
        self.tokenized_validation.set_format(
            "torch",
            columns=[
                "input_ids",
                "attention_mask",
                config["Data Handling"]["LabelColumnName"],
            ],
        )
        self.tokenized_test.set_format(
            "torch",
            columns=[
                "input_ids",
                "attention_mask",
                config["Data Handling"]["LabelColumnName"],
            ],
        )

        # if within a dummy experiment (checked by self.parameters["dev_mode"]),
        # limit the size of the datasets significantly
        if self.parameters["dev_mode"]:
            self.tokenized_train = self.tokenized_train.shuffle(seed=1091).select(
                range(5)
            )
            self.tokenized_validation = self.tokenized_validation.shuffle(
                seed=1091
            ).select(range(100))
            self.tokenized_test = self.tokenized_test.shuffle(seed=1091).select(
                range(100)
            )

    def make_label_request(self, samples: list):
        pass

    def save(self):
        pass


class LocalDataHandler(DataHandler):
    """A data handler for datasets that are stored locally.

    In the dataset_path folder, three files must exist which hold the data:
        {"train.csv", "evaluation.csv", "test.cs"} (see README.csv for more info)

    The data handler keeps track of both the raw dataset consisting of
    sentences and labels, and the tokenized version.

    Parameters
    ----------
    dataset_path : str
        The path of the file containing {"train.csv", "evaluation.csv", "test.cs"}
    classifier : classifier.Classifier
        The classifier instance which will be using the data. We will use this
        to know how to tokenize the data.
    parameters : Parameters
        The dictionary of parameters for the present experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run

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
        a Hugging Face dataset.
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
        dataset_path: str,
        classifier: Classifier,
        parameters: Parameters,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):

        super().__init__(classifier, parameters, wandb_run)

        # load the local dataset, splitting by the data file names
        data_files = {
            "train": "train.csv",
            "validation": "evaluation.csv",
            "test": "test.csv",
        }
        dataset_dictionary = datasets.load_dataset(dataset_path, data_files=data_files)

        # use this split to store the raw datasets
        self.dataset_train = dataset_dictionary["train"]
        self.dataset_validation = dataset_dictionary["validation"]
        self.dataset_test = dataset_dictionary["test"]

        # slightly altered tokenizing function allows for easy use of
        # dataset `map` method
        def tokenize_function(examples):
            return self.classifier.tokenize(
                examples[config["Data Handling"]["TextColumnName"]]
            )

        # to get each tokenized dataset, first map `tokenize_function` over each
        # of the raw datasets, setting batching to True for efficiency
        self.tokenized_train = self.dataset_train.map(tokenize_function, batched=True)
        self.tokenized_validation = self.dataset_validation.map(
            tokenize_function, batched=True
        )
        self.tokenized_test = self.dataset_test.map(tokenize_function, batched=True)

        # finally, format all tokenized datasets as PyTorch datasets, keeping
        # only the necessary columns
        self.tokenized_train.set_format(
            "torch",
            columns=[
                "input_ids",
                "attention_mask",
                config["Data Handling"]["LabelColumnName"],
            ],
        )
        self.tokenized_validation.set_format(
            "torch",
            columns=[
                "input_ids",
                "attention_mask",
                config["Data Handling"]["LabelColumnName"],
            ],
        )
        self.tokenized_test.set_format(
            "torch",
            columns=[
                "input_ids",
                "attention_mask",
                config["Data Handling"]["LabelColumnName"],
            ],
        )

        self.tokenized_train.remove_columns([config["Data Handling"]["TextColumnName"]])
        self.tokenized_validation.remove_columns(
            [config["Data Handling"]["TextColumnName"]]
        )
        self.tokenized_test.remove_columns([config["Data Handling"]["TextColumnName"]])

    def make_label_request(self, samples: list):
        pass

    def save(self):
        pass
