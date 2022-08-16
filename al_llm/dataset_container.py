# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Callable
import configparser
import os

import datasets

from al_llm.parameters import Parameters
from al_llm.utils.fake_data import FakeSentenceGenerator, FakeLabelGenerator


# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


class DatasetContainer(ABC):
    """Base class for dataset containers

    A dataset container which stores the various dataset splits and their
    tokenized versions.

    Parameters
    ----------
    parameters : Parameters
        The parameters for the current experiment

    Attributes
    ----------
    categories : dict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
    dataset_train : datasets.Dataset
        The raw dataset consisting of labelled sentences used for training, as
        a Hugging Face Dataset.
    dataset_validation : datasets.Dataset
        The raw dataset consisting of labelled sentences used for validation, as
        a Hugging Face Dataset.
    dataset_test : datasets.Dataset
        The raw dataset consisting of labelled sentences used for testing, as
        a Hugging Face Dataset.
    tokenized_train : datasets.Dataset
        The tokenized dataset for training, as a PyTorch dataset.
    tokenized_validation : datasets.Dataset
        The tokenized dataset for validation, as a PyTorch dataset.
    tokenized_test : datasets.Dataset
        The tokenized dataset for testing, as a PyTorch dataset.
    """

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.categories = {}
        self.dataset_train = None
        self.dataset_validation = None
        self.dataset_test = None
        self.tokenized_train = None
        self.tokenized_validation = None
        self.tokenized_test = None
        self.orig_train_size = 0

    def make_tokenized(self, tokenizer: Callable):
        """Make tokenized versions of all the dataset splits

        Parameters
        ----------
        tokenizer : Callable
            The tokenizer to use
        """

        # Tokenize each dataset split
        self.tokenized_train = self._tokenize_dataset(self.dataset_train, tokenizer)
        self.tokenized_validation = self._tokenize_dataset(
            self.dataset_validation, tokenizer
        )
        self.tokenized_test = self._tokenize_dataset(self.dataset_test, tokenizer)

    def _tokenize_dataset(
        self, dataset: datasets.Dataset, tokenizer: Callable, batched=True
    ) -> datasets.Dataset:
        """Tokenize a Hugging Face dataset

        Also sets the format to 'torch'.

        Parameters
        ----------
        dataset: datasets.Dataset
            The dataset to tokenize
        tokenizer: Callable
            The function to use for tokenizing
        batched : bool, default=True
            Whether to tokenize in a batch

        Returns
        -------
        tokenized : datasets.Dataset
            The tokenized dataset
        """

        # The function to apply to each datapoint
        def tokenize_function(examples):
            return tokenizer(examples[config["Data Handling"]["TextColumnName"]])

        # Tokenize the dataset
        tokenized = dataset.map(tokenize_function, batched=batched)

        # Set the format to pytorch
        tokenized.set_format(
            "torch",
            columns=[
                "input_ids",
                "attention_mask",
                config["Data Handling"]["LabelColumnName"],
            ],
        )

        return tokenized

    def add_item(self, item: dict, tokenizer: Callable):
        """Add an item to the training set

        Parameters
        ----------
        item : dict
            The item to add, as a dictionary mapping column names to values
        tokenizer: Callable
            The tokenizer with which to create the tokenized versions
        """

        # Use the `add_items` method, turning `item` into a singleton list
        item_as_singleton = {name: [value] for name, value in item.items()}
        self.add_items(item_as_singleton, tokenizer)

    def add_items(self, items: dict, tokenizer: Callable):
        """Add a list of items to the training set

        Parameters
        ----------
        item : dict
            The items to add, as a dictionary mapping column names to values
        tokenizer: Callable
            The tokenizer with which to create the tokenized versions
        """

        # First make a new dataset from the new items
        items_dataset = datasets.Dataset.from_dict(items)

        # Get the tokenized version
        items_tokenized = self._tokenize_dataset(items_dataset, tokenizer)

        # Add these to the training set
        self.dataset_train = datasets.concatenate_datasets(
            [self.dataset_train, items_dataset]
        )
        self.tokenized_train = datasets.concatenate_datasets(
            [self.tokenized_train, items_tokenized]
        )

    @abstractmethod
    def save(self):
        """Save the dataset"""
        pass

    def _preprocess_dataset(self):
        """Do any preprocessing on the dataset, just after it is loaded"""

        # If we're in dev mode, limit the size of the datasets significantly
        if self.parameters["dev_mode"]:
            train_slice_size = min(5, len(self.dataset_train))
            self.dataset_train = self.dataset_train.select(range(train_slice_size))
            validation_slice_size = min(5, len(self.dataset_validation))
            self.dataset_validation = self.dataset_validation.select(
                range(validation_slice_size)
            )
            test_slice_size = min(5, len(self.dataset_test))
            self.dataset_test = self.dataset_test.select(range(test_slice_size))


class DummyDatasetContainer(DatasetContainer):
    """A dummy dataset container

    A dataset container stores the various dataset splits and their tokenized
    versions.

    Parameters
    ----------
    parameters : Parameters
        The parameters for the current experiment

    Attributes
    ----------
    categories : dict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
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
    """

    TRAIN_SIZE = 100
    VALIDATION_SIZE = 20
    TEST_SIZE = 50

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        # Set the categories
        self.categories = {0: "Invalid", 1: "Valid"}

        # Generate some training sentences
        sentence_generator = FakeSentenceGenerator(parameters["seed"])
        train_sentences = sentence_generator.generate(self.TRAIN_SIZE)
        validation_sentences = sentence_generator.generate(self.VALIDATION_SIZE)
        test_sentences = sentence_generator.generate(self.TEST_SIZE)

        # Generate the class labels
        label_generator = FakeLabelGenerator(
            list(self.categories.keys()), parameters["seed"]
        )
        train_labels = label_generator.generate(self.TRAIN_SIZE)
        validation_labels = label_generator.generate(self.VALIDATION_SIZE)
        test_labels = label_generator.generate(self.TEST_SIZE)

        # Compose everything to make the datasets
        self.dataset_train = datasets.Dataset.from_dict(
            {
                config["Data Handling"]["TextColumnName"]: train_sentences,
                config["Data Handling"]["LabelColumnName"]: train_labels,
            }
        )
        self.dataset_validation = datasets.Dataset.from_dict(
            {
                config["Data Handling"]["TextColumnName"]: validation_sentences,
                config["Data Handling"]["LabelColumnName"]: validation_labels,
            }
        )
        self.dataset_test = datasets.Dataset.from_dict(
            {
                config["Data Handling"]["TextColumnName"]: test_sentences,
                config["Data Handling"]["LabelColumnName"]: test_labels,
            }
        )

    def save(self):
        pass


class HuggingFaceDatasetContainer(DatasetContainer, ABC):
    """A base dataset container for Hugging Face datasets

    A dataset container which stores the various dataset splits and their
    tokenized versions.

    Parameters
    ----------
    parameters : Parameters
        The parameters for the current experiment

    Attributes
    ----------
    categories : dict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
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
    """

    DATASET_NAME = ""
    CATEGORIES = {}

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        # Set the categories for this dataset
        self.categories = self.CATEGORIES

        # Download the datasets
        self.dataset_train = datasets.load_dataset(self.DATASET_NAME, split="train")
        self.dataset_validation = datasets.load_dataset(
            self.DATASET_NAME, split="validation"
        )
        self.dataset_test = datasets.load_dataset(self.DATASET_NAME, split="test")

        # Do any preprocessing on the dataset
        self._preprocess_dataset()

    def save(self):
        pass


class LocalDatasetContainer(DatasetContainer, ABC):
    """A base dataset container for local datasets

    A dataset container which stores the various dataset splits and their
    tokenized versions.

    Parameters
    ----------
    parameters : Parameters
        The parameters for the current experiment

    Attributes
    ----------
    categories : dict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
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
    """

    CATEGORIES = {}
    DATASET_NAME = ""

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        # Set the categories for this dataset
        self.categories = self.CATEGORIES

        # load the local dataset, splitting by the data file names
        data_files = {
            "train": "train.csv",
            "validation": "evaluation.csv",
            "test": "test.csv",
        }
        dataset_path = os.path.join(
            config["Data Handling"]["LocalDatasetDir"], self.DATASET_NAME
        )
        dataset_dictionary = datasets.load_dataset(dataset_path, data_files=data_files)

        # use this split to store the raw datasets
        self.dataset_train = dataset_dictionary["train"]
        self.dataset_validation = dataset_dictionary["validation"]
        self.dataset_test = dataset_dictionary["test"]

        # Do any preprocessing on the dataset
        self._preprocess_dataset()

    def save(self):
        pass


class RottenTomatoesDatasetHandler(HuggingFaceDatasetContainer):
    """A dataset container for the rotten tomatoes dataset

    A dataset container which stores the various dataset splits and their
    tokenized versions.

    Parameters
    ----------
    parameters : Parameters
        The parameters for the current experiment

    Attributes
    ----------
    categories : dict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
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
    """

    DATASET_NAME = "rotten_tomatoes"
    CATEGORIES = {0: "Negative", 1: "Positive"}

    def _preprocess_dataset(self):

        # Do any preprocessing defined by the base class
        super()._preprocess_dataset()

        # Rename the 'label' column
        self.dataset_train = self.dataset_train.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_test = self.dataset_test.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )


class DummyLocalDatasetContainer(LocalDatasetContainer):
    """A dataset container for a dummy local dataset

    A dataset container which stores the various dataset splits and their
    tokenized versions.

    Parameters
    ----------
    parameters : Parameters
        The parameters for the current experiment

    Attributes
    ----------
    categories : dict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
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
    """

    DATASET_NAME = "dummy_local_dataset"
    CATEGORIES = {0: "Negative", 1: "Positive"}
