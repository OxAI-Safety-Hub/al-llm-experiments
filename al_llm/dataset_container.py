# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Callable, Tuple
import configparser
from collections import OrderedDict

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
    categories : OrderedDict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
    dataset_train : datasets.Dataset
        The raw dataset consisting of labelled sentences used for training, as
        a Hugging Face Dataset. This is separated from the 'train' split of
        the dataset by selecting `parameters["train_dataset_size"]` datapoints.
    dataset_remainder : datasets.Dataset
        The remainder of the 'train' split after `dataset_train` has been
        selected. Used by the pool-based simulator.
    dataset_validation : datasets.Dataset
        The raw dataset consisting of labelled sentences used for validation, as
        a Hugging Face Dataset.
    dataset_test : datasets.Dataset
        The raw dataset consisting of labelled sentences used for testing, as
        a Hugging Face Dataset.
    tokenized_train : datasets.Dataset
        A tokenized version of `dataset_train`, consisting of PyTorch tensors.
    tokenized_remainder : datasets.Dataset
        A tokenized version of `dataset_remainder`, consisting of PyTorch
        tensors.
    tokenized_validation : datasets.Dataset
        A tokenized version of `dataset_validation`, consisting of PyTorch
        tensors.
    tokenized_test : datasets.Dataset
        A tokenized version of `dataset_test`, consisting of PyTorch tensors.
    """

    CATEGORIES = OrderedDict()

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.categories = self.CATEGORIES
        self.dataset_train = None
        self.dataset_remainder = None
        self.dataset_validation = None
        self.dataset_test = None
        self.tokenized_train = None
        self.tokenized_remainder = None
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
        self.tokenized_remainder = self._tokenize_dataset(
            self.dataset_remainder, tokenizer
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

    def _train_remainder_split(
        self, train_split: datasets.Dataset
    ) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """Split a dataset 'train' split into a train and remainder dataset

        We select `parameters["train_dataset_size"]` datapoints and set them
        as a train dataset, the rest going to a remainder dataset.

        Parameters
        ----------
        train_split : datasets.Dataset
            The train split of a dataset, ready for separating.

        Returns
        -------
        train_dataset : datasets.Dataset
            A train dataset of size at most `parameters["train_dataset_size"]`,
            selected from `train_split`.
        remainder_dataset : datasets.Dataset
            The remainder of the train split.
        """

        if self.parameters["supervised"]:
            self.parameters["train_dataset_size"] = len(train_split) - 1

        if len(train_split) < self.parameters["train_dataset_size"]:
            raise ValueError(
                f"Train split must be larger than train dataset size (currently"
                f" {len(train_split)} < {self.parameters['train_dataset_size']})"
            )

        # Shuffle the train split
        seed = int(config["Data Handling"]["PreprocessingSeed"])
        train_split = train_split.shuffle(seed=seed)

        # Select the train and remainder datasets
        train_range = range(self.parameters["train_dataset_size"])
        train_dataset = train_split.select(train_range)
        remainder_range = range(self.parameters["train_dataset_size"], len(train_split))
        remainder_dataset = train_split.select(remainder_range)

        return train_dataset, remainder_dataset

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

        # Get the required structure of the datasets as a datasets.Features object
        features = self._get_ambiguous_dataset_features()

        # First make a new dataset from the new items
        items_dataset = datasets.Dataset.from_dict(items, features=features)

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

        # Add an ambiguities column to the train dataset
        #   All of this data will not be ambiguous
        self.dataset_train = self.dataset_train.add_column(
            config["Data Handling"]["AmbiguitiesColumnName"],
            [0] * len(self.dataset_train),
        )

    def _get_dataset_features(self) -> datasets.Features:
        """Get the internal structure of the (non-tokenized) dataset

        Returns
        -------
        features : datasets.Features
            A mapping which specifies the types of the text and label columns
        """
        text_type = datasets.Value(dtype="string")
        label_type = datasets.ClassLabel(names=list(self.categories.keys()))
        features = datasets.Features(
            {
                config["Data Handling"]["TextColumnName"]: text_type,
                config["Data Handling"]["LabelColumnName"]: label_type,
            }
        )
        return features

    def _get_ambiguous_dataset_features(self) -> datasets.Features:
        """Get the internal structure of the (non-tokenized) dataset

        Returns
        -------
        features : datasets.Features
            A mapping which specifies the types of the text and label columns
        """
        text_type = datasets.Value(dtype="string")
        label_type = datasets.ClassLabel(names=list(self.categories.keys()))
        ambiguity_type = datasets.Value(dtype="int64")
        features = datasets.Features(
            {
                config["Data Handling"]["TextColumnName"]: text_type,
                config["Data Handling"]["LabelColumnName"]: label_type,
                config["Data Handling"]["AmbiguitiesColumnName"]: ambiguity_type,
            }
        )
        return features


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
    categories : OrderedDict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
    dataset_train : datasets.Dataset
        The raw dataset consisting of labelled sentences used for training, as
        a Hugging Face Dataset. This is separated from the 'train' split of
        the dataset by selecting `parameters["train_dataset_size"]` datapoints.
    dataset_remainder : datasets.Dataset
        The remainder of the 'train' split after `dataset_train` has been
        selected. Used by the pool-based simulator.
    dataset_validation : datasets.Dataset
        The raw dataset consisting of labelled sentences used for validation, as
        a Hugging Face Dataset.
    dataset_test : datasets.Dataset
        The raw dataset consisting of labelled sentences used for testing, as
        a Hugging Face Dataset.
    tokenized_train : datasets.Dataset
        A tokenized version of `dataset_train`, consisting of PyTorch tensors.
    tokenized_remainder : datasets.Dataset
        A tokenized version of `dataset_remainder`, consisting of PyTorch
        tensors.
    tokenized_validation : datasets.Dataset
        A tokenized version of `dataset_validation`, consisting of PyTorch
        tensors.
    tokenized_test : datasets.Dataset
        A tokenized version of `dataset_test`, consisting of PyTorch tensors.
    """

    CATEGORIES = OrderedDict([("inv", "Invalid"), ("val", "Valid")])

    REMAINDER_SIZE = 10
    VALIDATION_SIZE = 20
    TEST_SIZE = 50

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        # Generate some training sentences
        sentence_generator = FakeSentenceGenerator(parameters["seed"])
        train_sentences = sentence_generator.generate(
            parameters["train_dataset_size"] + self.REMAINDER_SIZE
        )
        validation_sentences = sentence_generator.generate(self.VALIDATION_SIZE)
        test_sentences = sentence_generator.generate(self.TEST_SIZE)

        # Generate the class labels
        label_generator = FakeLabelGenerator(
            list(self.categories.keys()), parameters["seed"]
        )
        train_labels = label_generator.generate(
            parameters["train_dataset_size"] + self.REMAINDER_SIZE
        )
        validation_labels = label_generator.generate(self.VALIDATION_SIZE)
        test_labels = label_generator.generate(self.TEST_SIZE)

        # Get the required structure of the datasets as a datasets.Features object
        features = self._get_dataset_features()

        # Compose everything to make the datasets
        train_split = datasets.Dataset.from_dict(
            {
                config["Data Handling"]["TextColumnName"]: train_sentences,
                config["Data Handling"]["LabelColumnName"]: train_labels,
            },
            features=features,
        )
        self.dataset_validation = datasets.Dataset.from_dict(
            {
                config["Data Handling"]["TextColumnName"]: validation_sentences,
                config["Data Handling"]["LabelColumnName"]: validation_labels,
            },
            features=features,
        )
        self.dataset_test = datasets.Dataset.from_dict(
            {
                config["Data Handling"]["TextColumnName"]: test_sentences,
                config["Data Handling"]["LabelColumnName"]: test_labels,
            },
            features=features,
        )

        # Divide the train split into train and remainder datasets
        self.dataset_train, self.dataset_remainder = self._train_remainder_split(
            train_split
        )

        # Add an ambiguities column to the train dataset
        #   All of this data will not be ambiguous
        self.dataset_train = self.dataset_train.add_column(
            config["Data Handling"]["AmbiguitiesColumnName"],
            [0] * len(self.dataset_train),
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
    dataset_train : datasets.Dataset
        The raw dataset consisting of labelled sentences used for training, as
        a Hugging Face Dataset. This is separated from the 'train' split of
        the dataset by selecting `parameters["train_dataset_size"]` datapoints.
    dataset_remainder : datasets.Dataset
        The remainder of the 'train' split after `dataset_train` has been
        selected. Used by the pool-based simulator.
    dataset_validation : datasets.Dataset
        The raw dataset consisting of labelled sentences used for validation, as
        a Hugging Face Dataset.
    dataset_test : datasets.Dataset
        The raw dataset consisting of labelled sentences used for testing, as
        a Hugging Face Dataset.
    tokenized_train : datasets.Dataset
        A tokenized version of `dataset_train`, consisting of PyTorch tensors.
    tokenized_remainder : datasets.Dataset
        A tokenized version of `dataset_remainder`, consisting of PyTorch
        tensors.
    tokenized_validation : datasets.Dataset
        A tokenized version of `dataset_validation`, consisting of PyTorch
        tensors.
    tokenized_test : datasets.Dataset
        A tokenized version of `dataset_test`, consisting of PyTorch tensors.
    """

    DATASET_NAME = ""
    CATEGORIES = OrderedDict()

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        # Download the datasets
        train_split = datasets.load_dataset(self.DATASET_NAME, split="train")
        self.dataset_validation = datasets.load_dataset(
            self.DATASET_NAME, split="validation"
        )
        self.dataset_test = datasets.load_dataset(self.DATASET_NAME, split="test")

        # Divide the train split into train and remainder datasets
        self.dataset_train, self.dataset_remainder = self._train_remainder_split(
            train_split
        )

        # Do any preprocessing on the dataset
        self._preprocess_dataset()

    def save(self):
        pass


class RottenTomatoesDatasetContainer(HuggingFaceDatasetContainer):
    """A dataset container for the rotten tomatoes dataset

    A dataset container which stores the various dataset splits and their
    tokenized versions.

    Parameters
    ----------
    parameters : Parameters
        The parameters for the current experiment

    Attributes
    ----------
    categories : OrderedDict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
    dataset_train : datasets.Dataset
        The raw dataset consisting of labelled sentences used for training, as
        a Hugging Face Dataset. This is separated from the 'train' split of
        the dataset by selecting `parameters["train_dataset_size"]` datapoints.
    dataset_remainder : datasets.Dataset
        The remainder of the 'train' split after `dataset_train` has been
        selected. Used by the pool-based simulator.
    dataset_validation : datasets.Dataset
        The raw dataset consisting of labelled sentences used for validation, as
        a Hugging Face Dataset.
    dataset_test : datasets.Dataset
        The raw dataset consisting of labelled sentences used for testing, as
        a Hugging Face Dataset.
    tokenized_train : datasets.Dataset
        A tokenized version of `dataset_train`, consisting of PyTorch tensors.
    tokenized_remainder : datasets.Dataset
        A tokenized version of `dataset_remainder`, consisting of PyTorch
        tensors.
    tokenized_validation : datasets.Dataset
        A tokenized version of `dataset_validation`, consisting of PyTorch
        tensors.
    tokenized_test : datasets.Dataset
        A tokenized version of `dataset_test`, consisting of PyTorch tensors.
    """

    DATASET_NAME = "rotten_tomatoes"
    CATEGORIES = OrderedDict([("neg", "Negative"), ("pos", "Positive")])

    def _preprocess_dataset(self):

        # Do any preprocessing defined by the base class
        super()._preprocess_dataset()

        # Rename the 'label' column
        self.dataset_train = self.dataset_train.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_remainder = self.dataset_remainder.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_test = self.dataset_test.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )


class WikiToxicDatasetContainer(HuggingFaceDatasetContainer):
    """A container for the Jigsaw Toxic Comment Challenge dataset

    This dataset was the basis of a Kaggle competition run by Jigsaw. [1]_

    The dataset has been modified in the following ways.
    - The comment texts have been cleaned, removing user signatures timestamps,
    IP addresses, pieces of code and stray quotation marks.
    - The six toxicity classifications have been combined into a single binary
    classification.

    Parameters
    ----------
    parameters : Parameters
        The parameters for the current experiment

    Attributes
    ----------
    categories : OrderedDict
        A dictionary of the classes in the data. The keys are the names of the
        categories as understood by the model, and the values are the
        human-readable names.
    dataset_train : datasets.Dataset
        The raw dataset consisting of labelled sentences used for training, as
        a Hugging Face Dataset. This is separated from the 'train' split of
        the dataset by selecting `parameters["train_dataset_size"]` datapoints.
    dataset_remainder : datasets.Dataset
        The remainder of the 'train' split after `dataset_train` has been
        selected. Used by the pool-based simulator.
    dataset_validation : datasets.Dataset
        The raw dataset consisting of labelled sentences used for validation, as
        a Hugging Face Dataset.
    dataset_test : datasets.Dataset
        The raw dataset consisting of labelled sentences used for testing, as
        a Hugging Face Dataset.
    tokenized_train : datasets.Dataset
        A tokenized version of `dataset_train`, consisting of PyTorch tensors.
    tokenized_remainder : datasets.Dataset
        A tokenized version of `dataset_remainder`, consisting of PyTorch
        tensors.
    tokenized_validation : datasets.Dataset
        A tokenized version of `dataset_validation`, consisting of PyTorch
        tensors.
    tokenized_test : datasets.Dataset
        A tokenized version of `dataset_test`, consisting of PyTorch tensors.

    References
    ----------
    [1] Victor et al., "Toxic Comment Classification Challenge", Kaggle Competition
    https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data,
    2019
    """

    DATASET_NAME = "OxAISH-AL-LLM/wiki_toxic"
    CATEGORIES = OrderedDict([("non", "Non-toxic"), ("tox", "Toxic")])

    def _preprocess_dataset(self):

        # Do any preprocessing defined by the base class
        super()._preprocess_dataset()

        # Remove the 'id' column
        self.dataset_train = self.dataset_train.remove_columns("id")
        self.dataset_remainder = self.dataset_remainder.remove_columns("id")
        self.dataset_validation = self.dataset_validation.remove_columns("id")
        self.dataset_test = self.dataset_test.remove_columns("id")

        # Rename the 'comment_text' column
        self.dataset_train = self.dataset_train.rename_column(
            "comment_text", config["Data Handling"]["TextColumnName"]
        )
        self.dataset_remainder = self.dataset_remainder.rename_column(
            "comment_text", config["Data Handling"]["TextColumnName"]
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "comment_text", config["Data Handling"]["TextColumnName"]
        )
        self.dataset_test = self.dataset_test.rename_column(
            "comment_text", config["Data Handling"]["TextColumnName"]
        )

        # Recast the 'label' column so it uses ClassLabels instead of Values
        new_train_features = self.dataset_train.features.copy()
        new_remainder_features = self.dataset_remainder.features.copy()
        new_validation_features = self.dataset_validation.features.copy()
        new_test_features = self.dataset_test.features.copy()

        new_train_features["label"] = datasets.ClassLabel(
            names=list(self.CATEGORIES.keys())
        )
        new_remainder_features["label"] = datasets.ClassLabel(
            names=list(self.CATEGORIES.keys())
        )
        new_validation_features["label"] = datasets.ClassLabel(
            names=list(self.CATEGORIES.keys())
        )
        new_test_features["label"] = datasets.ClassLabel(
            names=list(self.CATEGORIES.keys())
        )

        self.dataset_train = self.dataset_train.cast(new_train_features)
        self.dataset_remainder = self.dataset_remainder.cast(new_remainder_features)
        self.dataset_validation = self.dataset_validation.cast(new_validation_features)
        self.dataset_test = self.dataset_test.cast(new_test_features)

        # Rename the 'label' column
        self.dataset_train = self.dataset_train.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_remainder = self.dataset_remainder.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
        self.dataset_test = self.dataset_test.rename_column(
            "label", config["Data Handling"]["LabelColumnName"]
        )
