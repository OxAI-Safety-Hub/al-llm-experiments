# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from collections import OrderedDict

import datasets

from al_llm.parameters import Parameters
from al_llm.utils.fake_data import FakeSentenceGenerator, FakeLabelGenerator
from al_llm.constants import (
    TEXT_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    AMBIGUITIES_COLUMN_NAME,
    SKIPS_COLUMN_NAME,
    PREPROCESSING_SEED,
)


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
        print("Making tokenized train, val, remainder and test")
        self.tokenized_train = self._tokenize_dataset(
            self.dataset_train,
            tokenizer,
            columns=[
                "input_ids",
                "attention_mask",
                LABEL_COLUMN_NAME,
                SKIPS_COLUMN_NAME,
            ],
        )
        self.tokenized_validation = self._tokenize_dataset(
            self.dataset_validation,
            tokenizer,
            columns=[
                "input_ids",
                "attention_mask",
                LABEL_COLUMN_NAME,
            ],
        )
        self.tokenized_remainder = self._tokenize_dataset(
            self.dataset_remainder,
            tokenizer,
            columns=[
                "input_ids",
                "attention_mask",
                LABEL_COLUMN_NAME,
            ],
        )
        self.tokenized_test = self._tokenize_dataset(
            self.dataset_test,
            tokenizer,
            columns=[
                "input_ids",
                "attention_mask",
                LABEL_COLUMN_NAME,
            ],
        )

    def _tokenize_dataset(
        self,
        dataset: datasets.Dataset,
        tokenizer: Callable,
        columns: list,
        batched=True,
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
            return tokenizer(examples[TEXT_COLUMN_NAME])

        # Tokenize the dataset
        tokenized = dataset.map(
            tokenize_function, batched=batched, desc="Tokenizing dataset"
        )

        # Set the format to pytorch
        tokenized.set_format("torch", columns=columns)

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

        if len(train_split) < self.parameters["train_dataset_size"]:
            raise ValueError(
                f"Train split must be larger than train dataset size (currently"
                f" {len(train_split)} < {self.parameters['train_dataset_size']})"
            )

        # Shuffle the train split
        seed = int(PREPROCESSING_SEED)
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
        features = self._get_full_dataset_features()

        # First make a new dataset from the new items
        items_dataset = datasets.Dataset.from_dict(items, features=features)

        # Get the tokenized version
        items_tokenized = self._tokenize_dataset(
            items_dataset,
            tokenizer,
            columns=[
                "input_ids",
                "attention_mask",
                LABEL_COLUMN_NAME,
                SKIPS_COLUMN_NAME,
            ],
        )

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
            train_slice_size = min(20, len(self.dataset_train))
            self.dataset_train = self.dataset_train.select(range(train_slice_size))
            validation_slice_size = min(20, len(self.dataset_validation))
            self.dataset_validation = self.dataset_validation.select(
                range(validation_slice_size)
            )
            test_slice_size = min(20, len(self.dataset_test))
            self.dataset_test = self.dataset_test.select(range(test_slice_size))

        # Add an ambiguities and skip mask columns to the train dataset. All
        # of this data will unambiguous and non-skipped
        self.dataset_train = self.dataset_train.add_column(
            AMBIGUITIES_COLUMN_NAME,
            [0] * len(self.dataset_train),
        )
        self.dataset_train = self.dataset_train.add_column(
            SKIPS_COLUMN_NAME,
            [0] * len(self.dataset_train),
        )

    def _get_basic_dataset_features(self) -> datasets.Features:
        """Get the basic internal structure of the (non-tokenized) dataset

        This includes only the sample text and label.

        Returns
        -------
        features : datasets.Features
            A mapping which specifies the types of the text and label columns
        """
        text_type = datasets.Value(dtype="string")
        label_type = datasets.ClassLabel(names=list(self.categories.keys()))
        features = datasets.Features(
            {
                TEXT_COLUMN_NAME: text_type,
                LABEL_COLUMN_NAME: label_type,
            }
        )
        return features

    def _get_full_dataset_features(self) -> datasets.Features:
        """Get the full internal structure of the (non-tokenized) dataset

        This includes all features recorded during the experiment: the sample
        text, label, ambiguity value and the skip mask.

        Returns
        -------
        features : datasets.Features
            A mapping which specifies the types of the text and label columns
        """
        text_type = datasets.Value(dtype="string")
        label_type = datasets.ClassLabel(names=list(self.categories.keys()))
        ambiguity_type = datasets.Value(dtype="int64")
        skip_mask_type = datasets.Value(dtype="int64")
        features = datasets.Features(
            {
                TEXT_COLUMN_NAME: text_type,
                LABEL_COLUMN_NAME: label_type,
                AMBIGUITIES_COLUMN_NAME: ambiguity_type,
                SKIPS_COLUMN_NAME: skip_mask_type,
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
        features = self._get_basic_dataset_features()

        # Compose everything to make the datasets
        train_split = datasets.Dataset.from_dict(
            {
                TEXT_COLUMN_NAME: train_sentences,
                LABEL_COLUMN_NAME: train_labels,
            },
            features=features,
        )
        self.dataset_validation = datasets.Dataset.from_dict(
            {
                TEXT_COLUMN_NAME: validation_sentences,
                LABEL_COLUMN_NAME: validation_labels,
            },
            features=features,
        )
        self.dataset_test = datasets.Dataset.from_dict(
            {
                TEXT_COLUMN_NAME: test_sentences,
                LABEL_COLUMN_NAME: test_labels,
            },
            features=features,
        )

        # Divide the train split into train and remainder datasets
        self.dataset_train, self.dataset_remainder = self._train_remainder_split(
            train_split
        )

        # Add an ambiguities and skip mask columns to the train dataset. All
        # of this data will unambiguous and non-skipped
        self.dataset_train = self.dataset_train.add_column(
            AMBIGUITIES_COLUMN_NAME,
            [0] * len(self.dataset_train),
        )
        self.dataset_train = self.dataset_train.add_column(
            SKIPS_COLUMN_NAME,
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
    TOKENIZED_LENGTH_UPPER_QUARTILE = 0

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
    TOKENIZED_LENGTH_UPPER_QUARTILE = 32

    def _preprocess_dataset(self):
        # Do any preprocessing defined by the base class
        super()._preprocess_dataset()

        # Rename the 'label' column
        self.dataset_train = self.dataset_train.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_remainder = self.dataset_remainder.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_test = self.dataset_test.rename_column("label", LABEL_COLUMN_NAME)


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
    TOKENIZED_LENGTH_UPPER_QUARTILE = 102

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
            "comment_text", TEXT_COLUMN_NAME
        )
        self.dataset_remainder = self.dataset_remainder.rename_column(
            "comment_text", TEXT_COLUMN_NAME
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "comment_text", TEXT_COLUMN_NAME
        )
        self.dataset_test = self.dataset_test.rename_column(
            "comment_text", TEXT_COLUMN_NAME
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
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_remainder = self.dataset_remainder.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_test = self.dataset_test.rename_column("label", LABEL_COLUMN_NAME)


class PubMed20kRCTDatasetContainer(HuggingFaceDatasetContainer):
    """A container for the PubMed 20k RCT dataset

    This dataset was created by taking abstracts from medical journal
    publications and classifying each sentence as either background, objective,
    methods, results or conclusions. [1]_

    The dataset has been modified in the following ways.
    - A paired <span> html tag has been removed from one datapoint in the
    train dataset
    - The datapoints have been shuffled, and are thus no longer grouped by
    abstract.

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
    [1] Franck Dernoncourt and Ji Young Lee, "PubMed 200k RCT: a Dataset for
    Sequential Sentence Classification in Medical Abstracts", Proceedings of
    the Eighth International Joint Conference on Natural Language Processing
    (Volume 2: Short Papers), Asian Federation of Natural Language Processing,
    2017
    """

    DATASET_NAME = "OxAISH-AL-LLM/pubmed_20k_rct"
    CATEGORIES = OrderedDict(
        [
            ("bac", "Background"),
            ("obj", "Objective"),
            ("met", "Methods"),
            ("res", "Results"),
            ("con", "Conclusions"),
        ]
    )
    TOKENIZED_LENGTH_UPPER_QUARTILE = 42

    def _preprocess_dataset(self):
        # Do any preprocessing defined by the base class
        super()._preprocess_dataset()

        # Rename the 'label' column
        self.dataset_train = self.dataset_train.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_remainder = self.dataset_remainder.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_test = self.dataset_test.rename_column("label", LABEL_COLUMN_NAME)


class Trec6DatasetContainer(HuggingFaceDatasetContainer):
    """A dataset container for the TREC-6 dataset

    The Text REtrieval Conference (TREC) Question Classification dataset [1]_
    [2]_ is composed of a number of questions categorised by question type.
    This is the courser-grained version, which uses 6 classes.

    The original train dataset split has been divided into a new train split
    and a validation split, taking 10% for the latter.

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
    [1] Xin Li and Dan Roth, "Learning Question Classifiers", COLING 2002: The
    19th International Conference on Computational Linguistics, 2002
    [2] Eduard et al., "Toward Semantics-Based Answer Pinpointing",
    Proceedings of the First International Conference on Human Language
    Technology Research, 2001
    """

    DATASET_NAME = "OxAISH-AL-LLM/trec6"
    CATEGORIES = OrderedDict(
        [
            ("ABBR", "Abbreviation"),
            ("ENTY", "Entity"),
            ("DESC", "Description and abstract concept"),
            ("HUM", "Human being"),
            ("LOC", "Location"),
            ("NUM", "Numeric value"),
        ]
    )
    TOKENIZED_LENGTH_UPPER_QUARTILE = 14

    def _preprocess_dataset(self):
        # Do any preprocessing defined by the base class
        super()._preprocess_dataset()

        # Rename the 'label' column
        self.dataset_train = self.dataset_train.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_remainder = self.dataset_remainder.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_validation = self.dataset_validation.rename_column(
            "label", LABEL_COLUMN_NAME
        )
        self.dataset_test = self.dataset_test.rename_column("label", LABEL_COLUMN_NAME)
