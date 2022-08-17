import configparser

import datasets

import torch

from al_llm.parameters import Parameters
from al_llm.dataset_container import DummyDatasetContainer, DummyLocalDatasetContainer
from al_llm.classifier import DummyClassifier


# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


def _basic_dataset_container_tests(dataset_container, tokenize):

    # Tokenize the data
    dataset_container.make_tokenized(tokenize)

    # Make sure the tokenized datasets have values of the correct type
    for value in dataset_container.tokenized_train[0].values():
        assert isinstance(value, torch.Tensor)
    for value in dataset_container.tokenized_validation[0].values():
        assert isinstance(value, torch.Tensor)
    for value in dataset_container.tokenized_validation[0].values():
        assert isinstance(value, torch.Tensor)
    for value in dataset_container.tokenized_test[0].values():
        assert isinstance(value, torch.Tensor)

    # Make sure the lengths match up
    assert len(dataset_container.dataset_train) == len(
        dataset_container.tokenized_train
    )
    assert len(dataset_container.dataset_validation) == len(
        dataset_container.tokenized_validation
    )
    assert len(dataset_container.dataset_test) == len(dataset_container.tokenized_test)

    # Get the current length
    train_length = len(dataset_container.dataset_train)

    # Add an item
    item = {"text": "This is a test sentence", "labels": 0}
    dataset_container.add_item(item, tokenize)

    # Check that it's been added
    assert dataset_container.dataset_train[-1]["text"] == item["text"]
    assert dataset_container.dataset_train[-1]["labels"] == item["labels"]
    assert len(dataset_container.dataset_train) == train_length + 1
    assert len(dataset_container.tokenized_train) == train_length + 1

    # Add some more items, this time in a batch
    items = {
        "text": ["This is another test sentence", "As it this one"],
        config["Data Handling"]["LabelColumnName"]: [0, 0],
    }
    items_len = len(items["text"])
    dataset_container.add_items(items, tokenize)

    # Check that they've been added
    for i in range(items_len):
        i_dataset = -(items_len - i)
        assert dataset_container.dataset_train[i_dataset]["text"] == items["text"][i]
        assert (
            dataset_container.dataset_train[i_dataset]["labels"] == items["labels"][i]
        )
    assert len(dataset_container.dataset_train) == train_length + 1 + items_len
    assert len(dataset_container.tokenized_train) == train_length + 1 + items_len


def test_dummy_dataset_container():

    # Set up the dummy dataset container
    parameters = Parameters()
    dataset_container = DummyDatasetContainer(parameters)

    # Make sure the dataset splits are of the right type
    assert isinstance(dataset_container.dataset_train, datasets.Dataset)
    assert isinstance(dataset_container.dataset_validation, datasets.Dataset)
    assert isinstance(dataset_container.dataset_test, datasets.Dataset)

    # The tokenize function, is the dummy one from DummyClassifier
    def tokenize(text):
        return DummyClassifier.tokenize(None, text)

    _basic_dataset_container_tests(dataset_container, tokenize)


def test_local_dataset_container():

    # Set up the dummy dataset container
    parameters = Parameters()
    dataset_container = DummyLocalDatasetContainer(parameters)

    # The tokenize function, is the dummy one from DummyClassifier
    def tokenize(text):
        return DummyClassifier.tokenize(None, text)

    _basic_dataset_container_tests(dataset_container, tokenize)
