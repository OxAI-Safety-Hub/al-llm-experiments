import datasets

import torch

from al_llm.parameters import Parameters
from al_llm.dataset_container import DummyDatasetContainer
from al_llm.classifier import DummyClassifier
from al_llm.constants import (
    TEXT_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    AMBIGUITIES_COLUMN_NAME,
    SKIPS_COLUMN_NAME,
)


def _basic_dataset_container_tests(dataset_container, tokenize):

    # Tokenize the data
    dataset_container.make_tokenized(tokenize)

    # Make sure the tokenized datasets have values of the correct type
    for value in dataset_container.tokenized_train[0].values():
        assert isinstance(value, torch.Tensor)
    for value in dataset_container.tokenized_remainder[0].values():
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
    item = {
        TEXT_COLUMN_NAME: "This is a test sentence",
        LABEL_COLUMN_NAME: 0,
        AMBIGUITIES_COLUMN_NAME: 0,
        SKIPS_COLUMN_NAME: 0,
    }
    dataset_container.add_item(item, tokenize)

    # Check that it's been added
    assert (
        dataset_container.dataset_train[-1][TEXT_COLUMN_NAME] == item[TEXT_COLUMN_NAME]
    )
    assert (
        dataset_container.dataset_train[-1][LABEL_COLUMN_NAME]
        == item[LABEL_COLUMN_NAME]
    )
    assert len(dataset_container.dataset_train) == train_length + 1
    assert len(dataset_container.tokenized_train) == train_length + 1

    # Add some more items, this time in a batch
    items = {
        TEXT_COLUMN_NAME: ["This is another test sentence", "As it this one"],
        LABEL_COLUMN_NAME: [0, 0],
        AMBIGUITIES_COLUMN_NAME: [0, 0],
        SKIPS_COLUMN_NAME: [0, 0],
    }
    items_len = len(items[TEXT_COLUMN_NAME])
    dataset_container.add_items(items, tokenize)

    # Check that they've been added
    for i in range(items_len):
        i_dataset = -(items_len - i)
        assert (
            dataset_container.dataset_train[i_dataset][TEXT_COLUMN_NAME]
            == items[TEXT_COLUMN_NAME][i]
        )
        assert (
            dataset_container.dataset_train[i_dataset][LABEL_COLUMN_NAME]
            == items[LABEL_COLUMN_NAME][i]
        )
    assert len(dataset_container.dataset_train) == train_length + 1 + items_len
    assert len(dataset_container.tokenized_train) == train_length + 1 + items_len


def test_dummy_dataset_container():

    # Set up the dummy dataset container
    parameters = Parameters()
    dataset_container = DummyDatasetContainer(parameters)

    # Make sure the dataset splits are of the right type
    assert isinstance(dataset_container.dataset_train, datasets.Dataset)
    assert isinstance(dataset_container.dataset_remainder, datasets.Dataset)
    assert isinstance(dataset_container.dataset_validation, datasets.Dataset)
    assert isinstance(dataset_container.dataset_test, datasets.Dataset)

    # The tokenize function, is the dummy one from DummyClassifier
    def tokenize(text):
        return DummyClassifier.tokenize(None, text)

    _basic_dataset_container_tests(dataset_container, tokenize)
