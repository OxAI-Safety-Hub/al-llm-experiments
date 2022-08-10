from al_llm.experiment import Experiment
from al_llm.classifier import DummyGPT2Classifier
from al_llm.data_handler import HuggingFaceDataHandler, LocalDataHandler

import datasets

# create parameters and classifier to pass to data handlers
dummy_args = Experiment.make_dummy_experiment()
dummy_args["parameters"]["num_epochs"] = 1
dummy_args["classifier"] = DummyGPT2Classifier(dummy_args["parameters"])

# create a HuggingFaceDataHandler to compare output types of each data handler
hug = HuggingFaceDataHandler(
    "rotten_tomatoes", dummy_args["classifier"], dummy_args["parameters"]
)
# create LocalDataHandler using dummy_local_dataset
loc = LocalDataHandler(
    "local_datasets/dummy_local_dataset",
    dummy_args["classifier"],
    dummy_args["parameters"],
)


def tests():
    # check both datahandlers store data of the same type
    assert(isinstance(hug.dataset_test, datasets.arrow_dataset.Dataset))
    assert(isinstance(loc.dataset_test, datasets.arrow_dataset.Dataset))
    assert(isinstance(hug.tokenized_validation, datasets.arrow_dataset.Dataset))
    assert(isinstance(loc.tokenized_validation, datasets.arrow_dataset.Dataset))

    # dummy samples and labels to test with
    samples = ["one", "two", "three"]
    labels = [0, 1, 1]

    # store size of train sets before `new_labelled()` calls
    hug_train_size = len(hug.tokenized_train)
    loc_train_size = len(loc.tokenized_train)

    # add samples to each data handler
    hug_new_data = hug.new_labelled(samples, labels)
    loc_new_data = loc.new_labelled(samples, labels)

    # check that `new_labelled()` function returns a dataset of the right size
    assert(len(hug_new_data) == 3 and len(loc_new_data) == 3)

    # check that the returned type is consistent
    assert(isinstance(hug_new_data, datasets.arrow_dataset.Dataset))
    assert(isinstance(loc_new_data, datasets.arrow_dataset.Dataset))

    # check that new_labelled correctly updates stored train data
    assert(hug_train_size + 3 == len(hug.tokenized_train))
    assert(loc_train_size + 3 == len(loc.tokenized_train))
