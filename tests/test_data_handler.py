from al_llm.experiment import Experiment
from al_llm.classifier import Classifier
from al_llm.data_handler import HuggingFaceDataHandler, LocalDataHandler
from transformers import AutoTokenizer
from typing import Union, Any

import datasets


# define a dummy classifer that only loads the tokenizer for gpt2
# without loading the rest of the model to save time for each test
class DummyClassifierForTests(Classifier):
    def __init__(self, parameters: dict, run_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.parameters = parameters
        self.run_id = run_id

    def train_afresh(self, data: Any):
        pass

    def train_update(self, data: Any):
        pass

    def tokenize(self, text: Union[str, list]):
        return self.tokenizer(text)

    def initialise(self):
        pass

    def save(self):
        pass

    def make_label_request(self, samples: list):
        pass


# Set the run_id
run_id = "test"

# create parameters and classifier to pass to data handlers
dummy_args = Experiment.make_dummy_experiment(run_id, is_running_pytests=True)
dummy_args["parameters"]["num_epochs"] = 1
dummy_args["classifier"] = DummyClassifierForTests(dummy_args["parameters"], run_id)

# create a HuggingFaceDataHandler to compare output types of each data handler
hugging_data_handler = HuggingFaceDataHandler(
    "rotten_tomatoes", dummy_args["classifier"], dummy_args["parameters"], run_id
)
# create LocalDataHandler using dummy_local_dataset
local_data_handler = LocalDataHandler(
    "local_datasets/dummy_local_dataset",
    dummy_args["classifier"],
    dummy_args["parameters"],
    run_id,
)


def test_data_handler_consistency():
    # check both datahandlers store data of the same type
    assert isinstance(hugging_data_handler.dataset_test, datasets.arrow_dataset.Dataset)
    assert isinstance(local_data_handler.dataset_test, datasets.arrow_dataset.Dataset)
    assert isinstance(
        hugging_data_handler.tokenized_validation, datasets.arrow_dataset.Dataset
    )
    assert isinstance(
        local_data_handler.tokenized_validation, datasets.arrow_dataset.Dataset
    )

    # dummy samples and labels to test with
    samples = ["one", "two", "three"]
    labels = [0, 1, 1]

    # store size of train sets before `new_labelled()` calls
    hug_train_size = len(hugging_data_handler.tokenized_train)
    loc_train_size = len(local_data_handler.tokenized_train)

    # add samples to each data handler
    hug_new_data = hugging_data_handler.new_labelled(samples, labels)
    loc_new_data = local_data_handler.new_labelled(samples, labels)

    # check that `new_labelled()` function returns a dataset of the right size
    assert len(hug_new_data) == 3 and len(loc_new_data) == 3

    # check that new_labelled correctly updates stored train data
    assert hug_train_size + 3 == len(hugging_data_handler.tokenized_train)
    assert loc_train_size + 3 == len(local_data_handler.tokenized_train)
