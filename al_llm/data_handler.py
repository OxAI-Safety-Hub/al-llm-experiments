# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from typing import Union
import configparser
import tempfile
import os
import json

import torch

import datasets
import wandb

from al_llm.classifier import Classifier
from al_llm.parameters import Parameters
from al_llm.dataset_container import DatasetContainer


# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


class DataHandler:
    """Data handler for loading and processing the data

    The data handler keeps track of both the raw dataset consisting of
    sentences and labels, and the tokenized version.

    The training set is added to as the active learning experiment progresses.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for this experiment
    classifier : classifier.Classifier
        The classifier instance which will be using the data. We will use this
        to know how to tokenize the data.
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run

    Attributes
    ----------
    classifier : classifier.Classifier
        The classifier instance which will be using the data.
    """

    ARTIFACT_NAME = "added-data"

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        classifier: Classifier,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):

        # Store the arguments
        self.parameters = parameters
        self.dataset_container = dataset_container
        self.classifier = classifier
        self.wandb_run = wandb_run

        # Tokenize the dataset using the classifier's tokenize function
        self.dataset_container.make_tokenized(self.classifier.tokenize)

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
        samples_dict = self.dataset_container.tokenized_train[
            -self.parameters["num_samples"] :
        ]
        tokenized_samples = datasets.Dataset.from_dict(samples_dict)
        tokenized_samples.set_format("torch")
        return tokenized_samples

    def new_labelled(self, samples: list, labels: list):
        """Add new labelled samples to the dataset

        Parameters
        ----------
        samples : list
            The list of sample strings
        labels : list
            Labels for the samples
        """

        # Add the items using the dataset container
        items = {
            config["Data Handling"]["TextColumnName"]: samples,
            config["Data Handling"]["LabelColumnName"]: labels,
        }
        self.dataset_container.add_items(items, self.classifier.tokenize)

    def make_label_request(self, samples: list):
        """Make a request for labels for the samples from the human

        Parameters
        ----------
        samples : list
            The sample sentences for which to get the labels
        """
        pass

    def save(self, unlabelled_samples: list):
        """Save the current dataset

        This saves the raw sentence-label pairs that have been added to the
        dataset through AL, possibly including any samples that are waiting
        to be labelled at the start of the next iteration, as a dict in wandb.

        Parameters
        ----------
        unlabelled_samples : list
            A list of any generated samples needing labelling, to be stored
            until the next iteration alongside the added labelled data
        """

        # get all datapoints from dataset_train after `train_dataset_size`,
        # i.e. only data added by AL process
        added_data = self.dataset_container.dataset_train[
            self.parameters["train_dataset_size"] :
        ]

        # add the samples in `unlabelled_samples` (if there are any)
        added_data[config["Data Handling"]["TextColumnName"]].extend(unlabelled_samples)

        # save this dict to WandB, using a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # store the dataset in this directory
            file_path = os.path.join(
                tmpdirname, config["Data Handling"]["DatasetFileName"]
            )
            with open(file_path, "w") as file:
                json.dump(added_data, file)

            # upload the dataset to WandB as an artifact
            artifact = wandb.Artifact(
                self.ARTIFACT_NAME, type=config["Data Handling"]["DatasetType"]
            )
            artifact.add_dir(tmpdirname)
            self.wandb_run.log_artifact(artifact)

    def load(self):
        """Load the data stored on Weights and Biases

        Returns
        ----------
        added_data : dict
            The dictionary of sentences and labels added in earlier iterations
        """

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # download the dataset into this directory from wandb
            artifact_path_components = (
                config["Wandb"]["Entity"],
                config["Wandb"]["Project"],
                self.ARTIFACT_NAME + ":latest",
            )
            artifact_path = "/".join(artifact_path_components)
            artifact = self.wandb_run.use_artifact(
                artifact_path,
                type=config["Data Handling"]["DatasetType"],
            )
            artifact.download(tmpdirname)

            # load dataset from this directory
            file_path = os.path.join(
                tmpdirname, config["Data Handling"]["DatasetFileName"]
            )

            with open(file_path, "r") as file:
                added_data = json.load(file)

            return added_data
