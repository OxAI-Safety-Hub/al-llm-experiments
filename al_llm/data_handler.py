# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from typing import Union
import configparser

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

    def save(self):
        """Save the current dataset"""
        self.dataset_container.save()