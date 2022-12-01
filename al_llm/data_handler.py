# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from typing import Union, Optional

import torch

import datasets
import wandb

from al_llm.classifier import Classifier
from al_llm.parameters import Parameters
from al_llm.dataset_container import DatasetContainer
from al_llm.utils.artifacts import save_dataset_extension, load_dataset_extension
from al_llm.constants import (
    TEXT_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    AMBIGUITIES_COLUMN_NAME,
)
from al_llm.utils import UnlabelledSamples, PromptOutput


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
    replay_dataset_extension : Optional[datasets.Dataset]
        If we're replaying a run, this is the dataset extension from that run
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

        # The dataset extension from the run we're replaying, if we're doing
        # that
        self.replay_dataset_extension: Optional[datasets.Dataset] = None

        # Add 'categories' to the config
        wandb.config.update({"categories": self.dataset_container.CATEGORIES})

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

    def new_labelled(self, samples: UnlabelledSamples, prompt_output: PromptOutput):
        """Add new labelled samples to the dataset

        Parameters
        ----------
        samples : UnlabelledSamples
            The list of sample strings
        prompt_output : PromptOutput
            Data class containing the prompt output (labels, ambiguities, skip
            mask)
        """

        # Add the items using the dataset container
        items = {
            TEXT_COLUMN_NAME: list(samples),
            LABEL_COLUMN_NAME: prompt_output.labels,
            AMBIGUITIES_COLUMN_NAME: prompt_output.ambiguities,
        }
        self.dataset_container.add_items(items, self.classifier.tokenize)

    def save(self, unlabelled_samples: UnlabelledSamples):
        """Save the current dataset

        This saves the raw sentence-label pairs that have been added to the
        dataset through AL, possibly including any samples that are waiting
        to be labelled at the start of the next iteration, as a dict in wandb.

        Parameters
        ----------
        unlabelled_samples : UnlabelledSamples
            A list of any generated samples needing labelling, to be stored
            until the next iteration alongside the added labelled data
        """

        # get all datapoints from dataset_train after `train_dataset_size`,
        # i.e. only data added by AL process
        added_data = self.dataset_container.dataset_train[
            self.parameters["train_dataset_size"] :
        ]

        # add the samples in `unlabelled_samples` (if there are any)
        # the other column of the dictionary remains a shorter list to be extended
        # in the next iteration when labels are provided by a human
        added_data[TEXT_COLUMN_NAME].extend(unlabelled_samples)

        # save this dict to WandB, using the ArtifactManager
        save_dataset_extension(self.wandb_run, added_data)

    def load(self):
        """Load the data stored on Weights and Biases

        Returns
        ----------
        added_data : dict
            The dictionary of sentences and labels added in earlier iterations
        """

        added_data = load_dataset_extension(self.wandb_run)
        return added_data

    def load_replay_dataset_extension(self, replayed_run: wandb.sdk.wandb_run.Run):
        """Load the dataset extension from a run we're replaying

        Parameters
        ----------
        replayed_run: wandb.sdk.wandb_run.Run
            The run which we're replaying
        """

        print("Loading data from replayed run...")
        dataset_extension_dict = load_dataset_extension(
            self.wandb_run, dataset_wandb_run=replayed_run
        )
        self.replay_dataset_extension = datasets.Dataset.from_dict(
            dataset_extension_dict
        )

    def get_replay_samples(self, iteration: int) -> UnlabelledSamples:
        """Get a set of samples from the replay dataset extension

        Returns the set of sentences in the slice:
            `iteration * num_samples : (iteration + 1) * num_samples`

        Parameters
        ----------
        iteration : int
            The iteration from which to select the samples

        Returns
        -------
        samples : UnlabelledSamples
            The list of sentences selected
        """

        start = iteration * self.parameters["num_samples"]
        end = (iteration + 1) * self.parameters["num_samples"]
        return UnlabelledSamples(
            self.replay_dataset_extension[TEXT_COLUMN_NAME][start:end]
        )

    def get_replay_prompt_output(self, iteration: int) -> PromptOutput:
        """Get a set of prompt output from the replay dataset extension

        Returns the set of labels, ambiguities and the skip mask in the slice:
            `iteration * num_samples : (iteration + 1) * num_samples`

        Parameters
        ----------
        iteration : int
            The iteration from which to select the samples

        Returns
        -------
        prompt_output : PromptOutput
            The data class containing the replayed labels, ambiguities and
            skip mask
        """

        # Get the labels and ambiguities
        start = iteration * self.parameters["num_samples"]
        end = (iteration + 1) * self.parameters["num_samples"]
        labels = self.replay_dataset_extension[LABEL_COLUMN_NAME][start:end]
        ambiguities = self.replay_dataset_extension[AMBIGUITIES_COLUMN_NAME][start:end]

        # Replace the labels with the corresponding category names
        categories = self.dataset_container.CATEGORIES
        labels = [list(categories.keys())[label] for label in labels]

        return PromptOutput(labels=labels, ambiguities=ambiguities)
