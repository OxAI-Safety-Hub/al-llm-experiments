import os
import tempfile
import json
import configparser
from typing import Any

from transformers import AutoModelForSequenceClassification
import datasets
import wandb

# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


class ArtifactManager:
    """A static class to handle all artifact saving and loading"""

    @staticmethod
    def save_dataset_extension(
        wandb_run: wandb.sdk.wandb_run.Run,
        added_data: datasets.Dataset,
    ):
        """Save a dataset extention to wandb as an artifact

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run that this dataset extension should be saved to.
        added_data : datasets.Dataset
            The dataset extension to save to wandb.
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            # store the dataset in this directory
            file_path = os.path.join(
                tmpdirname, config["Added Data Loading"]["DatasetFileName"]
            )
            with open(file_path, "w") as file:
                json.dump(added_data, file)

            # upload the dataset to WandB as an artifact
            artifact = wandb.Artifact(
                wandb_run.name, type=config["Added Data Loading"]["DatasetType"]
            )
            artifact.add_dir(tmpdirname)
            wandb_run.log_artifact(artifact)

    @staticmethod
    def load_dataset_extension(
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> datasets.Dataset:
        """Load a dataset extention from wandb

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run where this dataset extension is saved.

        Returns
        ----------
        added_data : datasets.Dataset
            The dataset extension.
        """

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # download the dataset into this directory from wandb
            artifact_path_components = (
                config["Wandb"]["Entity"],
                config["Wandb"]["Project"],
                wandb_run.name + ":latest",
            )
            artifact_path = "/".join(artifact_path_components)
            artifact = wandb_run.use_artifact(
                artifact_path,
                type=config["Added Data Loading"]["DatasetType"],
            )
            artifact.download(tmpdirname)

            # load dataset from this directory
            file_path = os.path.join(
                tmpdirname, config["Added Data Loading"]["DatasetFileName"]
            )

            with open(file_path, "r") as file:
                added_data = json.load(file)

            return added_data

    @staticmethod
    def save_classifier_model(
        wandb_run: wandb.sdk.wandb_run.Run,
        model: Any,
        artifact_name: str,
    ):
        """Save a classifier model to wandb as an artifact

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run that this dataset extension should be saved to.
        model : Any
            The classifier model to saved
        artifact_name : str
            The artifact name according to the specific classifier
        """

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # store the model in this directory
            file_path = os.path.join(
                tmpdirname, config["Classifier Loading"]["ModelFileName"]
            )
            model.save_pretrained(file_path)

            # upload this model to weights and biases as an artifact
            artifact = wandb.Artifact(
                artifact_name, type=config["Classifier Loading"]["ClassifierType"]
            )
            artifact.add_dir(tmpdirname)
            wandb_run.log_artifact(artifact)

    @staticmethod
    def load_classifier_model(
        wandb_run: wandb.sdk.wandb_run.Run,
        artifact_name: str,
    ) -> Any:
        """Load a classifier model from wandb

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run with which to load the model
        artifact_name : str
            The artifact name according to the specific classifier

        Returns
        ----------
        model : Any
            The classifier model
        """

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # download the model into this directory from wandb
            artifact_path_components = (
                config["Wandb"]["Entity"],
                config["Wandb"]["Project"],
                artifact_name + ":latest",
            )
            artifact_path = "/".join(artifact_path_components)
            artifact = wandb_run.use_artifact(
                artifact_path,
                type=config["Classifier Loading"]["ClassifierType"],
            )
            artifact.download(tmpdirname)

            # load model from this directory
            file_path = os.path.join(
                tmpdirname, config["Classifier Loading"]["ModelFileName"]
            )
            model = AutoModelForSequenceClassification.from_pretrained(file_path)
            return model

    @staticmethod
    def save_dual_label_results(
        wandb_run: wandb.sdk.wandb_run.Run,
        data_dict: dict,
        results_dict: dict,
    ):
        """Save the results of a dual labelling process

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run that this dataset extension should be saved to.
        data_dict : dict
            The dataset with both sets of labels and ambiguities to save
        results_dict : dict
            The results of the dual labelling process to save
        """

        # save the results to WandB, using a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # store the labels in this directory
            labels_file_path = os.path.join(
                tmpdirname, config["Dual Labelling Loading"]["LabelsFileName"]
            )
            with open(labels_file_path, "w") as file:
                json.dump(data_dict, file)

            # store the results in this directory
            results_file_path = os.path.join(
                tmpdirname, config["Dual Labelling Loading"]["ResultsFileName"]
            )
            with open(results_file_path, "w") as file:
                json.dump(results_dict, file, indent=4)

            # upload the dataset to WandB as an artifact
            artifact = wandb.Artifact(
                wandb_run.name + "_dl", type=config["Dual Labelling Loading"]["ArtifactType"]
            )
            artifact.add_dir(tmpdirname)
            wandb_run.log_artifact(artifact)