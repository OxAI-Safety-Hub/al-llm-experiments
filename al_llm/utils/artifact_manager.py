import os
import tempfile
import json
import configparser

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
