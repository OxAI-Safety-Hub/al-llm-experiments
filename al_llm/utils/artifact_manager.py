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
