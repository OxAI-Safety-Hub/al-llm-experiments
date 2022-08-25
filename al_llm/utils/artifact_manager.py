import os
import tempfile
import json
import configparser
from typing import Any, Tuple

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
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
                wandb_run.project,
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
                wandb_run.project,
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
                wandb_run.name + "_dl",
                type=config["Dual Labelling Loading"]["ArtifactType"],
            )
            artifact.add_dir(tmpdirname)
            wandb_run.log_artifact(artifact)

    @staticmethod
    def save_tapted_model(
        wandb_run: wandb.sdk.wandb_run.Run,
        model: Any,
        training_args: dict,
        base_model_name: str,
        dataset_name: str,
    ):
        """Save a tapted model and it's parameters to wandb as an artifact

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run that this tapted model should be saved to.
        model : Any
            The tapted model to saved
        training_args : dict
            The training arguments which the tapt process used
        base_model_name : str
            The name of the base model used
        dataset_name : str
            The name of the dataset used
        """

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # store the model in this directory
            model_file_path = os.path.join(
                tmpdirname, config["TAPT Model Loading"]["ModelFileName"]
            )
            model.save_pretrained(model_file_path)

            # store the training_args in this directory
            dict_file_path = os.path.join(
                tmpdirname, config["TAPT Model Loading"]["ParametersFileName"]
            )
            with open(dict_file_path, "w") as f:
                json.dump(training_args, f, indent=4)

            # upload this file to weights and biases as an artifact
            artifact = wandb.Artifact(
                base_model_name + "---" + dataset_name,
                type=config["TAPT Model Loading"]["TAPTModelType"],
            )
            artifact.add_dir(tmpdirname)
            wandb_run.log_artifact(artifact)

    @staticmethod
    def load_tapted_model(
        wandb_run: wandb.sdk.wandb_run.Run,
        base_model_name: str,
        dataset_name: str,
        purpose: str,
    ) -> Tuple[Any, dict]:
        """Load a tapted model and it's parameters from wandb

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run for loading the model.
        base_model_name : str
            The name of the base model
        dataset_name : str
            The name of the dataset used for tapting
        purpose : str
            What will this model be used for. "sample_generator" or "classifier"

        Returns
        ----------
        model : Any
            The loaded tapted model
        training_args : dict
            The training arguments which the tapt process used
        """

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # download the model into this directory from wandb
            artifact_name = base_model_name + "---" + dataset_name
            artifact_path_components = (
                config["Wandb"]["Entity"],
                config["TAPT Model Loading"]["TAPTProject"],
                artifact_name + ":latest",
            )
            artifact_path = "/".join(artifact_path_components)
            artifact = wandb_run.use_artifact(
                artifact_path,
                type=config["TAPT Model Loading"]["TAPTModelType"],
            )
            artifact.download(tmpdirname)

            # load the dictionary containing the parameters
            dict_file_path = os.path.join(
                tmpdirname, config["TAPT Model Loading"]["ParametersFileName"]
            )
            with open(dict_file_path, "rb") as f:
                training_args = json.load(f)

            # load model from this directory
            model_file_path = os.path.join(
                tmpdirname, config["TAPT Model Loading"]["ModelFileName"]
            )
            # add the correct head depending on the purpose
            if purpose == "classifier":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_file_path, num_labels=2
                )
            elif purpose == "sample_generator":
                model = AutoModelForCausalLM.from_pretrained(model_file_path)
            else:
                raise Exception("Unrecognised 'purpose' when loading tapted model")

            return model, training_args
