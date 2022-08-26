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


class SaveLoadHelper:
    """A static class to aid the ArtifactManager"""

    @staticmethod
    def save_json(data: Any, tmp: str, file_name: str):
        """Save data into a json file in a temporary directory

        Parameters
        ----------
        data : Any
            The data to store in the json file.
        tmp : str
            The temporary directory to use as an in between.
        file_name : str
            The file name to save the data into.
        """

        file_path = os.path.join(tmp, file_name)
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def load_json(tmp: str, file_name: str) -> Any:
        """Load data from a json file in a temporary directory

        Parameters
        ----------
        tmp : str
            The temporary directory to use as an in between.
        file_name : str
            The file name to load the data from.

        Returns
        ----------
        data : Any
            The data stored in the json file.
        """

        file_path = os.path.join(tmp, file_name)
        with open(file_path, "r") as file:
            return json.load(file)

    @staticmethod
    def save_model(model: Any, tmp: str, file_name: str):
        """Save a model in a temporary directory

        Parameters
        ----------
        model : Any
            The model to save to the temporaty directory.
        tmp : str
            The temporary directory to use as an in between.
        file_name : str
            The file name to save the model into.
        """

        file_path = os.path.join(tmp, file_name)
        model.save_pretrained(file_path)

    @staticmethod
    def load_model(tmp: str, file_name: str, purpose: str) -> Any:
        """Load a model from a temporary directory

        Parameters
        ----------
        tmp : str
            The temporary directory to use as an in between.
        file_name : str
            The file name the model is stored in.
        purpose : str
            Which head should we load the model with

        Returns
        ----------
        model : Any
            The model from the temporaty directory.
        """

        model_file_path = os.path.join(
            tmp, config["TAPT Model Loading"]["ModelFileName"]
        )
        # add the correct head depending on the purpose
        if purpose == "classifier":
            return AutoModelForSequenceClassification.from_pretrained(
                model_file_path, num_labels=2
            )
        elif purpose == "sample_generator":
            return AutoModelForCausalLM.from_pretrained(model_file_path)
        else:
            raise Exception("Unrecognised 'purpose' when loading tapted model")

    @staticmethod
    def upload_artifact(
        wandb_run: wandb.sdk.wandb_run.Run,
        artifact_name: str,
        artifact_type: str,
        tmp: str,
    ):
        """Upload the contents of a temporary directory as a wandb artifact

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run to save the artifact to.
        artifact_name : str
            The name of the artifact.
        artifact_type : str
            The type of the artifact.
        tmp : str
            The temporary directory to use as an in between.
        """

        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_dir(tmp)
        wandb_run.log_artifact(artifact)

    @staticmethod
    def download_artifact(
        wandb_run: wandb.sdk.wandb_run.Run,
        project: str,
        artifact_name: str,
        artifact_type: str,
        tmp: str,
    ):
        """Download a wandb artifact into a temporary directory

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run to save the artifact to.
        project : str
            The name of the project the artifact is stored under.
        artifact_name : str
            The name of the artifact.
        artifact_type : str
            The type of the artifact.
        tmp : str
            The temporary directory to use as an in between.
        """

        artifact_path_components = (
            config["Wandb"]["Entity"],
            project,
            artifact_name + ":latest",
        )
        artifact_path = "/".join(artifact_path_components)
        artifact = wandb_run.use_artifact(
            artifact_path,
            type=artifact_type,
        )
        artifact.download(tmp)


class ArtifactManager:
    """A static class to handle all artifact saving and loading"""

    TAPT_PROJECT = "TAPT-Models"
    TAPT_MODEL_TYPE = "TAPT-model"
    TAPT_MODEL_FILE_NAME = "model_home.pt"
    TAPT_PARAMETERS_FILE_NAME = "parameters_home.json"

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

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmp:

            SaveLoadHelper.save_json(
                added_data, tmp, config["Added Data Loading"]["DatasetFileName"]
            )

            SaveLoadHelper.upload_artifact(
                wandb_run=wandb_run,
                artifact_name=f"de_{wandb_run.name}",
                artifact_type=config["Added Data Loading"]["DatasetType"],
                tmp=tmp,
            )

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
        with tempfile.TemporaryDirectory() as tmp:

            SaveLoadHelper.download_artifact(
                wandb_run=wandb_run,
                project=wandb_run.project,
                artifact_name=f"de_{wandb_run.name}",
                artifact_type=config["Added Data Loading"]["DatasetType"],
                tmp=tmp,
            )

            added_data = SaveLoadHelper.load_json(
                tmp, config["Added Data Loading"]["DatasetFileName"]
            )

            return added_data

    def save_classifier_model(
        wandb_run: wandb.sdk.wandb_run.Run,
        model: Any,
    ):
        """Save a classifier model to wandb as an artifact

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run that this dataset extension should be saved to.
        model : Any
            The classifier model to saved
        """

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmp:

            SaveLoadHelper.save_model(
                model, tmp, config["Classifier Loading"]["ModelFileName"]
            )

            SaveLoadHelper.upload_artifact(
                wandb_run=wandb_run,
                artifact_name=f"cl_{wandb_run.name}",
                artifact_type=config["Classifier Loading"]["ClassifierType"],
                tmp=tmp,
            )

    def load_classifier_model(
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> Any:
        """Load a classifier model from wandb

        Parameters
        ----------
        wandb_run : wandb.sdk.wandb_run.Run
            The run with which to load the model

        Returns
        ----------
        model : Any
            The classifier model
        """

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmp:

            SaveLoadHelper.download_artifact(
                wandb_run=wandb_run,
                project=wandb_run.project,
                artifact_name=f"cl_{wandb_run.name}",
                artifact_type=config["Classifier Loading"]["ClassifierType"],
                tmp=tmp,
            )

            model = SaveLoadHelper.load_model(
                tmp, config["Classifier Loading"]["ModelFileName"], "classifier"
            )
            return model

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

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmp:

            SaveLoadHelper.save_json(
                data_dict, tmp, config["Dual Labelling Loading"]["LabelsFileName"]
            )
            SaveLoadHelper.save_json(
                results_dict, tmp, config["Dual Labelling Loading"]["ResultsFileName"]
            )

            SaveLoadHelper.upload_artifact(
                wandb_run=wandb_run,
                artifact_name=f"dl_{wandb_run.name}",
                artifact_type=config["Dual Labelling Loading"]["ArtifactType"],
                tmp=tmp,
            )

    @classmethod
    def save_tapted_model(
        cls,
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
        with tempfile.TemporaryDirectory() as tmp:

            SaveLoadHelper.save_model(model, tmp, cls.TAPT_MODEL_FILE_NAME)
            SaveLoadHelper.save_json(training_args, tmp, cls.TAPT_PARAMETERS_FILE_NAME)

            SaveLoadHelper.upload_artifact(
                wandb_run=wandb_run,
                artifact_name=base_model_name + "---" + dataset_name,
                artifact_type=cls.TAPT_MODEL_TYPE,
                tmp=tmp,
            )

    @classmethod
    def load_tapted_model(
        cls,
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
        with tempfile.TemporaryDirectory() as tmp:

            SaveLoadHelper.download_artifact(
                wandb_run=wandb_run,
                project=cls.TAPT_PROJECT,
                artifact_name=base_model_name + "---" + dataset_name,
                artifact_type=cls.TAPT_MODEL_TYPE,
                tmp=tmp,
            )

            training_args = SaveLoadHelper.load_json(tmp, cls.TAPT_PARAMETERS_FILE_NAME)
            model = SaveLoadHelper.load_model(tmp, cls.TAPT_MODEL_FILE_NAME, purpose)
            return model, training_args
