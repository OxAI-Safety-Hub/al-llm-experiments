import os
import tempfile
import json
from typing import Any, Tuple, Optional

from transformers import (
    PreTrainedModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
import datasets

import wandb

from al_llm.constants import WANDB_ENTITY, TAPTED_MODEL_DEFAULT_TAG


# Saving and loading constants: Dataset Extensions
DATASET_EXT_ARTIFACT_TYPE = "dataset-extension"
DATASET_EXT_DATASET_FILE_NAME = "added_data.json"

# Saving and loading constants: Classifier Models
CLASSIFIER_ARTIFACT_TYPE = "classifier-model"
CLASSIFIER_MODEL_FILE_NAME = "model_home.pt"

# Saving and loading constants: Dual Labelling Results
DUAL_LAB_ARTIFACT_TYPE = "dual-label-results"
DUAL_LAB_LABELS_FILE_NAME = "labels.json"
DUAL_LAB_RESULTS_FILE_NAME = "results.json"

# Saving and loading constants: TAPT Models
TAPT_PROJECT_NAME = "TAPT-Models"
TAPT_ARTIFACT_TYPE = "TAPT-model"
TAPT_MODEL_FILE_NAME = "model_home.pt"
TAPT_PARAMETERS_FILE_NAME = "parameters_home.json"


def _save_json(data: Any, tmp: str, file_name: str):
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


def _load_json(tmp: str, file_name: str) -> Any:
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


def _save_model(model: PreTrainedModel, tmp: str, file_name: str):
    """Save a model in a temporary directory

    Parameters
    ----------
    model : PreTrainedModel
        The model to save to the temporaty directory.
    tmp : str
        The temporary directory to use as an in between.
    file_name : str
        The file name to save the model into.
    """

    file_path = os.path.join(tmp, file_name)
    model.save_pretrained(file_path)


def _load_model(
    tmp: str, file_name: str, purpose: str, *, num_categories: Optional[int] = None
) -> PreTrainedModel:
    """Load a model from a temporary directory

    Parameters
    ----------
    tmp : str
        The temporary directory to use as an in between.
    file_name : str
        The file name the model is stored in.
    purpose : str
        Which head should we load the model with
    num_categories : int, optional
        The number of class labels, when usings the model as a classifier.

    Returns
    ----------
    model : PreTrainedModel
        The model from the temporaty directory.
    """

    model_file_path = os.path.join(tmp, file_name)
    # add the correct head depending on the purpose
    if purpose == "classifier":
        if num_categories is None:
            raise ValueError(
                "`num_categories` can't be none when loading a model as a classifier"
            )
        return AutoModelForSequenceClassification.from_pretrained(
            model_file_path, num_labels=num_categories
        )
    elif purpose == "sample_generator":
        return AutoModelForCausalLM.from_pretrained(model_file_path)
    else:
        raise Exception("Unrecognised 'purpose' when loading tapted model")


def _upload_artifact(
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


def _download_artifact(
    wandb_run: wandb.sdk.wandb_run.Run,
    project: str,
    artifact_name: str,
    artifact_type: str,
    tmp: str,
    artifact_version: str = TAPTED_MODEL_DEFAULT_TAG,
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
    artifact_version : str, default=TAPTED_MODEL_DEFAULT_TAG
        The artifact version to load. By default it will load the most
        recent version.
    """

    artifact_path_components = (
        WANDB_ENTITY,
        project,
        artifact_name + ":" + artifact_version,
    )
    artifact_path = "/".join(artifact_path_components)
    artifact = wandb_run.use_artifact(
        artifact_path,
        type=artifact_type,
    )
    artifact.download(tmp)


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

        _save_json(added_data, tmp, DATASET_EXT_DATASET_FILE_NAME)

        _upload_artifact(
            wandb_run=wandb_run,
            artifact_name=f"de_{wandb_run.name}",
            artifact_type=DATASET_EXT_ARTIFACT_TYPE,
            tmp=tmp,
        )


def load_dataset_extension(
    wandb_run: wandb.sdk.wandb_run.Run,
    *,
    dataset_wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    dataset_extension_version: str = "latest",
) -> dict:
    """Load a dataset extention from wandb

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run
        The current run.
    dataset_wandb_run : wandb.sdk.wandb_run.Run, optional
        The run where the dataset extension artifact is located. If `None`, we
        take it to be the current run.
    dataset_extension_version : str, default="latest"
        The version of the dataset extension to use

    Returns
    ----------
    added_data : dict
        The dataset extension.
    """

    if dataset_wandb_run is None:
        dataset_wandb_run = wandb_run

    # use a temporary directory as an inbetween
    with tempfile.TemporaryDirectory() as tmp:

        _download_artifact(
            wandb_run=wandb_run,
            project=dataset_wandb_run.project,
            artifact_name=f"de_{dataset_wandb_run.name}",
            artifact_type=DATASET_EXT_ARTIFACT_TYPE,
            tmp=tmp,
            artifact_version=dataset_extension_version,
        )

        added_data = _load_json(tmp, DATASET_EXT_DATASET_FILE_NAME)

        return added_data


def save_classifier_model(
    wandb_run: wandb.sdk.wandb_run.Run,
    model: PreTrainedModel,
):
    """Save a classifier model to wandb as an artifact

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run
        The run that this dataset extension should be saved to.
    model : PreTrainedModel
        The classifier model to saved
    """

    # use a temporary directory as an inbetween
    with tempfile.TemporaryDirectory() as tmp:

        _save_model(model, tmp, CLASSIFIER_MODEL_FILE_NAME)

        _upload_artifact(
            wandb_run=wandb_run,
            artifact_name=f"cl_{wandb_run.name}",
            artifact_type=CLASSIFIER_ARTIFACT_TYPE,
            tmp=tmp,
        )


def load_classifier_model(
    wandb_run: wandb.sdk.wandb_run.Run, num_categories: int
) -> PreTrainedModel:
    """Load a classifier model from wandb

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run
        The run with which to load the model
    num_categories : int
        The number of categories in the classification task

    Returns
    ----------
    model : PreTrainedModel
        The classifier model
    """

    # use a temporary directory as an inbetween
    with tempfile.TemporaryDirectory() as tmp:

        _download_artifact(
            wandb_run=wandb_run,
            project=wandb_run.project,
            artifact_name=f"cl_{wandb_run.name}",
            artifact_type=CLASSIFIER_ARTIFACT_TYPE,
            tmp=tmp,
        )

        model = _load_model(
            tmp, CLASSIFIER_MODEL_FILE_NAME, "classifier", num_categories=num_categories
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

        _save_json(data_dict, tmp, DUAL_LAB_LABELS_FILE_NAME)
        _save_json(results_dict, tmp, DUAL_LAB_RESULTS_FILE_NAME)

        _upload_artifact(
            wandb_run=wandb_run,
            artifact_name=f"dl_{wandb_run.name}",
            artifact_type=DUAL_LAB_ARTIFACT_TYPE,
            tmp=tmp,
        )


def save_tapted_model(
    wandb_run: wandb.sdk.wandb_run.Run,
    model: PreTrainedModel,
    training_args: dict,
    base_model_name: str,
    dataset_name: str,
):
    """Save a tapted model and it's parameters to wandb as an artifact

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run
        The run that this tapted model should be saved to.
    model : PreTrainedModel
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

        _save_model(model, tmp, TAPT_MODEL_FILE_NAME)
        _save_json(training_args, tmp, TAPT_PARAMETERS_FILE_NAME)

        _upload_artifact(
            wandb_run=wandb_run,
            artifact_name=base_model_name + "---" + dataset_name,
            artifact_type=TAPT_ARTIFACT_TYPE,
            tmp=tmp,
        )


def load_tapted_model(
    wandb_run: wandb.sdk.wandb_run.Run,
    base_model_name: str,
    dataset_name: str,
    purpose: str,
    *,
    num_categories: Optional[int] = None,
    tapted_model_version: str = TAPTED_MODEL_DEFAULT_TAG,
) -> Tuple[PreTrainedModel, dict]:
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
    num_categories : int, optional
        The number of class labels, when usings the model as a classifier.
    tapted_model_version : str, default=TAPTED_MODEL_DEFAULT_TAG
        The artifact version of the tapted model to load. By default it will
        load the most recent version

    Returns
    ----------
    model : PreTrainedModel
        The loaded tapted model
    training_args : dict
        The training arguments which the tapt process used
    """

    # use a temporary directory as an inbetween
    with tempfile.TemporaryDirectory() as tmp:

        _download_artifact(
            wandb_run=wandb_run,
            project=TAPT_PROJECT_NAME,
            artifact_name=base_model_name + "---" + dataset_name,
            artifact_type=TAPT_ARTIFACT_TYPE,
            tmp=tmp,
            artifact_version=tapted_model_version,
        )

        training_args = _load_json(tmp, TAPT_PARAMETERS_FILE_NAME)
        model = _load_model(
            tmp, TAPT_MODEL_FILE_NAME, purpose, num_categories=num_categories
        )
        return model, training_args
