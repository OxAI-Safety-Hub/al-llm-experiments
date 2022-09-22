from argparse import ArgumentParser
import os
from tempfile import TemporaryDirectory
import json

import wandb

import pandas as pd

import matplotlib.pyplot as plt

from al_llm.constants import (
    WANDB_ENTITY,
    WANDB_PROJECTS,
    DATASET_EXT_PREFIX,
    DATASET_EXT_DATASET_FILE_NAME,
)
from al_llm import Experiment


# Set up the arg parser
parser = ArgumentParser(
    description="Perform an analysis task on a run",
)
parser.add_argument(
    "run_id",
    type=str,
    help=("The run ID to analyse"),
)
task_help = """The task to run. Allowed values are as follows.
"class_proportion":
    the proportion of selected samples per iteration which lie in each class.
"ambiguities_proportion":
    the proportion of selected samples per iteration which are marked as 
    ambiguous.
"""
parser.add_argument(
    "task",
    choices=["class_proportion", "ambiguities_proportion"],
    metavar="task",
    help=task_help,
)
parser.add_argument(
    "--project-name",
    type=str,
    help=("The name of the project which holds the run"),
    default=WANDB_PROJECTS["experiment"],
)
parser.add_argument(
    "--artifact-version",
    type=str,
    help=("The version of the dataset extension artifact to use"),
    default="latest",
)
parser.add_argument(
    "--chart-title",
    type=str,
    help=("The title to give the output chart"),
)

# Get the arguments
cmd_args = parser.parse_args()

# The W&B public API
api = wandb.Api()

print("Obtaining run metadata...")

# Load up the run
run_path = f"{WANDB_ENTITY}/{cmd_args.project_name}/{cmd_args.run_id}"
run = api.run(run_path)

# Get the relevant parameters
num_samples = run.config["num_samples"]
dataset_name = run.config["dataset_name"]

# Get the OrderedDict of categories
categories = Experiment.MAP_DATASET_CONTAINER[dataset_name].CATEGORIES

print("Loading dataset extension...")

# Make the path to the artifact on W&B
artifact_path = (
    f"{WANDB_ENTITY}/{cmd_args.project_name}/{DATASET_EXT_PREFIX}"
    f"{cmd_args.run_id}:{cmd_args.artifact_version}"
)

# Download the artifact and load it as a dictionary
dataset_extension_artifact = api.artifact(artifact_path)
with TemporaryDirectory() as tmp:
    dataset_extension_artifact.download(root=tmp)
    filepath = os.path.join(tmp, DATASET_EXT_DATASET_FILE_NAME)
    with open(filepath) as f:
        dataset_extension_dict = json.load(f)

# Make it into a pandas dataframe
dataset_extension_df = pd.DataFrame(dataset_extension_dict)

print("Performing analysis...")

if cmd_args.task == "class_proportion":

    # Get the sequence of labels
    labels = dataset_extension_df["labels"].to_numpy()

    # Reshape it into batches, one per iteration
    assert labels.shape[0] % num_samples == 0
    num_iterations = labels.shape[0] // num_samples
    labels_batched = labels.reshape((num_iterations, num_samples))

    # Count the number of occurrences of each class, per iteration
    class_props = {}
    for i, cat_name in enumerate(categories.values()):
        class_props[cat_name] = (labels_batched == i).sum(axis=1) / num_samples

    # Make it into a dataframe
    class_props_df = pd.DataFrame(class_props)

    # Plot the area chart
    ax = class_props_df.plot.area()

elif cmd_args.task == "ambiguities_proportion":

    # Get the sequence of labels
    ambiguities = dataset_extension_df["ambiguities"].to_numpy()

    # Reshape it into batches, one per iteration
    assert ambiguities.shape[0] % num_samples == 0
    num_iterations = ambiguities.shape[0] // num_samples
    ambiguities_batched = ambiguities.reshape((num_iterations, num_samples))

    # Count the number of occurrences of each class, per iteration
    ambiguities_props = {
        "Ambiguous": (ambiguities_batched == 1).sum(axis=1) / num_samples,
        "Non-ambiguous": (ambiguities_batched == 0).sum(axis=1) / num_samples,
    }

    # Make it into a dataframe
    ambiguities_props_df = pd.DataFrame(ambiguities_props)

    # Plot the area chart
    ax = ambiguities_props_df.plot.area()

if cmd_args.chart_title is not None:
    ax.set_title(cmd_args.chart_title, wrap=True)

# Show the chart
plt.show()
