from argparse import ArgumentParser
import os
from tempfile import TemporaryDirectory
import json

import wandb

import numpy as np

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
    description=(
        "Analyse the proportion of selected samples used for each "
        "iteration in an experiment which lie in each class"
    ),
)
parser.add_argument(
    "run_id",
    type=str,
    help=("The run ID to analyse"),
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

# Get the sequence of labels
labels = dataset_extension_df["labels"].to_numpy()

# Reshape it into batches, one per iteration
assert labels.shape[0] % num_samples == 0
num_iterations = labels.shape[0] // num_samples
labels_batched = labels.reshape((num_iterations, num_samples))

# Count the number of occurrences of each class, per iteration
class_counts = {}
for i, cat_name in enumerate(categories.values()):
    class_counts[cat_name] = (labels_batched == i).sum(axis=1)

# Make it into a dataframe
class_counts_df = pd.DataFrame(class_counts)

# Plot the area chart
ax = class_counts_df.plot.area()
plt.show()