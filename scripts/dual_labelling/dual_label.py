import argparse
import os
import tempfile
import wandb
import configparser
import json

# Parser to pass the run id through to the program
parser = argparse.ArgumentParser(
    description="Pretrain the model on the unlabelled data."
)
parser.add_argument(
    "--run-id",
    type=str,
    help="The run id of the experiment whos added data we should dual label.",
)
args = parser.parse_args()

# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")

# Initialise a run to retrieve this data
run = wandb.init(
    project=config["Wandb"]["Project"],
    entity=config["Wandb"]["Entity"],
    name="dual_labelling_access_run",
)

# Download this data from wandb
#   use a temporary directory as an inbetween
with tempfile.TemporaryDirectory() as tmpdirname:
    # download the dataset into this directory from wandb
    artifact_path_components = (
        config["Wandb"]["Entity"],
        config["Wandb"]["Project"],
        args.run_id + ":latest",
    )
    artifact_path = "/".join(artifact_path_components)
    artifact = run.use_artifact(
        artifact_path,
        type=config["Data Handling"]["DatasetType"],
    )
    artifact.download(tmpdirname)

    # load dataset from this directory
    file_path = os.path.join(tmpdirname, config["Data Handling"]["DatasetFileName"])

    with open(file_path, "r") as file:
        data_dict = json.load(file)

num_labels = len(data_dict["labels"])

# Log an introductory message to the user, allowing opt out
print("------------------------------------------------------")
print("Welcome to the dual labelling program.")
print("------------------------------------------------------")
print("Your job is to label each of these sentences using the labels provided.")
print(f"In this dataset, there are {num_labels} sentences to label.")
print("You must do this in one sitting. Are you happy to continue?")
decision = input("Answer (y/n): ")
print("------------------------------------------------------")

# If the user chooses 'y' then, the rest of the program will run
if (decision.lower() == "y"):
    print("happy to continue")
