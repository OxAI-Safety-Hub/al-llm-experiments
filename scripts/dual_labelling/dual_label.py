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

# use a temporary directory as an inbetween
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
        added_data = json.load(file)

print(len(added_data["text"]))
print(len(added_data["labels"]))
print(len(added_data["ambiguities"]))
