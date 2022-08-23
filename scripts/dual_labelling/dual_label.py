import argparse
from ast import Num
import os
import tempfile
import wandb
import configparser
import json
import textwrap
from collections import OrderedDict
from typing import Tuple

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
    id=args.run_id,
    resume=True,
)

# Get the parameters which this experiment used
parameters = run.config
categories = OrderedDict([(0, "Negative"), (1, "Positive")])

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


def wrap(text: str) -> str:
    """Wrap some text to the line width"""
    return textwrap.fill(text, width=70)


def prompt(sentence: str) -> Tuple[int, int]:
    """Prompts the user for the label and ambiguity of a sentence

    Parameters
    ----------
    sentence : str
        The sentence to label.

    Returns
    ----------
    new_label, new_ambiguity : int, int
        A tuple of the label and ambiguity defined by the user.
    """

    # Build the message with the sample plus the category selection
    text = "\n"
    text += wrap(f"{sentence!r}") + "\n"
    text += wrap("How would you classify this?") + "\n"
    for j, cat_human_readable in enumerate(categories.values()):
        text += wrap(f"[{j}] {cat_human_readable}") + "\n"
    # If also checking for ambiguity, add these options
    if parameters["ambiguity_mode"] != "none":
        for j, cat_human_readable in enumerate(categories.values()):
            text += (
                wrap(f"[{j+len(categories)}] {cat_human_readable} (ambiguous)") + "\n"
            )

    # Print the message
    print(text)

    # Keep asking the user for a label until they give a valid one
    if parameters["ambiguity_mode"] == "none":
        max_valid_label = len(categories) - 1
    else:
        max_valid_label = 2 * len(categories) - 1
    prompt = wrap(f"Enter a number (0-{max_valid_label}):")
    valid_label = False
    while not valid_label:
        label_str = input(prompt)
        try:
            label = int(label_str)
        except ValueError:
            continue
        if label >= 0 and label <= max_valid_label:
            valid_label = True

    new_label = list(categories.keys())[label % len(categories)]
    new_ambiguity = label // len(categories)
    return new_label, new_ambiguity


def calculate_consistency(new_labels: list, new_ambiguities: list) -> float:
    """Calculates the consistency (%) of the two human labellers

    Parameters
    ----------
    new_labels : list
        A list of the new labels
    new_ambiguities : list
        A list of the new ambiguities

    Returns
    ----------
    consistency : float
        The percentage of sentences which both humans labelled the same
    """

    # keep track of the total number of consistent labels
    num_consistent_labels = 0

    for i in range(num_labels):
        # if the labels and ambiguities are consistent
        if (
            new_labels[i] == data_dict["labels"][i]
            and new_ambiguities[i] == data_dict["ambiguities"][i]
        ):
            # increment the tally
            num_consistent_labels += 1

    # calculate and return the consistency
    consistency = (num_consistent_labels / num_labels) * 100
    return consistency


# If the user chooses 'y' then, the rest of the program will run
if decision.lower() == "y":

    # list to store labelled values
    new_labels = []
    new_ambiguities = []

    # for each sentence that needs labelling
    for i in range(num_labels):
        sentence = data_dict["text"][i]

        # prompt for label and ambiguity
        l, a = prompt(data_dict["text"][i])

        # record these values
        new_labels.append(l)
        new_ambiguities.append(a)

    # calculate and display labelling consistency
    labelling_consistency = calculate_consistency(new_labels, new_ambiguities)
    print("------------------------------------------------------")
    print(f"Labelling consistency: {labelling_consistency}%")
    print("------------------------------------------------------")
