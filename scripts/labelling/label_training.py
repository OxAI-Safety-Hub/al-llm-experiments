import argparse
import sys
import wandb

from al_llm.parameters import Parameters
from al_llm.experiment import Experiment

from al_llm.constants import WANDB_ENTITY, LABEL_TRAINING_SEED, LABEL_TRAINING_PROJECT
from labelling_helper import label_and_get_results

# Parser to pass the run id through to the program
parser = argparse.ArgumentParser(description="Learn how to correctly label a dataset.")
parser.add_argument(
    "--dataset-name",
    type=str,
    help="The name of the dataset to learn to label.",
)
parser.add_argument(
    "--num-labels",
    type=int,
    help="The number of examples to learn from.",
)
parser.add_argument(
    "--seed",
    type=int,
    help="The seed to use for shuffling the dataset.",
    default=LABEL_TRAINING_SEED,
)

args = parser.parse_args()

# Create a dataset container for the dataset
dummy_parameters = Parameters(train_dataset_size=1000)
dataset_container = Experiment.MAP_DATASET_CONTAINER[args.dataset_name](
    dummy_parameters
)

# Create a sub dataset to use for labelling
dataset = dataset_container.dataset_train
dataset = dataset.shuffle(seed=args.seed)
sub_dataset = dataset[: args.num_labels]

# Log an introductory message to the user, allowing opt out
print("------------------------------------------------------")
print("Welcome to the labelling training program.")
print("------------------------------------------------------")
print(f"You will be given a subsection of the {args.dataset_name} dataset to label.")
print(f"In this dataset, there are {args.num_labels} sentences to label.")
print("You must do this in one sitting. Are you happy to continue?")
decision = input("Answer (Y/n): ")
print("------------------------------------------------------")

if decision != "y" and decision != "":
    sys.exit()


def log_results(consistency: float):
    """Logs the results of this training to wandb

    Parameters
    ----------
    consistency : float
        The proportion of sentences which both humans labelled the same
    """

    label_training_parameters = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "num_labels": args.num_labels,
    }

    wandb_run = wandb.init(
        project=LABEL_TRAINING_PROJECT,
        entity=WANDB_ENTITY,
        config=label_training_parameters,
    )

    wandb_run.log({"consistency": consistency})


# label the data and analyse results
new_labels, new_ambiguities, results = label_and_get_results(
    samples=sub_dataset["text"],
    existing_labels=sub_dataset["labels"],
    existing_ambiguities=sub_dataset["ambiguities"],
    categories=dataset_container.CATEGORIES,
    show_feedback=True,
    score_ambiguities=False,
)

print("------------------------------------------------------")
print(f"Labelling consistency: {results['consistency']}")
print("------------------------------------------------------")

# save the results to wandb
log_results(results["consistency"])

# end the program
print("------------------------------------------------------")
print("Thank you for using the labelling training program!")
print("------------------------------------------------------")
