import argparse
from al_llm.dataset_container import (
    RottenTomatoesDatasetContainer,
    WikiToxicDatasetContainer,
)
from al_llm.parameters import Parameters

from labelling_helper import label_and_get_results

# Parser to pass the run id through to the program
parser = argparse.ArgumentParser(description="Learn how to correctly label a dataset.")
parser.add_argument(
    "--dataset-name",
    type=str,
    help="The name of the dataset to learn to label.",
)
parser.add_argument(
    "--seed",
    type=int,
    help="The seed to use for shuffling the dataset.",
)
parser.add_argument(
    "--num-labels",
    type=int,
    help="The number of examples to learn from.",
)

args = parser.parse_args()

# Create a dataset container for the dataset
dummy_parameters = Parameters()
if args.dataset_name == "rotten_tomatoes":
    dataset_container = RottenTomatoesDatasetContainer(dummy_parameters)
elif args.dataset_name == "wiki_toxic":
    dataset_container = WikiToxicDatasetContainer(dummy_parameters)
else:
    raise Exception(f"{args.dataset_name} is not a valid dataset_name.")

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
decision = input("Answer (y/n): ")
print("------------------------------------------------------")


def save_results(new_labels: list, new_ambiguities: list, consistency: float):
    """Calculates the consistency of the two human labellers

    Parameters
    ----------
    new_labels : list
        A list of the new labels
    new_ambiguities : list
        A list of the new ambiguities
    consistency : float
        The proportion of sentences which both humans labelled the same
    """

    pass


# If the user chooses 'y' then, the rest of the program will run
if decision.lower() == "y":

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
    save_results(new_labels, new_ambiguities, results["consistency"])

    # end the program
    print("------------------------------------------------------")
    print("Thank you for using the labelling training program!")
    print("------------------------------------------------------")
