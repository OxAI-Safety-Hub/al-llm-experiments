import argparse
import wandb

from al_llm.utils.artifacts import save_dual_label_results, load_dataset_extension
from al_llm.constants import WANDB_ENTITY
from labelling_helper import label_and_get_results

# Parser to pass the run id through to the program
parser = argparse.ArgumentParser(description="Dual label existing experiment labels.")
parser.add_argument(
    "--run-id",
    type=str,
    help="The run id of the experiment whos added data we should dual label.",
)
parser.add_argument(
    "--project-name",
    type=str,
    help="The W&B project containing the run.",
)
parser.add_argument(
    "--score-ambiguities",
    help="If flagged, the ambiguities of labels have to match",
    action="store_true",
)
args = parser.parse_args()

# Initialise a run to retrieve this data
run = wandb.init(
    project=args.project_name,
    entity=WANDB_ENTITY,
    id=args.run_id,
    resume="must",
)

# Get the parameters which this experiment used
run_config = run.config

# Get the dataset extension from wandb
data_dict = load_dataset_extension(run)
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


def save_results(new_labels: list, new_ambiguities: list, consistency: float):
    """Saves the results of this dual labelling process to wandb

    Parameters
    ----------
    new_labels : list
        A list of the new labels
    new_ambiguities : list
        A list of the new ambiguities
    consistency : float
        The proportion of sentences which both humans labelled the same
    """

    # Prepare data_dict for saving
    data_dict["new_labels"] = new_labels
    data_dict["new_ambiguities"] = new_ambiguities

    # Prepare results for saving
    results = {
        "num_labels": num_labels,
        "score_ambiguities": args.score_ambiguities,
        "labelling_consistency": consistency,
    }

    # Save the results to wandb as an artifact
    save_dual_label_results(run, data_dict, results)


# If the user chooses 'y' then, the rest of the program will run
if decision.lower() == "y":

    new_labels, new_ambiguities, results = label_and_get_results(
        samples=data_dict["text"],
        existing_labels=data_dict["labels"],
        existing_ambiguities=data_dict["ambiguities"],
        categories=run_config["categories"],
        show_feedback=False,
        score_ambiguities=args.score_ambiguities,
    )

    print("------------------------------------------------------")
    print(f"Labelling consistency: {results['consistency']}")
    print("------------------------------------------------------")

    # save the results to wandb
    save_results(new_labels, new_ambiguities, results["consistency"])

    # end the program
    print("------------------------------------------------------")
    print("Thank you for using the dual labelling program!")
    print("------------------------------------------------------")
