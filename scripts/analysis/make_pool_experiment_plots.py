from argparse import ArgumentParser

import wandb

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

from al_llm.constants import (
    WANDB_ENTITY,
    WANDB_PROJECTS,
)
from al_llm import Experiment


# Set up the arg parser
parser = ArgumentParser(
    description="Perform an analysis task on a run",
)
parser.add_argument(
    "--wandb-tag",
    type=str,
    help=("The name of the project which holds the run"),
    default="pool-based",
)
parser.add_argument(
    "--chart-title",
    type=str,
    help=("The title to give the output chart"),
)
parser.add_argument(
    "--colormap",
    type=str,
    help=("The matplotlib colormap from which to select the colours"),
    default="cividis",
)

# Get the arguments
cmd_args = parser.parse_args()

# The W&B public API
api = wandb.Api()

print("Obtaining run data...")

# Load up the run
runs_path = f"{WANDB_ENTITY}/{WANDB_PROJECTS['experiment']}"
runs = api.runs(runs_path, filters={"tags": cmd_args.wandb_tag})

# Three-dimensional dict of all the run histories, by dataset, train size and
# acquisition function
run_histories = {}

# Record the different train sizes
train_sizes = []

for i, run in enumerate(tqdm(runs)):

    # Get the run history as eval.f1 by iteration
    run_history = run.history()
    run_history = run_history[["iteration", "eval.f1"]]
    run_history = run_history.loc[pd.notna(run_history["eval.f1"])]

    # Get relevant run parameters
    dataset_name = run.config["dataset_name"]
    train_dataset_size = run.config["train_dataset_size"]
    acquisition_fun = run.config["acquisition_function"]

    # Add the history to the dict, creating keys if necessary
    if dataset_name not in run_histories:
        run_histories[dataset_name] = {}
    if train_dataset_size not in run_histories[dataset_name]:
        run_histories[dataset_name][train_dataset_size] = {}
    if acquisition_fun not in run_histories[dataset_name][train_dataset_size]:
        run_histories[dataset_name][train_dataset_size][acquisition_fun] = []
    run_histories[dataset_name][train_dataset_size][acquisition_fun].append(run_history)

    # Add the train size to the list of these, if not already there
    if train_dataset_size not in train_sizes:
        train_sizes.append(train_dataset_size)

    # if i > 9:
    #     break

# Sort the list of train sizes
train_sizes = sorted(train_sizes)

print("Making plot...")

# Get the size of each dimension
num_datasets = len(run_histories.keys())
num_train_sizes = len(train_sizes)

fig, axs = plt.subplots(
    num_datasets,
    num_train_sizes,
    sharex=True,
    figsize=(num_datasets * 2, num_train_sizes * 2),
)

# Loop through all the dimensions
for y, (dataset_name, sub_dict) in enumerate(run_histories.items()):
    for x, train_dataset_size in enumerate(train_sizes):

        if train_dataset_size in sub_dict:
            sub_sub_dict = sub_dict[train_dataset_size]
        else:
            continue

        ax = axs[y][x]

        for acquisition_fun, run_history_list in sub_sub_dict.items():

            # Select a colour for the run
            if acquisition_fun == "random":
                colour = "g"
            else:
                colour = "b"

            # Plot each run
            for run_history in run_history_list:
                ax.plot(run_history["iteration"], run_history["eval.f1"], colour)

for ax, dataset_name in zip(axs[:, 0], run_histories.keys()):
    ax.set_ylabel("Eval F1")
    ax.annotate(
        dataset_name,
        xy=(0, 0.5),
        xytext=(-ax.yaxis.labelpad - 70, 0),
        xycoords="axes fraction",
        textcoords="offset points",
        size="large",
        ha="right",
        va="center",
        rotation="vertical",
    )

for ax, train_dataset_size in zip(axs[num_datasets - 1], train_sizes):
    ax.set_xlabel("Iteration")
    ax.annotate(
        f"Train size: {train_dataset_size}",
        xy=(0.5, 0),
        xytext=(0, -ax.xaxis.labelpad - 70),
        xycoords="axes fraction",
        textcoords="offset points",
        size="large",
        ha="center",
        va="baseline",
    )

handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

# fig.tight_layout()

plt.show()
