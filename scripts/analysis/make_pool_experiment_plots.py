from argparse import ArgumentParser

import wandb

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm

from al_llm.constants import (
    WANDB_ENTITY,
    WANDB_PROJECTS,
)


def make_human_readable(string: str):
    """Turn a variable name into a more human-friendly version"""
    return string.replace("_", " ").capitalize()


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
    default="Eval F1 by iteration for random and max uncertainty acquisition functions across datasets and initial train set sizes",
)
parser.add_argument(
    "--colormap",
    type=str,
    help=("The matplotlib colormap from which to select the colours"),
    default="Set1",
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

# Record the different train sizes and acquisition functions
train_sizes = []
acquisition_funs = []

for run in tqdm(runs):

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

    # Add the train size and acquisition to the list of these, if not already
    # there
    if train_dataset_size not in train_sizes:
        train_sizes.append(train_dataset_size)
    if acquisition_fun not in acquisition_funs:
        acquisition_funs.append(acquisition_fun)

print("Computing statistics...")

# Sort the list of train sizes
train_sizes = sorted(train_sizes)

# Run statistics for each dataset, train size and acquisition function
run_stats = {}

# Build up the 3D dict
for dataset_name, sub_dict in run_histories.items():
    run_stats[dataset_name] = {}
    for train_dataset_size, sub_sub_dict in sub_dict.items():
        run_stats[dataset_name][train_dataset_size] = {}
        for acquisition_fun, run_history_list in sub_sub_dict.items():

            # Record the iterations, assuming that they are the same for each
            # run
            iterations = run_history_list[0]["iteration"].to_numpy()

            # Make a 2D numpy array holding all the eval F1 values
            eval_f1_values = np.zeros((len(run_history_list), len(iterations)))
            for i, run_history in enumerate(run_history_list):
                eval_f1_values[i, :] = run_history["eval.f1"].to_numpy()

            # Compute the mean and standard deviation
            eval_f1_mean = np.mean(eval_f1_values, axis=0)
            eval_f1_std = np.std(eval_f1_values, axis=0)

            # Save everything in the `run_stats` dict
            run_stats[dataset_name][train_dataset_size][acquisition_fun] = {
                "iterations": iterations,
                "mean": eval_f1_mean,
                "std": eval_f1_std,
            }

print("Making plot...")

# Get the size of each dimension
num_datasets = len(run_histories.keys())
num_train_sizes = len(train_sizes)

# The figure is a grid of subplots
fig, axs = plt.subplots(
    num_datasets,
    num_train_sizes,
    sharex=True,
    figsize=(num_datasets * 2, num_train_sizes * 2),
)

# Load the colour map
cmap = mpl.colormaps[cmd_args.colormap]

# Loop through all the dimensions
for y, (dataset_name, sub_dict) in enumerate(run_stats.items()):
    for x, train_dataset_size in enumerate(train_sizes):

        if train_dataset_size in sub_dict:
            sub_sub_dict = sub_dict[train_dataset_size]
        else:
            continue

        ax = axs[y][x]

        # Label every y axis
        ax.set_ylabel("Eval F1")

        for acquisition_fun, stats in sub_sub_dict.items():

            # Select a colour for the run
            colour_index = acquisition_funs.index(acquisition_fun)
            mean_colour = cmap(colour_index)
            error_colour = cmap(colour_index, alpha=0.5)

            # Plot the mean and std of the eval f1 values
            ax.fill_between(
                stats["iterations"],
                y1=stats["mean"] - stats["std"],
                y2=stats["mean"] + stats["std"],
                color=error_colour,
            )
            ax.plot(
                stats["iterations"],
                stats["mean"],
                color=mean_colour,
                label=make_human_readable(acquisition_fun),
            )

# Add the row labels
for ax, dataset_name in zip(axs[:, 0], run_histories.keys()):
    ax.annotate(
        make_human_readable(dataset_name),
        xy=(0, 0.5),
        xytext=(-ax.yaxis.labelpad - 70, 0),
        xycoords="axes fraction",
        textcoords="offset points",
        size="x-large",
        ha="right",
        va="center",
        rotation="vertical",
    )

# Add the column labels
for ax, train_dataset_size in zip(axs[num_datasets - 1], train_sizes):
    ax.set_xlabel("Iteration")
    ax.annotate(
        f"Train size: {train_dataset_size}",
        xy=(0.5, 0),
        xytext=(0, -ax.xaxis.labelpad - 70),
        xycoords="axes fraction",
        textcoords="offset points",
        size="x-large",
        ha="center",
        va="baseline",
    )

# Add a legend
handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="right", fontsize="x-large")

# Set the title
fig.suptitle(cmd_args.chart_title, wrap=True, fontsize="xx-large")

# Adjust the spacing a little bit: this is messed up a little by the
# annotations
fig.subplots_adjust(top=0.94, right=0.9)

plt.show()
