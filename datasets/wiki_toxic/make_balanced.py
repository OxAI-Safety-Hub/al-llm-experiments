import pandas as pd
import numpy as np

SPLIT_FILENAME = "train.csv"
OUTPUT_FILENAME = "balanced_train.csv"

# Set the set
np.random.seed(23598)

# load in the csv containing the training data
train_orig = pd.read_csv(SPLIT_FILENAME)

# work out the category with the fewest samples and store how many there are
# of that category
values, counts = np.unique(train_orig["label"].to_numpy(), return_counts=True)
minimum_count = np.amin(counts)

# create a list containing all the subsets of the original dataset, each
# containing `minimum_count` samples of a different category
minimised_subsets = []

# iterate over all categories
for value in values:
    # get the samples from the original dataset that have the label `value`
    train_subset = train_orig[train_orig["label"] == value]

    # take a random sample of size `minimum_count` from this subset
    minimised_subset = train_subset.sample(minimum_count)

    # store this subset in the array position `value`
    minimised_subsets.append(minimised_subset)

# concatenate all the subsets into a new balanced dataset and shuffle
balanced_train = pd.concat(minimised_subsets)
balanced_train = balanced_train.sample(frac=1)

# convert `balanced_train` from a dataframe to a csv file
balanced_train.to_csv(OUTPUT_FILENAME, index=False)
