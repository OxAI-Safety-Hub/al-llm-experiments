import pandas as pd
import numpy as np

VALIDATION_PROPORTION = 0.1

# Set the seed
np.random.seed(23598)

# Load the original train set and the test set
train_orig = pd.read_csv("train_orig.csv")
test = pd.read_csv("test_orig.csv")


# Replace `` and '' by "
def replace_marks(text: str):
    return text.replace("``", '"').replace("''", '"')


train_orig["text"] = train_orig["text"].map(replace_marks)
test["text"] = test["text"].map(replace_marks)

# Remove any duplicates in the training set
train_orig = train_orig.drop_duplicates(subset=["text"])

# Shuffle
train_orig = train_orig.sample(frac=1)

# Divide into train and validation
division_index = int(len(train_orig) * VALIDATION_PROPORTION)
validation = train_orig[:division_index]
train = train_orig[division_index:]

# Save the new splits
test.to_csv("test.csv", index=False)
train.to_csv("train.csv", index=False)
validation.to_csv("validation.csv", index=False)
