import pandas as pd

# Load the original train set and the test set
train = pd.read_csv("train_orig.csv")

# Remove any duplicates in the training set
train = train.drop_duplicates(subset=["text"])

train.to_csv("train.csv", index=False)
