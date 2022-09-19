import re
import math

import pandas as pd

from tqdm import tqdm

TOXIC_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Time and date regexes
TIME = r"([0-9]{1,2}:[0-9]{2}( (am|AM|pm|PM))?)"
DAY = r"([23]?(1(st)?|2(nd)?|3(rd)?|[4-9](th)?)|1[0-9](th)?)"
MONTH = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Nov|Dec)"
YEAR = r"('?[0-9]{2}|[0-9]{4})"
DATE = rf"(({DAY} {MONTH}|{MONTH} {DAY})(,? {YEAR})?)"
TIMESTAMP = rf"((({TIME},? (\(UTC\) )?)?{DATE}|({DATE},? )?{TIME})(\s+\(UTC\))?)"

# The 'talk' part at the end of a signature
TALK = r"((\|\s*|\(\s*)?[tT]alk((\s*[-|•, ]\s*|\s+)[cC]ontribs)?(\s*[-|)])?)"

# IP addresses
IP = r"([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})"

# Username and the username part of a the signature
USERNAME = r"([^#<>[\]|{}/@\s]+)"
USER_SIG = rf"((((?:\s)[-–—]\s*)?(\((User:)?{USERNAME}\)|User:{USERNAME})|(?:\s)[-–—]\s*{USERNAME})(\s+{TALK})?)"

# A full signature
SIGNATURE = rf"(((([-–—]\s*)?{IP}(\s+{USER_SIG})?|(?:\s)[-–—]\s*[uU]nsigned|{TALK}|{USER_SIG})(\s+{TIMESTAMP})?)|{TIMESTAMP}(\s+{TALK})?)"

# List of the patterns to remove
REGEX_REMOVE = [
    r"^(\"+|'+)",  # Initial quotation marks
    r"(\"+|'+)$",  # Final quotation marks
    r"^REDIRECT.*$",  # The whole comment is a redirect
    rf"^\s*{SIGNATURE}",  # Initial signature
    rf"{SIGNATURE}\s*$",  # Final signature
    r" \[[0-9]+\]|\[[0-9]+\] ",  # Citations
    r"‖\s+[tT]alk - [-a-zA-Z0-9._()\s]+‖",
    r"==[^=]+==",
    r"^::+",
    r"^\s*\(UTC\)",
    rf"Unblock {IP}",
    r"2nd Unblock Request",
    r":Category:",
    r"File:[^\s]+",
    r"\{\|.+\|\}",  # Embedded code
    # r"\{\{.+\s.+\}\}", # Embedded code
    r"^\s+",  # Initial whitespace
    r"\s+$",  # Trailing whitespace
]

# List of patterns to replaces
REGEX_REPLACE = {
    "\n+": "\n",
    "\\'": "'",
    '""+': '"',
    "''+": "'",
    # r"(WP|Wikipedia):[^\s]+": "URL", # Wikipedia internal links
    r"[^\s]+#[^\s]+": "URL",  # Wikipedia internal links
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)": "URL",  # ULRs
    r"([uU]ser_[tT]alk|[tT]alk):[^\s]+": "URL",  # Talk links
}


def clean_sentence(sentence):
    """Preprocess a sentence using the regex rules"""
    for pattern in REGEX_REMOVE:
        sentence = re.sub(pattern, "", sentence)
    for pattern, repl in REGEX_REPLACE.items():
        sentence = re.sub(pattern, repl, sentence)
    return sentence


def make_binary_label(row):
    """Make a row label binary by combining all toxicity types"""
    for column in TOXIC_COLUMNS:
        if row[column] == 1:
            return 1
    return 0


print("Loading original data...")

# Load up the original data
train_df = pd.read_csv("orig_train.csv").set_index("id")
test_text_df = pd.read_csv("orig_test.csv").set_index("id")
test_labels_df = pd.read_csv("orig_test_labels.csv").set_index("id")

# Remove the datapoints which have no label
test_text_df = test_text_df.loc[test_labels_df["toxic"] != -1]
test_labels_df = test_labels_df.loc[test_labels_df["toxic"] != -1]

# Join the test text and labels to make a complete dataset
test_df = test_text_df.join(test_labels_df)

print("Cleaning train split...")
for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
    row["comment_text"] = clean_sentence(row["comment_text"])

print("Cleaning test split...")
for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
    row["comment_text"] = clean_sentence(row["comment_text"])


# Some texts will get reduced to the empty string. Let's remove them first
print("Removing empty texts...")
train_df = train_df.loc[train_df["comment_text"] != ""]
test_df = test_df.loc[test_df["comment_text"] != ""]

# Get rid of any duplicates we made
print("Removing duplicate entries...")
train_df = train_df.drop_duplicates(subset=["comment_text"])
test_df = test_df.drop_duplicates(subset=["comment_text"])

print("Creating binary column...")

# Make the new binary column
train_df["label"] = train_df.apply(make_binary_label, axis=1)
test_df["label"] = test_df.apply(make_binary_label, axis=1)

# Remove all other classification columns
train_df = train_df.drop(columns=TOXIC_COLUMNS)
test_df = test_df.drop(columns=TOXIC_COLUMNS)

print("Creating eval split...")

# Shuffle the current train split
train_df = train_df.sample(frac=1)

# The new size of the train split
train_size = math.floor(len(train_df) * 0.8)

# Separate into train and eval splits
eval_df = train_df[train_size:]
train_df = train_df[:train_size]

# print("Saving to disk...")
with open("train.csv", "w") as f:
    train_df.to_csv(f)
with open("eval.csv", "w") as f:
    eval_df.to_csv(f)
with open("test.csv", "w") as f:
    test_df.to_csv(f)
