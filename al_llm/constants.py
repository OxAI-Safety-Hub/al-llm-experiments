# Dataset table column names
TEXT_COLUMN_NAME = "text"
LABEL_COLUMN_NAME = "labels"
AMBIGUITIES_COLUMN_NAME = "ambiguities"

# Seed used for preprocessing datasets
PREPROCESSING_SEED = 7497

# W&B details
WANDB_ENTITY = "oxai-safety-labs-active-learning"

# The names of the W&B projects
WANDB_PROJECTS = {
    "sandbox": "Sandbox",
    "hyperparameter_tuning": "Hyperparameter-Tuning",
    "experiment": "Experiments",
}

# The maximum size to clear the W&B cache down to
CACHE_SIZE = "10GB"

# The metrics to use to evaluate the performance of the classifier
EVALUATE_METRICS = ["accuracy", "f1", "precision", "recall"]
