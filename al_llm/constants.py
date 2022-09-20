# Dataset table column names
TEXT_COLUMN_NAME = "text"
LABEL_COLUMN_NAME = "labels"
AMBIGUITIES_COLUMN_NAME = "ambiguities"

# Seeds used for running experiments
EXPERIMENT_SEEDS = [7086, 3202, 6353, 3437]

# Seed used for preprocessing datasets
PREPROCESSING_SEED = 7497

# Seed for label training and label training project
LABEL_TRAINING_SEED = 42
LABEL_TRAINING_PROJECT = "Labelling-Training"

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

# The default tag for loading tapted models from W&B. Set to "latest" to always
# load the latest model
TAPTED_MODEL_DEFAULT_TAG = "default"
