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

# Saving and loading constants: Dataset Extensions
DATASET_EXT_ARTIFACT_TYPE = "dataset-extension"
DATASET_EXT_DATASET_FILE_NAME = "added_data.json"
DATASET_EXT_PREFIX = "de_"

# Saving and loading constants: Classifier Models
CLASSIFIER_ARTIFACT_TYPE = "classifier-model"
CLASSIFIER_MODEL_FILE_NAME = "model_home.pt"

# Saving and loading constants: Dual Labelling Results
DUAL_LAB_ARTIFACT_TYPE = "dual-label-results"
DUAL_LAB_LABELS_FILE_NAME = "labels.json"
DUAL_LAB_RESULTS_FILE_NAME = "results.json"

# Saving and loading constants: TAPT Models
TAPT_PROJECT_NAME = "TAPT-Models"
TAPT_ARTIFACT_TYPE = "TAPT-model"
TAPT_MODEL_FILE_NAME = "model_home.pt"
TAPT_PARAMETERS_FILE_NAME = "parameters_home.json"
