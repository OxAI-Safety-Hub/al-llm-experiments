import os
from venv import create
import argparse

# Parser to allow all training arguments to be passed when running this
# script straight from the command line.
parser = argparse.ArgumentParser(
    description="Pretrain the model on the unlabelled data."
)
parser.add_argument(
    "--model-name", type=str, help="The hugging face model name.", required=True
)
parser.add_argument(
    "--dataset-name", type=str, help="The dataset name or path.", required=True
)
parser.add_argument(
    "--batch-size", type=int, help="The batch size for training.", default=4, nargs="?"
)
parser.add_argument(
    "--block-size",
    type=int,
    help="Number of tokens to truncate longer training samples to",
    default=128,
    nargs="?",
)
parser.add_argument("--seed", type=int, help="The seed.", default=327532, nargs="?")
parser.add_argument(
    "--use-balanced-dataset",
    help="Whether to use a balanced version of the training dataset for tapting",
    action="store_true"
)
parser.add_argument(
    "--output-dir",
    type=str,
    help="The directory to store results to",
    default="output_dir",
    nargs="?",
)
parser.add_argument(
    "--num-epochs",
    type=int,
    help="The number of training epochs",
    default=3,
    nargs="?",
)
args = parser.parse_args()

# Create the virtual environment if one does not already
if not os.path.exists("venv"):
    create("venv", with_pip=True)

# Install the requirements
os.system("venv/bin/pip install --upgrade pip")
os.system(
    "venv/bin/pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113"
)
os.system("venv/bin/pip install -r requirements.txt")

# Form the command that calls the run_clm.py script with our
# specified parameters
command_args = [
    "venv/bin/python run_clm.py",
    f"--model_name_or_path {args.model_name}",
    f"--dataset_name {args.dataset_name}",
    f"--dataset_config_name {args.dataset_name}",
    f"--block_size {args.block_size}",
    f"--per_device_train_batch_size {args.batch_size}",
    f"--per_device_eval_batch_size {args.batch_size}",
    f"--seed {args.seed}",
    "--save_total_limit 1",
    "--do_train",
    "--do_eval",
    f"--output_dir {args.output_dir}",
    "--overwrite_output_dir",
    "--logging_strategy epoch",
    "--logging_steps 1",
    f"--num_train_epochs {args.num_epochs}",
]
if args.use_balanced_dataset:
    command_args.append("--use-balanced-dataset")
command = " ".join(command_args)

# Run this command (start training)
os.system(command)
