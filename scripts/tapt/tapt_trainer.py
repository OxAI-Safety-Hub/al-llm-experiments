import os
from venv import create
import argparse

parser = argparse.ArgumentParser(
    description="Pretrain the model on the unlabelled data."
)
parser.add_argument("--model-name", type=str, help="The hugging face model name.")
parser.add_argument("--dataset-name", type=str, help="The dataset name or path.")
parser.add_argument(
    "--batch-size", type=int, help="The batch size for training.", default=8, nargs="?"
)
parser.add_argument("--seed", type=int, help="The seed.", default=327532, nargs="?")
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
    default="3",
    nargs="?",
)
args = parser.parse_args()

# Create the virtual environment if one does not already
if not os.path.exists("venv"):
    create("venv", with_pip=True)

# Install the requirements
os.system("venv/bin/pip install -r requirements.txt")
os.system(
    "venv/bin/pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
)

command_args = [
    "venv/bin/python run_clm.py",
    f"--model_name_or_path {args.model_name}",
    f"--dataset_name {args.dataset_name}",
    f"--dataset_config_name {args.dataset_name}",
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

command = " ".join(command_args)

os.system(command)
