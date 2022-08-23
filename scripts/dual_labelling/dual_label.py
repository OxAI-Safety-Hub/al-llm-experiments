import argparse

# Parser to pass the run id through to the program
parser = argparse.ArgumentParser(
    description="Pretrain the model on the unlabelled data."
)
parser.add_argument(
    "--run-id",
    type=str,
    help="The run id of the experiment whos added data we should dual label.",
)
args = parser.parse_args()

print(args.run_id)
