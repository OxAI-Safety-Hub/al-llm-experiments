from random import randrange

from al_llm import Experiment, Parameters

# The number of experiments of each type to run
num_experiments_each = 4

# The random seeds
seeds = [randrange(100000) for i in range(num_experiments_each)]

# Common parameters
parameters = Parameters(
    classifier="gpt2",
    dataset_name="rotten_tomatoes",
    sample_generator_base_model="pool",
    train_dataset_size=1000,
    num_samples=50,
    batch_size=4,
    num_iterations=10,
)

# Two kinds of experiment: never-refresh and always-refresh
for name, refresh_every in (("never-refresh", 100), ("always-refresh", 1)):
    for i in range(num_experiments_each):

        parameters["seed"] = seeds[i]
        parameters["refresh_every"] = refresh_every

        args = Experiment.make_experiment(parameters, f"pool-test-b-{name}-{i}")
        experiment = Experiment(**args)
        experiment.run_full()
