from al_llm import Experiment, Parameters
from al_llm.experiment import ProjectOption

parameters = Parameters(
    dataset_name="rotten_tomatoes",
    num_iterations=5,
    refresh_every=1,
    batch_size=4,
    num_epochs_update=2,
    num_epochs_afresh=10,
    num_samples=10,
    num_warmup_steps=0,
    sample_pool_size=50,
    learning_rate=5e-5,
    dev_mode=False,
    seed=459834,
    send_alerts=True,
    validation_proportion=0.2,
    train_dataset_size=500,
    full_loop=True,
    supervised=False,
    classifier_base_model="gpt2",
    acquisition_function="max_uncertainty",
    sample_generator_base_model="pool",
    use_tapted_sample_generator=False,
    use_tapted_classifier=True,
    ambiguity_mode="only_mark",
    is_running_pytests=False,
)

args = Experiment.make_experiment(
    parameters, ProjectOption.Experiment, "rotten_tomatoes_pool_2"
)
experiment = Experiment(**args)
experiment.run()
