import datetime

from al_llm import Experiment, Parameters


# Change Parameters to a dataclass with optional and necessary params
# Probably can just move the stuff from make_experiment into the innit
# might need to modify tests
params_max = Parameters(
    dataset_name="wiki_toxic",
    num_iterations=1,
    classifier_base_model="gpt2",
    sample_generator_base_model="pool",
    acquisition_function="max",
    batch_size=8,
    eval_batch_size=32,
    seed=7086,
    tapted_model_version="e8-b16-bl256-bal",  # Where is this happening? will be used in utils.load_tapted_model
    save_classifier_every=0,
    # train_dataset_size=10,
    #
    # # cuda_device="cuda:0",
    # # dataset_name="dummy",
    # refresh_every=1,
    # refresh_on_last=True,
    # eval_every=0,
    # test_every=-1,
    # # batch_size=16,
    # # eval_batch_size=128,
    # num_epochs_update=3,
    # num_epochs_afresh=5,
    # num_samples=10,
    # num_warmup_steps=0,
    # sample_pool_size=1024,
    # learning_rate=5e-5,
    # dev_mode=False,
    # send_alerts=False,
    # validation_proportion=0.2,
    # full_loop=True,
    # supervised=False,
    # # classifier_base_model="dummy" # Can remove and make this necessary arguments,
    # num_classifier_models=1,
    # # acquisition_function="dummy",
    # # sample_generator_base_model="dummy",
    # use_tapted_sample_generator=False,
    # use_tapted_classifier=False,
    # use_tbt_sample_generator=False,
    # use_mmh_sample_generator=False,
    # sample_generator_temperature=0.5,
    # sample_generator_top_k=50,
    # sample_generator_max_length=-1,
    # tbt_pre_top_k=256,
    # tbt_uncertainty_weighting=1,
    # tbt_uncertainty_scheduler="constant",
    # mmh_num_steps=50,
    # mmh_mask_probability=0.15,
    # use_automatic_labeller=False,
    # automatic_labeller_model_name="textattack/roberta-base-rotten-tomatoes",
    # ambiguity_mode="only_mark",
    # allow_skipping=False,
    # replay_run="",
    # use_suggested_labels=False,
    # # cuda_device="cuda:0",
    # is_running_pytests=False,
)


params_rand = Parameters(
    dataset_name="wiki_toxic",
    num_iterations=1,
    classifier_base_model="gpt2",
    # THIS IS WRONG
    sample_generator_base_model="random",
    acquisition_function="max_uncertainty",
    batch_size=8,
    eval_batch_size=32,
    seed=7086,
    tapted_model_version="e8-b16-bl256-bal",
    save_classifier_every=0,
)


datetime_string = datetime.datetime.now().strftime("%Y%m%d_H%M%S")
# Make the experiment
args_max = Experiment.make_experiment(
    parameters=params_max,
    run_id=datetime_string + "_max",
    project_name="Sandbox",
)

experiment_max = Experiment(**args_max)
experiment_max.run()


# Make the experiment
args_rand = Experiment.make_experiment(
    parameters=params_rand,
    run_id=datetime_string + "_rand",
    project_name="Sandbox",
)

experiment_rand = Experiment(**args_rand)
experiment_rand.run()
