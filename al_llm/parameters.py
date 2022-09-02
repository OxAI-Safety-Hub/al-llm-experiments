class Parameters(dict):
    """A sub-class of dict storing the parameters for this experiment.

    All parameters are dummy parameters by default. This means that we take
    small values or basic implementations of components, meant primarily for
    testing.

    To run an actual experiment, one should explicitly specify all the
    components needed.

    Functions in the same way as a dict.

    Parameters
    ----------
    dataset_name : str, default="dummy"
        The name of the hugging face dataset.
    num_iterations : int, default=5
        The number of iterations over which to run the active learning.
    refresh_every : int, default=2
        How often to retrain the classifier from scratch.
    batch_size : int, default=8
        Batch size of the dataloader which the classifier trains from.
    eval_batch_size : int, default=8
        Batch size of the dataloader which the classifier evaluates from.
    num_epochs_update : int, default=3
        The number of epochs to train for when updating the classifier
        model with new datapoints
    num_epochs_afresh : int, default=3
        The number of epochs to train for when training the classifier
        model afresh, starting from scratch
    num_samples : int, default=5
        Number of samples generated by `sample_generator.generate()`, this is
        the number of samples which will be sent to the human every iteration.
    num_warmup_steps : int, default=0
        The number of warmup steps to use in the learning rate scheduler
    sample_pool_size : int, default=20
        When using an acquisition function, this is the number of samples
        to generate first, from which the function selects the appropriate
        number.
    learning_rate : float, default=5e-5
        Initial learning rate for training.
    dev_mode : bool, default=False
        Whether we're doing some development testing. The current effect
        is to reduce the size of the dataset considerably.
    seed : int, default=459834
        The random seed to use for random number generation. The seed is
        set at the beginning of each AL iteration to `seed+iteration`.
    send_alerts : bool, default=True
        True if the experiment should send alerts to the slack channel.
    validation_proportion : float, default=0.2
        Proportion of the training data to be used for validation, if it's not
        provided by the Hugging Face dataset.
    train_dataset_size : int, default=10
        The size of the initial set of labelled data, for training the
        classifier. A set of this size is selected from the 'train' split
        of the dataset; the rest are collected as a pool of remainder data,
        used by the pool-based simulator.
    full_loop : bool, default=True
        Run the whole experiment in one go, going through all the AL loops.
    supervised : bool, default=False
        Run this experiment using standard supervised learning.
    classifier_base_model : str, default="dummy"
        The name of the base model the classifier should use.
    acquisition_function : str, default="dummy"
        The name of the acquisition function to use.
    sample_generator_base_model : str, default="dummy"
        The name of the base model the sample generator should use.
    use_tapted_sample_generator : bool, default=False
        True if a pretrained sample generator should be used.
    use_tapted_classifier : bool, default=False
        True if a pretrained classifier should be used.
    sample_generator_temperature : float, default=1.0
        The temperature used when generating new samples
    sample_generator_top_k : int, default=50
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering when doing sample generation
    ambiguity_mode : str, default="only_mark"
        How the experiment treat ambiguous data. Default is "only_mark" which
        allows the human to mark data as ambiguous but the experiment will
        run as if it isn't. "none" means the user does not have the choice of
        marking it as ambiguous.
    cuda_device : str, default="cuda:0"
        The string specifying the CUDA device to use
    is_running_pytests : bool, default=False
        If true, wandb will be disabled for the test experiments
    save_classifier_every : int, default=0
        Specifies how often to save the classifier model. A value of -1 means
        that we never save. A value of 0 means that we only save after the
        last iteration. A positive value k means that we save every k
        iterations, and also on the last iteration.
    """

    def __init__(
        self,
        dataset_name="dummy",
        num_iterations=5,
        refresh_every=2,
        batch_size=8,
        eval_batch_size=8,
        num_epochs_update=3,
        num_epochs_afresh=3,
        num_samples=5,
        num_warmup_steps=0,
        sample_pool_size=20,
        learning_rate=5e-5,
        dev_mode=False,
        seed=459834,
        send_alerts=False,
        validation_proportion=0.2,
        train_dataset_size=10,
        full_loop=True,
        supervised=False,
        classifier_base_model="dummy",
        acquisition_function="dummy",
        sample_generator_base_model="dummy",
        use_tapted_sample_generator=False,
        use_tapted_classifier=False,
        sample_generator_temperature=1.0,
        sample_generator_top_k=50,
        ambiguity_mode="only_mark",
        cuda_device="cuda:0",
        is_running_pytests=False,
        save_classifier_every=0,
        *args,
        **kwargs,
    ):

        # sets the parameters provided
        #   'supervised' may override some parameters
        super().__init__(
            dataset_name=dataset_name,
            num_iterations=1 if supervised else num_iterations,
            refresh_every=refresh_every,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_epochs_update=num_epochs_update,
            num_epochs_afresh=num_epochs_afresh,
            num_samples=num_samples,
            num_warmup_steps=num_warmup_steps,
            sample_pool_size=sample_pool_size,
            learning_rate=learning_rate,
            dev_mode=dev_mode,
            seed=seed,
            send_alerts=send_alerts,
            validation_proportion=validation_proportion,
            train_dataset_size=train_dataset_size,
            full_loop=True if supervised else full_loop,
            supervised=supervised,
            classifier_base_model=classifier_base_model,
            acquisition_function=acquisition_function,
            sample_generator_base_model=sample_generator_base_model,
            use_tapted_sample_generator=use_tapted_sample_generator,
            use_tapted_classifier=use_tapted_classifier,
            sample_generator_temperature=sample_generator_temperature,
            sample_generator_top_k=sample_generator_top_k,
            ambiguity_mode=ambiguity_mode,
            cuda_device=cuda_device,
            is_running_pytests=is_running_pytests,
            save_classifier_every=save_classifier_every,
            *args,
            **kwargs,
        )
