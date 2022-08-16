class Parameters(dict):
    """A sub-class of dict storing the parameters for this experiment.

    All parameters are dummy parameters by default. This means that we take
    small values or basic implementations of components, meant primarily for
    testing.

    To run an actual experiment, one should explicitly specify all the 
    components needed.

    Functions in the same way as a dict. It has the following structure:

        dataset_name : str, default="dummy"
            The name of (hugging face) or path of the (local) dataset.
        num_iterations : int, default=5
            The number of iterations over which to run the active learning.
        refresh_every : int, default=2
            How often to retrain the classifier from scratch.
        batch_size : int, default=8
            Batch size of the dataloaders which the classifier trains from.
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
            True if the experiment should only use dummy values.
        seed : int, default=459834
            The random seed to use for random number generation. The seed is
            set at the beginning of each AL iteration to `seed+iteration`.
        send_alerts : bool, default=True
            True if the experiment should send alerts to the slack channel.
        validation_proportion : float, default=0.2
            Proportion of the training data to be used for validation, if it's not
            provided by the Hugging Face dataset.
        classifier : str, default="DummyClassifier"
            The name of the classifier to use.
        acquisition_function : str, default="DummyAF"
            The name of the acquisition function to use.
        sample_generator : str, default="DummySampleGenerator"
            The name of the sample generator to use.
    """

    # defined default paramets
    default_parameters = {
        "dataset_name": "dummy",
        "num_iterations": 5,
        "refresh_every": 2,
        "batch_size": 8,
        "num_epochs_update": 3,
        "num_epochs_afresh": 3,
        "num_samples": 5,
        "num_warmup_steps": 0,
        "sample_pool_size": 20,
        "learning_rate": 5e-5,
        "dev_mode": False,
        "seed": 459834,
        "send_alerts": False,
        "validation_proportion": 0.2,
        "classifier": "DummyClassifier",
        "acquisition_function": "DummyAF",
        "sample_generator": "DummySampleGenerator",
    }

    def __init__(self, *args, **kw):
        # sets the parameters provided
        super().__init__(*args, **kw)

        # if any of these are not provided, use the default value
        for key, value in self.default_parameters.items():
            if not self.__contains__(key):
                self[key] = value
