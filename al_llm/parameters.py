class Parameters(dict):
    """A sub-class of dict storing the parameters for this experiment, functions
    in the same way as a dict. It has the following sturcture:

        is_dummy : bool
            True if the experiment should only use dummy values.
        num_iterations : int, default=5
            The number of iterations over which to run the active learning.
        refresh_every : int, default=2
            How often to retrain the classifier from scratch.
        num_samples : int, default=5
            Number of samples generated by `sample_generator.generate()`, this is
            the number of samples which will be sent to the human every iteration.
        sample_pool_size : int, default=20
            When using an acquisition function, this is the number of samples
            to generate first, from which the function selects the appropriate
            number."""

    def __init__(self, *args, **kw):
        # sets the parameters provided
        super().__init__(*args, **kw)

        # defined default paramets
        default_parameters = {
            "num_iterations": 5,
            "refresh_every": 2,
            "num_samples": 5,
            "sample_pool_size": 20,
        }

        # if any of these are not provided, use the default value
        for key, value in default_parameters.items():
            if not super().__contains__(key):
                super().__setitem__(key, value)
