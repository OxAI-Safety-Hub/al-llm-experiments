# from __future__ import annotations

from typing import Optional
import inspect
from argparse import ArgumentParser, Namespace

from al_llm.constants import TAPTED_MODEL_DEFAULT_TAG


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
    num_iterations : int, default=51
        The number of iterations over which to run the active learning.
        Here one iteration is
    refresh_every : int, default=1
        How often to retrain the classifier from scratch. A value of `-1`
        means we never refresh.
    refresh_on_last : bool, default=True
        Whether to refresh the model on the last iteration.
    eval_every : int, default=0
        How often run an eval loop when training. A value of `-1` means we
        never run the eval loop. A value of `0` means we only do it on the
        last epoch per iteration.
    test_every : int, default=-1
        How often run a test loop when training. A value of `-1` means we
        never run the test loop. A value of `0` means we only do it on the
        last epoch per iteration.
    batch_size : int, default=16
        Batch size of the dataloader which the classifier trains from.
    eval_batch_size : int, default=128
        Batch size of the dataloader which the classifier evaluates from.
    num_epochs_update : int, default=3
        The number of epochs to train for when updating the classifier model
        with new datapoints
    num_epochs_afresh : int, default=5
        The number of epochs to train for when training the classifier model
        afresh, starting from scratch
    num_samples : int, default=10
        Number of samples generated by `sample_generator.generate()`, this is
        the number of samples which will be sent to the human every iteration.
    num_warmup_steps : int, default=0
        The number of warmup steps to use in the learning rate scheduler
    sample_pool_size : int, default=1024
        When using an acquisition function, this is the number of samples to
        generate first, from which the function selects the appropriate
        number.
    learning_rate : float, default=5e-5
        Initial learning rate for training.
    dev_mode : bool, default=False
        Whether we're doing some development testing. The current effect is to
        reduce the size of the dataset considerably.
    seed : int, default=459834
        The random seed to use for random number generation. The seed is set
        at the beginning of each AL iteration to `seed+iteration`.
    send_alerts : bool, default=True
        True if the experiment should send alerts to the slack channel.
    validation_proportion : float, default=0.2
        Proportion of the training data to be used for validation, if it's not
        provided by the Hugging Face dataset.
    train_dataset_size : int, default=10
        The size of the initial set of labelled data, for training the
        classifier. A set of this size is selected from the 'train' split of
        the dataset; the rest are collected as a pool of remainder data, used
        by the pool-based simulator.
    full_loop : bool, default=True
        Run the whole experiment in one go, going through all the AL loops.
    supervised : bool, default=False
        Run this experiment using standard supervised learning.
    classifier_base_model : str, default="dummy"
        The name of the base model the classifier should use.
    num_classifier_models : int, default=1
        The number of models to use in the classifier ensemble. The output
        probabilities are the averages of the output probabilities of each
        model.
    acquisition_function : str, default="dummy"
        The name of the acquisition function to use.
    sample_generator_base_model : str, default="dummy"
        The name of the base model the sample generator should use.
    use_tapted_sample_generator : bool, default=False
        True if a pretrained sample generator should be used.
    use_tapted_classifier : bool, default=False
        True if a pretrained classifier should be used.
    tapted_model_version : str, default=TAPTED_MODEL_DEFAULT_TAG
        The artifact version of the tapted model to use.
    use_tbt_sample_generator : bool, default=False
        Whether to use the token-by-token sample generator.
    use_mmh_sample_generator : bool, default=False
        Whether to use the Masked Metropolis-Hastings sample generator.
    sample_generator_temperature : float, default=0.5
        The temperature used when generating new samples
    sample_generator_top_k : int, default=50
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering when doing sample generation
    sample_generator_max_length : int, default=-1
        The maximum length of sentences to generate, in number of tokes. A
        value of -1 means that we use the upper quartile value for the length
        of the tokenized training sentences from the current dataset.
    tbt_pre_top_k : int, default=256
        When doing token-by-token generation, this is the number of tokens
        which get selected to add the uncertainties to. We take the top k
        tokens ordered according to the probability given by the generating
        model. This is done for efficiency reasons, to avoid having to compute
        the uncertainties for every token.
    tbt_uncertainty_weighting : float, default=1
        When doing token-by-token generation, this is the weighting to use
        when adding the uncertainty to the logit value.
    tbt_uncertainty_scheduler : string, default="constant"
        Which scheduler to use to vary the uncertainty weighting throught the
        token-by-token generation process. Possible values are as follows. -
        "constant": Don't vary the weighting - "linear": Start with no
        uncertainty contribution, then increase linearly to reach the maximum
        at `sample_generator_max_length`.
    mmh_num_steps : int, default=50
        When doing Masked Metropolis-Hasting sampling, this is the number of
        steps for which to run the algorithm
    mmh_mask_probability : float, default=0.15
        When doing Masked Metropolis-Hasting sampling, this is the probability
        that each token is masked at each step.
    use_automatic_labeller : bool, default=False
        Whether to use a pretrained classifier to provide the labels, instead
        of a human.
    automatic_labeller_model_name : str,
    default="textattack/roberta-base-rotten-tomatoes"
        The name of the Hugging Face model to use as the automatical labeller.
        This is a model hosted in the Hugging Face repository of models.
    ambiguity_mode : str, default="only_mark"
        How the experiment treat ambiguous data. Default is "only_mark" which
        allows the human to mark data as ambiguous but the experiment will run
        as if it isn't. "none" means the user does not have the choice of
        marking it as ambiguous.
    allow_skipping : bool, default=False
        Whether to allow samples for labelling to be marked as 'to skip',
        rather than being given a label. Skipped samples are not used when
        training the classifier.
    replay_run : str, default=""
        If non-empty, we replay the run with this ID, using the samples and
        labels generated there. Useful to redo the evaluation or testing on a
        particular run.
    use_suggested_labels : bool, default=False,
        Use labels suggested by the sampler generator as the labels for the
        samples, bypassing human labelling. This only makes sense for certain
        sample generators (currently the pool-based one and masked
        Metropolis-Hastings). When `sample_generator_base_model` is set to
        "pool", suggested labels are used regardless of this setting.
    cuda_device : str, default="cuda:0"
        The string specifying the CUDA device to use
    is_running_pytests : bool, default=False
        If true, wandb will be disabled for the test experiments
    save_classifier_every : int, default=-1
        Specifies how often to save the classifier model. A value of -1 means
        that we never save. A value of 0 means that we only save after the
        last iteration. A positive value k means that we save every k
        iterations, and also on the last iteration.
    """

    def __init__(
        self,
        dataset_name: str = "dummy",
        num_iterations: int = 51,
        refresh_every: int = 1,
        refresh_on_last: bool = True,
        eval_every: int = 0,
        test_every: int = -1,
        batch_size: int = 16,
        eval_batch_size: int = 128,
        num_epochs_update: int = 3,
        num_epochs_afresh: int = 5,
        num_samples: int = 10,
        num_warmup_steps: int = 0,
        sample_pool_size: int = 1024,
        learning_rate: float = 5e-5,
        dev_mode: bool = False,
        seed: int = 459834,
        send_alerts: bool = False,
        validation_proportion: float = 0.2,
        train_dataset_size: int = 10,
        full_loop: bool = True,
        supervised: bool = False,
        classifier_base_model: str = "dummy",
        num_classifier_models: int = 1,
        acquisition_function: str = "dummy",
        sample_generator_base_model: str = "dummy",
        use_tapted_sample_generator: bool = False,
        use_tapted_classifier: bool = False,
        tapted_model_version: str = TAPTED_MODEL_DEFAULT_TAG,
        use_tbt_sample_generator: bool = False,
        use_mmh_sample_generator: bool = False,
        sample_generator_temperature: float = 0.5,
        sample_generator_top_k: int = 50,
        sample_generator_max_length: int = -1,
        tbt_pre_top_k: int = 256,
        tbt_uncertainty_weighting: float = 1,
        tbt_uncertainty_scheduler: str = "constant",
        mmh_num_steps: int = 50,
        mmh_mask_probability: float = 0.15,
        use_automatic_labeller: bool = False,
        automatic_labeller_model_name: str = "textattack/roberta-base-rotten-tomatoes",
        ambiguity_mode: str = "only_mark",
        allow_skipping: bool = False,
        replay_run: str = "",
        use_suggested_labels: bool = False,
        cuda_device: str = "cuda:0",
        is_running_pytests: bool = False,
        save_classifier_every: int = -1,
        *args,
        **kwargs,
    ):
        # If we're running supervised learning, we need to run a full loop
        if supervised:
            full_loop = True

        # We can't use both the TBT and MMH sample generators
        if use_tbt_sample_generator and use_mmh_sample_generator:
            raise ValueError(
                "Can't have both parameters 'use_tbt_sample_generator' and "
                "'use_mmh_sample_generator' set to True"
            )

        # sets the parameters provided
        #   'supervised' may override some parameters
        super().__init__(
            dataset_name=dataset_name,
            num_iterations=1 if supervised else num_iterations,
            refresh_every=refresh_every,
            refresh_on_last=refresh_on_last,
            test_every=test_every,
            eval_every=eval_every,
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
            full_loop=full_loop,
            supervised=supervised,
            classifier_base_model=classifier_base_model,
            num_classifier_models=num_classifier_models,
            acquisition_function=acquisition_function,
            sample_generator_base_model=sample_generator_base_model,
            use_tapted_sample_generator=use_tapted_sample_generator,
            use_tapted_classifier=use_tapted_classifier,
            tapted_model_version=tapted_model_version,
            use_tbt_sample_generator=use_tbt_sample_generator,
            use_mmh_sample_generator=use_mmh_sample_generator,
            sample_generator_temperature=sample_generator_temperature,
            sample_generator_top_k=sample_generator_top_k,
            sample_generator_max_length=sample_generator_max_length,
            tbt_pre_top_k=tbt_pre_top_k,
            tbt_uncertainty_weighting=tbt_uncertainty_weighting,
            tbt_uncertainty_scheduler=tbt_uncertainty_scheduler,
            mmh_num_steps=mmh_num_steps,
            mmh_mask_probability=mmh_mask_probability,
            use_automatic_labeller=use_automatic_labeller,
            automatic_labeller_model_name=automatic_labeller_model_name,
            ambiguity_mode=ambiguity_mode,
            allow_skipping=allow_skipping,
            replay_run=replay_run,
            use_suggested_labels=use_suggested_labels,
            cuda_device=cuda_device,
            is_running_pytests=is_running_pytests,
            save_classifier_every=save_classifier_every,
            *args,
            **kwargs,
        )

    def update_from_dict(self, dictionary: dict, *, skip_keys: list = []):
        """Update the parameters using a dictionary

        Parameters
        ----------
        dictionary : dict
            The dictionary of parameters to use to update the values
        skip_keys : list, default = []
            A list of keys in `dictionary` to ignore
        """

        # Get the method signature for the constructor
        signature = inspect.signature(self.__init__)

        # Loop over all the parameters in the signature, and add the
        # corresponding value from `dictionary` if it exists, and it is
        # permitted by `skip_keys` to do so
        for name in dict(signature.parameters).keys():
            if name in dictionary and name not in skip_keys:
                self.__setitem__(name, dictionary[name])

    @classmethod
    def add_to_arg_parser(
        cls,
        parser: ArgumentParser,
        *,
        included_parameters: Optional[list] = None,
        override_defaults_dict: dict = None,
    ):
        """Add the parameters to an ArgumentParser instance

        This adds the possible parameters as command line options, depending
        on type, together with their default values, which can be overridden
        by `defaults`.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to which to add the parameters
        included_parameters : list, optional
            A list of the parameters to include. If `None` then all parameters
            are included.
        override_defaults_dict : dict, default={}
            A dictionary of default values for the parameters, which override
            the ones defined in `Parameters`
        """

        # Get the method signature for the constructor
        signature = inspect.signature(cls.__init__)

        # Loop over all the parameters in the signature
        for name, parameter in dict(signature.parameters).items():
            # We don't want to add the `self` argument, or any of the `*args`
            # or `**kwargs`
            if name in ["self", "args", "kwargs"]:
                continue

            # Only add parameters in `included_parameters`, it it is set
            if included_parameters is not None and name not in included_parameters:
                continue

            # Get the default for this parameter
            if override_defaults_dict is not None and name in override_defaults_dict:
                default = override_defaults_dict[name]
            else:
                default = parameter.default

            if parameter.annotation == bool:
                # For boolean parameters we add two flags, one for true and
                # one for false, then set the default
                cmd_name = "--" + name.replace("_", "-")
                parser.add_argument(
                    cmd_name,
                    dest=name,
                    action="store_true",
                    default=default,
                    help=f"Set parameter {name} to `True`",
                )
                cmd_neg_name = "--no-" + name.replace("_", "-")
                parser.add_argument(
                    cmd_neg_name,
                    dest=name,
                    action="store_false",
                    default=not default,
                    help=f"Set parameter {name} to `False`",
                )

            else:
                # Other parameters are regular arguments
                cmd_name = "--" + name.replace("_", "-")
                parser.add_argument(
                    cmd_name,
                    type=parameter.annotation,
                    default=default,
                    help=f"Set the value of parameter {name}",
                )

    @classmethod
    def from_argparse_namespace(
        cls, namespace: Namespace, *, defaults: dict = {}
    ) -> "Parameters":
        """Build a `Parameters` object from and argparse namespace

        Parameters
        ----------
        namespace : Namespace
            The parsed command line arguments, from an ArgumentParser instance
        defaults : dict, default={}
            A dictionary of default values for the parameters, which override
            the ones defined in `Parameters`

        Returns
        -------
        parameters : Parameters
            The `Parameters` object built from the namespace
        """

        # Get the method signature for the constructor
        signature = inspect.signature(cls.__init__)

        # The arguments which we'll use to construct the `Parameters` instance
        parameters_args = defaults

        # Loop over all the parameters in the signature
        for name in list(signature.parameters):
            # We don't want to add the `self` argument, or any of the `*args`
            # or `**kwargs`
            if name in ["self", "args", "kwargs"]:
                continue

            # Add the value of the argument from the namespace, if it exists
            if name in namespace:
                parameters_args[name] = namespace.__dict__[name]

        # Return the `Parameters` object constructed from the arguments
        return cls(**parameters_args)
