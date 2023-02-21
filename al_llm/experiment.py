import datasets
from transformers import set_seed

import wandb

from al_llm.data_handler import DataHandler
from al_llm.dataset_container import (
    DatasetContainer,
    DummyDatasetContainer,
    RottenTomatoesDatasetContainer,
    WikiToxicDatasetContainer,
    PubMed20kRCTDatasetContainer,
    Trec6DatasetContainer,
)
from al_llm.classifier import (
    Classifier,
    DummyClassifier,
    PlainGPT2Classifier,
    PlainDistilGPT2Classifier,
    PlainBERTClassifier,
    TAPTGPT2Classifier,
    TAPTDistilGPT2Classifier,
    TAPTBERTClassifier,
)
from al_llm.sample_generator import (
    PlainGPT2SampleGenerator,
    SampleGenerator,
    DummySampleGenerator,
    TAPTGPT2SampleGenerator,
    TAPTDistilGPT2SampleGenerator,
    PoolSampleGenerator,
    ReplaySampleGenerator,
    PlainGPT2TokenByTokenSampleGenerator,
    TAPTGPT2TokenByTokenSampleGenerator,
    PlainBERTMaskedMHSampleGenerator,
    TAPTBERTMaskedMHSampleGenerator,
)
from al_llm.acquisition_function import (
    DummyAF,
    MaxUncertaintyAF,
    RandomAF,
)
from al_llm.interface import (
    CLIBrokenLoopInterface,
    Interface,
    CLIInterface,
    PoolSimulatorInterface,
    AutomaticLabellerInterface,
    ReplayInterface,
)
from al_llm.parameters import Parameters
from al_llm.constants import (
    TEXT_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    AMBIGUITIES_COLUMN_NAME,
    WANDB_ENTITY,
    CACHE_SIZE,
    DEFAULT_REPLAY_SKIP_KEYS,
)
from al_llm.utils import UnlabelledSamples


class NotHappyToResumeError(Exception):
    """Exception raised when user is not happy to resume existant run"""

    pass


class Experiment:
    """The experiment runner

    Parameters
    ----------
    parameters : Parameters
        A dictionary of parameters to identify this experiment
    dataset_container : DatasetContainer
        The dataset container to use
    data_handler : DataHandler
        The data handler to use
    classifier : Classifier
        The classifier instance to use.
    sample_generator : SampleGenerator
        The generator which produces samples for labelling
    interface : Interface
        The interface instance to use
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    """

    MAP_DATASET_CONTAINER = {
        "dummy": DummyDatasetContainer,
        "rotten_tomatoes": RottenTomatoesDatasetContainer,
        "wiki_toxic": WikiToxicDatasetContainer,
        "pubmed_20k_rct": PubMed20kRCTDatasetContainer,
        "trec6": Trec6DatasetContainer,
    }
    MAP_PLAIN_CLASSIFIER = {
        "dummy": DummyClassifier,
        "gpt2": PlainGPT2Classifier,
        "distilgpt2": PlainDistilGPT2Classifier,
        "bert": PlainBERTClassifier,
    }
    MAP_TAPT_CLASSIFIER = {
        "gpt2": TAPTGPT2Classifier,
        "distilgpt2": TAPTDistilGPT2Classifier,
        "bert": TAPTBERTClassifier,
    }
    MAP_ACQUISITION_FUNCTION = {
        "none": None,
        "dummy": DummyAF,
        "random": RandomAF,
        "max_uncertainty": MaxUncertaintyAF,
    }
    MAP_PLAIN_SAMPLE_GENERATOR = {
        "dummy": DummySampleGenerator,
        "gpt2": PlainGPT2SampleGenerator,
    }
    MAP_TAPT_SAMPLE_GENERATOR = {
        "distilgpt2": TAPTDistilGPT2SampleGenerator,
        "gpt2": TAPTGPT2SampleGenerator,
    }
    MAP_TBT_PLAIN_SAMPLE_GENERATOR = {
        "gpt2": PlainGPT2TokenByTokenSampleGenerator,
    }
    MAP_TBT_TAPT_SAMPLE_GENERATOR = {
        "gpt2": TAPTGPT2TokenByTokenSampleGenerator,
    }
    MAP_MMH_PLAIN_SAMPLE_GENERATOR = {
        "bert": PlainBERTMaskedMHSampleGenerator,
    }
    MAP_MMH_TAPT_SAMPLE_GENERATOR = {
        "bert": TAPTBERTMaskedMHSampleGenerator,
    }

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        data_handler: DataHandler,
        classifier: Classifier,
        sample_generator: SampleGenerator,
        interface: Interface,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):
        # Set the instance attributes
        self.parameters = parameters
        self.dataset_container = dataset_container
        self.data_handler = data_handler
        self.classifier = classifier
        self.sample_generator = sample_generator
        self.interface = interface
        self.wandb_run = wandb_run

    def run(self):
        # Clean up to stop the wandb cache from overfilling
        self._clear_cache()

        if self.parameters["full_loop"]:
            self._run_full()
        else:
            self._run_single_iteration()

    def _run_full(self):
        """Run the whole experiment in one go, through all iterations"""

        # Start the user interface
        self.interface.begin()

        for iteration in range(self.parameters["num_iterations"]):
            # Perform a single iteration of model update
            self._train_classifier_on_current_data(iteration)

            is_last_iteration = iteration != self.parameters["num_iterations"] - 1

            if not is_last_iteration:
                # Obtain new samples to label
                samples = self._get_samples()

                if not self.parameters["supervised"]:
                    # Get the labels from the human
                    prompt_output = self.interface.prompt(samples)

                    # Add these samples to the dataset
                    self.data_handler.new_labelled(samples, prompt_output)

            # Save the current version of the classifier and dataset
            self._save(iteration)

        # End the interface
        self.interface.end()

        # Alert the slack channel that the experiment is complete
        if self.parameters["send_alerts"]:
            wandb.alert(
                title="Full Experiment Complete",
                text="The `run_full()` experiment has been completed.",
            )

        # End the W&B run
        self.wandb_run.finish()

    def _run_single_iteration(self):
        """Run a single iteration of active learning"""

        # Start the interface
        self.interface.begin()

        # determine the current iteration
        try:
            iteration = wandb.run.summary["iteration"] + 1
        except KeyError:
            # if no iteration has been logged already this must be 0
            wandb.log({"iteration": 0})
            iteration = 0

        # If `iteration` != 0, load any data saved on WandB and prompt the
        # human for labels for generated sentences from the previous iteration
        if iteration != 0:
            added_data = self._load_and_prompt()

            # add the additional data into the local training datasets
            self.data_handler.new_labelled(
                added_data[TEXT_COLUMN_NAME],
                added_data[LABEL_COLUMN_NAME],
                added_data[AMBIGUITIES_COLUMN_NAME],
            )

        # Perform a single iteration of model update, obtaining new samples
        # to label
        self._train_classifier_on_current_data(iteration)
        samples = self._get_samples()

        # Save the current version of the classifier and dataset to wandb, including
        # the new samples awaiting labels from the human
        self._save(iteration, samples)

        # Alert the slack channel that the iteration is complete
        if self.parameters["send_alerts"]:
            wandb.alert(
                title="AL Loop Iteration Complete",
                text="There is new data to be labelled.",
            )

    def _load_and_prompt(self):
        """Load dataset from WandB and prompt human for labels

        Load in the dataset stored on Weights and Biases, prompt the human for
        labels for any unlabelled sentences generated in the previous iteration,
        and then return the dictionary containing all extra data.

        Returns
        ----------
        added_data : dict
            Dictionary containing sentences and labels not in the original
            dataset (i.e. added by Active Learning loop)
        """
        # Load the data from WandB
        added_data = self.data_handler.load()

        # Get the unlabelled sentences saved to WandB by taking the
        # last `num_samples` items from added_data's 'text' column
        unlabelled_added = added_data[TEXT_COLUMN_NAME][
            -self.parameters["num_samples"] :
        ]

        # Prompt the human for labels
        prompt_output = self.interface.prompt(unlabelled_added)

        # Append these labels onto the end of the added_data
        added_data[LABEL_COLUMN_NAME].extend(prompt_output.labels)
        added_data[AMBIGUITIES_COLUMN_NAME].extend(prompt_output.ambiguities)

        # Return the added_data dataset
        return added_data

    def _clear_cache(self):
        """Clear some space in the Weights and Biases cache"""

        # Get the artifacts cache and clear it down to a size of 5GB
        c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
        c.cleanup(wandb.util.from_human_size(CACHE_SIZE))

    def _train_classifier_on_current_data(self, iteration: int):
        """Train the classifier with the latest datapoints.
        Depending on the value of self.parameters["refresh_every"] and iteration,
            we may also retrain the classifier on the whole dataset
            (rather than just new datapoints)

        Parameters
        ----------
        iteration : int
            The index of the current iteration number, starting with 0
        """

        # Set the random number seed, so that the experiment is
        # reproducible whether we do full loop AL or broken loop
        set_seed(self.parameters["seed"] + iteration)

        dataset_samples = self.data_handler.get_latest_tokenized_datapoints()

        # Train either a fresh model or update the existing one. If this is
        # the last iteration of this experiment, it will end on a call to
        # _train_afresh.
        if (
            iteration == 0
            or (
                self.parameters["refresh_every"] != -1
                and iteration % self.parameters["refresh_every"] == 0
            )
            or (
                iteration + 1 == self.parameters["num_iterations"]
                and self.parameters["refresh_on_last"]
            )
        ):
            self._train_afresh(dataset_samples, iteration)
        else:
            self._train_update(dataset_samples, iteration)

    def _get_samples(self) -> UnlabelledSamples:
        """Get new samples from the sample generator

        Returns
        -------
        samples : UnlabelledSamples
            The latest samples for labelling
        """

        # If performing supervised learning, skip the sample generation
        if self.parameters["supervised"]:
            return []

        # Generate some new samples to query
        samples = self.sample_generator.generate()

        return samples

    def _train_afresh(
        self,
        dataset_samples: datasets.Dataset,
        iteration: int,
    ):
        """Fine-tune the classifier from scratch"""
        self.interface.train_afresh(iteration=iteration)
        self.classifier.train_afresh(
            self.dataset_container.tokenized_train,
            iteration,
            new_tokenized_samples=dataset_samples,
        )

    def _train_update(
        self,
        dataset_samples: datasets.Dataset,
        iteration: int,
    ):
        """Fine-tune the classifier with new datapoints, without resetting"""
        self.interface.train_update(iteration=iteration)
        self.classifier.train_update(
            dataset_samples,
            iteration,
        )

    def _save(
        self,
        iteration: int,
        unlabelled_samples: UnlabelledSamples = UnlabelledSamples(),
    ):
        """Save the current classifier and dataset to wandb"""

        # Only save the classifier if we are at the correct iteration according
        # to the 'save_classifier_every' parameter
        save_classifier_every = self.parameters["save_classifier_every"]
        iteration_max = self.parameters["num_iterations"] - 1
        if (
            not self.parameters["full_loop"]
            or (save_classifier_every > 0 and iteration % save_classifier_every == 0)
            or (save_classifier_every >= 0 and iteration == iteration_max)
        ):
            self.classifier.save()

        # Always save the new samples though
        self.data_handler.save(unlabelled_samples)

    @classmethod
    def make_experiment(
        cls,
        parameters: Parameters,
        project_name: str,
        run_id: str,
        *,
        tags: list = [],
        replay_skip_keys=DEFAULT_REPLAY_SKIP_KEYS,
    ) -> dict:
        """Get experiment instances to feed into the constructor

        This method sets up all the experiment components according to the
        `Parameters` instance. It also initialised the Weights and Biases
        logging.

        Parameters
        ----------
        parameters : Parameters
            The dictionary of parameters for the present experiment
        project_name : str
            The wandb project which this experiment should be logged to
        run_id : str
            The ID of the current run
        tags : list, default=[]
            A list of tags to associate to the W&B run
        replay_skip_keys : list, default=DEFAULT_REPLAY_SKIP_KEYS
            When replaying a run, these are the parameter keys which are not
            copied from the replayed run.

        Returns
        -------
        experiment_args : dict
            A dictionary of the non-optional arguments for `Experiment`

        Notes
        -----

        By default all parameters are dummy parameters. This means that
        calling this method with a plain `Parameters` instance will create a
        dummy experiment.

        Example
        -------
        Make a dummy experiment
        >>> parameters = Parameters()
        >>> args = Experiment.make_experiment(parameters, "dummy")
        >>> experiment = Experiment(**args)
        """

        # Whether we're doing a replay run
        do_replay_run = parameters["replay_run"] != ""

        if do_replay_run:
            # Build up the full path, using the entity and project name, if
            # not already specified
            replay_run_path = parameters["replay_run"]
            if replay_run_path.count("/") == 0:
                replay_run_path = "/".join(
                    [WANDB_ENTITY, project_name, replay_run_path]
                )
            elif replay_run_path.count("/") == 1:
                replay_run_path = "/".join([WANDB_ENTITY, replay_run_path])

            # Print a message to say that we're replaying a run
            print()
            print("+" * 79)
            text = f"+ Replaying run: {replay_run_path}"
            print(text + " " * max(0, 78 - len(text)) + "+")
            print("+" * 79)
            print()

            # Get the run to replay
            api = wandb.Api()
            replayed_run = api.run(replay_run_path)

            # Update the parameters to match
            print("Updating parameters to match replayed run...")
            parameters.update_from_dict(
                replayed_run.config,
                skip_keys=replay_skip_keys,
            )

        # initialise weights and biases
        #   Set resume to allow which resumes the previous run if there is already
        #   a run with the id `run_id`.
        #   Set mode to disabled when running pytests so that a login is not required
        #   for the program to run.
        wandb_run = wandb.init(
            project=project_name,
            entity=WANDB_ENTITY,
            resume="allow",
            id=run_id,
            tags=tags,
            mode="disabled" if parameters["is_running_pytests"] else "online",
        )

        # Ensure that if a run is being resumed, it is intentional
        if wandb_run.resumed:
            print("WARNING: Resuming an already existant run.")
            if do_replay_run:
                print("Moreover, this is a replay run.")
            happy_to_continue = False
            while not happy_to_continue:
                choice = input("Do you want to continue? (Y/n): ")
                if choice.lower() == "n":
                    raise NotHappyToResumeError
                elif choice.lower() == "y" or choice.lower() == "":
                    happy_to_continue = True

        # Set the seed now, because the data handler may do some shuffling
        set_seed(parameters["seed"])

        # Set up the dataset_container
        dc_class = cls.MAP_DATASET_CONTAINER[parameters["dataset_name"]]
        dataset_container = dc_class(parameters)

        # Set up the classifier
        classifier_model_name = parameters["classifier_base_model"]
        if parameters["use_tapted_classifier"]:
            classifier = cls.MAP_TAPT_CLASSIFIER[classifier_model_name](
                parameters, dataset_container, wandb_run
            )
        else:
            classifier = cls.MAP_PLAIN_CLASSIFIER[classifier_model_name](
                parameters, dataset_container, wandb_run
            )

        # Set up the data handler
        data_handler = DataHandler(parameters, dataset_container, classifier, wandb_run)

        # Load the dataset extension from the replayed run, if we're doing that
        if do_replay_run:
            data_handler.load_replay_dataset_extension(replayed_run)

        # Set up the acquisition function (could be None)
        af_name = parameters["acquisition_function"]
        af_class = cls.MAP_ACQUISITION_FUNCTION[af_name]
        if af_class is None or do_replay_run:
            acquisition_function = None
        elif af_name == "max_uncertainty":
            acquisition_function = af_class(parameters, classifier)
        else:
            acquisition_function = af_class(parameters)

        # Set up the sample generator
        sg_model_name = parameters["sample_generator_base_model"]
        if do_replay_run:
            sample_generator = ReplaySampleGenerator(
                parameters=parameters,
                dataset_container=dataset_container,
                wandb_run=wandb_run,
                data_handler=data_handler,
            )
        elif sg_model_name == "pool":
            sample_generator = PoolSampleGenerator(
                parameters=parameters,
                dataset_container=dataset_container,
                wandb_run=wandb_run,
                acquisition_function=acquisition_function,
            )
        elif (
            parameters["use_tbt_sample_generator"]
            and not parameters["use_tapted_sample_generator"]
        ):
            sample_generator = cls.MAP_TBT_PLAIN_SAMPLE_GENERATOR[sg_model_name](
                parameters=parameters,
                classifier=classifier,
                dataset_container=dataset_container,
                wandb_run=wandb_run,
                acquisition_function=acquisition_function,
            )
        elif (
            parameters["use_tbt_sample_generator"]
            and parameters["use_tapted_sample_generator"]
        ):
            sample_generator = cls.MAP_TBT_TAPT_SAMPLE_GENERATOR[sg_model_name](
                parameters=parameters,
                classifier=classifier,
                dataset_container=dataset_container,
                wandb_run=wandb_run,
                acquisition_function=acquisition_function,
            )
        elif (
            parameters["use_mmh_sample_generator"]
            and not parameters["use_tapted_sample_generator"]
        ):
            sample_generator = cls.MAP_MMH_PLAIN_SAMPLE_GENERATOR[sg_model_name](
                parameters=parameters,
                classifier=classifier,
                dataset_container=dataset_container,
                wandb_run=wandb_run,
                acquisition_function=acquisition_function,
            )
        elif (
            parameters["use_mmh_sample_generator"]
            and parameters["use_tapted_sample_generator"]
        ):
            sample_generator = cls.MAP_MMH_TAPT_SAMPLE_GENERATOR[sg_model_name](
                parameters=parameters,
                classifier=classifier,
                dataset_container=dataset_container,
                wandb_run=wandb_run,
                acquisition_function=acquisition_function,
            )
        elif parameters["use_tapted_sample_generator"]:
            sample_generator = cls.MAP_TAPT_SAMPLE_GENERATOR[sg_model_name](
                parameters=parameters,
                dataset_container=dataset_container,
                wandb_run=wandb_run,
                acquisition_function=acquisition_function,
            )
        else:
            sample_generator = cls.MAP_PLAIN_SAMPLE_GENERATOR[sg_model_name](
                parameters=parameters,
                dataset_container=dataset_container,
                wandb_run=wandb_run,
                acquisition_function=acquisition_function,
            )

        # Update the W&B config with TAPT parameters, if we're using a TAPTed
        # sample generator
        if parameters["use_tapted_sample_generator"] and not do_replay_run:
            tapt_parameters = sample_generator.get_training_parameters()
            wandb.config.update({"tapt_sample_generator": tapt_parameters})

        # Set up the interface
        if do_replay_run:
            interface = ReplayInterface(
                parameters, dataset_container, wandb_run, data_handler
            )
        elif parameters["use_automatic_labeller"]:
            interface = AutomaticLabellerInterface(
                parameters,
                dataset_container,
                wandb_run,
                parameters["automatic_labeller_model_name"],
            )
        elif sg_model_name == "pool" or parameters["use_suggested_labels"]:
            interface = PoolSimulatorInterface(parameters, dataset_container, wandb_run)
        elif parameters["full_loop"]:
            interface = CLIInterface(parameters, dataset_container, wandb_run)
        else:
            interface = CLIBrokenLoopInterface(parameters, dataset_container, wandb_run)

        # Log the parameters to the run as it's config. If resuming the run and the
        #   parameters do not match, it will correctly throw an error.
        wandb.config.update(parameters)

        experiment_args = {
            "parameters": parameters,
            "dataset_container": dataset_container,
            "data_handler": data_handler,
            "classifier": classifier,
            "sample_generator": sample_generator,
            "interface": interface,
            "wandb_run": wandb_run,
        }

        return experiment_args
