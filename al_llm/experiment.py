import string
from typing import Union
import configparser

import torch

import datasets
from transformers import set_seed

import wandb

from al_llm.data_handler import DataHandler
from al_llm.dataset_container import (
    DatasetContainer,
    DummyDatasetContainer,
    RottenTomatoesDatasetHandler,
)
from al_llm.classifier import Classifier, DummyClassifier, GPT2Classifier
from al_llm.sample_generator import (
    PlainGPT2SampleGenerator,
    SampleGenerator,
    DummySampleGenerator,
)
from al_llm.acquisition_function import (
    DummyAF,
    MaxUncertaintyAF,
)
from al_llm.interface import CLIBrokenLoopInterface, Interface, CLIInterface
from al_llm.parameters import Parameters


# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


class Experiment:
    """The experiment runner

    Parameters
    ----------
    data_handler : DataHandler
        The starting dataset of labelled samples
    categories : dict
        A dictionary of categories used by the classifier. The keys are the
        names of the categories as understood by the model, and the values
        are the human-readable names.
    classifier : Classifier
        The classifier instance to use.
    sample_generator : SampleGenerator
        The generator which produces samples for labelling
    interface : Interface
        The interface instance to use
    parameters : Parameters
        A dictionary of parameters to identify this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    already_finetuned : bool, default=False
        Is the classifier already fine-tuned on the dataset?


    Attributes
    ----------
    categories : dict
        A dictionary containing the categories used in the dataset labels column
        (as an `int`) which are the keys for the human-readable versions of each
        (as a `str`)
    """

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        data_handler: DataHandler,
        classifier: Classifier,
        sample_generator: SampleGenerator,
        interface: Interface,
        wandb_run: wandb.sdk.wandb_run.Run,
        already_finetuned: bool = False,
    ):

        # Set the instance attributes
        self.parameters = parameters
        self.dataset_container = dataset_container
        self.data_handler = data_handler
        self.classifier = classifier
        self.sample_generator = sample_generator
        self.interface = interface
        self.wandb_run = wandb_run
        self.already_finetuned = already_finetuned

    def run_full(self):
        """Run the whole experiment in one go, through all iterations"""

        # Start the interface
        self.interface.begin(parameters=self.parameters)

        for iteration in range(self.parameters["num_iterations"]):

            # Perform a single iteration of model update, obtaining new samples
            # to label
            samples = self._train_and_get_samples(iteration)

            # Get the labels from the human
            labels = self.interface.prompt(samples)

            # Add these samples to the dataset
            self.data_handler.new_labelled(samples, labels)

            # Save the current version of the classifier and dataset
            self._save()

        # End the interface
        self.interface.end()

        # Alert the slack channel that the experiment is complete
        if self.parameters["send_alerts"]:
            wandb.alert(
                title="Full Experiment Complete",
                text="The `run_full()` experiment has been completed.",
            )

    def run_single_iteration(self, iteration: int):
        """Run a single iteration of active learning

        Parameters
        ----------
        iteration : int
            The index of the current iteration number, starting with 0
        """

        # Start the interface
        self.interface.begin(parameters=self.parameters)

        # Perform a single iteration of model update, obtaining new samples
        # to label
        samples = self._train_and_get_samples(iteration)

        # Make a request for labels from the human
        self.data_handler.make_label_request(samples)

        # Save the current version of the classifier and dataset
        self._save()

        # Alert the slack channel that the iteration is complete
        if self.parameters["send_alerts"]:
            wandb.alert(
                title="AL Loop Iteration Complete",
                text="There is new data to be labelled.",
            )

    def _train_and_get_samples(self, iteration: int) -> list:
        """Train the classifier with the latest datapoints, and get new samples

        Parameters
        ----------
        iteration : int
            The index of the current iteration number, starting with 0

        Returns
        -------
        samples : list
            The latest samples for labelling
        """

        # Set the random number seed, so that the experiment is
        # reproducible whether we do full loop AL or broken loop
        set_seed(self.parameters["seed"] + iteration)

        dataset_samples = self.data_handler.get_latest_tokenized_datapoints()

        # Produce the latest classifier
        if iteration == 0:
            self.classifier.initialise()
        elif iteration % self.parameters["refresh_every"] == 0:
            self._train_afresh(iteration)
        else:
            self._train_update(dataset_samples, iteration)

        # Generate some new samples to query
        samples = self.sample_generator.generate()

        return samples

    def _train_afresh(self, iteration: int):
        """Fine-tune the classifier from scratch"""
        self.interface.train_afresh()
        self.classifier.train_afresh(
            self.dataset_container.tokenized_train,
            iteration,
        )

    def _train_update(
        self,
        dataset_samples: Union[datasets.Dataset, torch.utils.data.Dataset],
        iteration: int,
    ):
        """Fine-tune the classifier with new datapoints, without resetting"""
        self.interface.train_update()
        self.classifier.train_update(
            dataset_samples,
            iteration,
        )

    def _save(self):
        """Save the current classifier and dataset"""
        self.classifier.save()
        self.data_handler.save()

    @classmethod
    def make_dummy_experiment(
        cls,
        run_id: string,
        full_loop=True,
        is_running_pytests: bool = False,
    ):
        """Get dummy instances to feed into the constructor

        Parameters
        ----------
        run_id : str
            The ID of the current run
        full_loop : bool, default=True
            Design the experiment to run the full loop of active learning

        Returns
        -------
        dummy_args : dict
            A dictionary of the non-optional arguments for `Experiment`,
            whose values are dummy instances

        Example
        -------
        >>> dummy_args = Experiment.make_dummy_experiment()
        >>> experiment = Experiment(**dummy_args)
        """

        parameters = Parameters(dev_mode=True)

        # initialise weights and biases
        #   Set resume to allow which resumes the previous run if there is already
        #   a run with the id `run_id`.
        #   Set mode to disabled when running pytests so that a login is not required
        #   for the program to run.
        wandb_run = wandb.init(
            project=config["Wandb"]["Project"],
            entity=config["Wandb"]["Entity"],
            resume="allow",
            id=run_id,
            mode="disabled" if is_running_pytests else "online",
            config=parameters,
        )

        # Set the seed now, because the data handler may do some shuffling
        set_seed(parameters["seed"])

        dataset_container = DummyDatasetContainer(parameters)
        classifier = DummyClassifier(parameters, dataset_container, wandb_run)
        data_handler = DataHandler(parameters, dataset_container, classifier, wandb_run)
        acquisition_function = DummyAF(parameters)
        sample_generator = DummySampleGenerator(
            parameters, acquisition_function=acquisition_function
        )
        if full_loop:
            interface = CLIInterface(dataset_container, wandb_run)
        else:
            interface = CLIBrokenLoopInterface(dataset_container, wandb_run)

        dummy_args = {
            "parameters": parameters,
            "dataset_container": dataset_container,
            "data_handler": data_handler,
            "classifier": classifier,
            "sample_generator": sample_generator,
            "interface": interface,
            "wandb_run": wandb_run,
        }

        return dummy_args

    @classmethod
    def make_experiment(
        cls,
        run_id: str,
        is_running_pytests: bool = False,
    ):
        """Get experiment instances to feed into the constructor

        Default setup expects Rotten Tomatoes dataset, and uses a classifier built
        on GPT-2, the HuggingFaceDataHandler, a GPT-2-based sentence generator that
        produces real sentences and filters using a maximum uncertainty acquisition
        function, and the Command Line Interface. Also sets `dev_mode` parameter to
        `True`, which reduces size of datasets provided to the classifier.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset this experiment should use
        run_id : str
            The ID of the current run

        Returns
        -------
        experiment_args : dict
            A dictionary of the non-optional arguments for `Experiment`

        Example
        -------
        >>> experiment_args = Experiment.make_experiment("rotten_tomatoes")
        >>> experiment = Experiment(**experiment_args)
        """

        parameters = Parameters(dev_mode=True)

        # initialise weights and biases
        #   Set resume to allow which resumes the previous run if there is already
        #   a run with the id `run_id`.
        #   Set mode to disabled when running pytests so that a login is not required
        #   for the program to run.
        wandb_run = wandb.init(
            project=config["Wandb"]["Project"],
            entity=config["Wandb"]["Entity"],
            resume="allow",
            id=run_id,
            mode="disabled" if is_running_pytests else "online",
            config=parameters,
        )

        # Set the seed now, because the data handler may do some shuffling
        set_seed(parameters["seed"])

        dataset_container = RottenTomatoesDatasetHandler(parameters)
        classifier = GPT2Classifier(parameters, dataset_container, wandb_run)
        data_handler = DataHandler(parameters, dataset_container, classifier, wandb_run)
        acquisition_function = MaxUncertaintyAF(parameters, classifier)
        sample_generator = PlainGPT2SampleGenerator(
            parameters, acquisition_function=acquisition_function
        )
        interface = CLIInterface(dataset_container, wandb_run)

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
