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
    PoolSampleGenerator,
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
)
from al_llm.parameters import Parameters


# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


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
    already_finetuned : bool, default=False
        Is the classifier already fine-tuned on the dataset?
    """

    MAP_DATASET_CONTAINER = {
        "dummy": DummyDatasetContainer,
        "rotten_tomatoes": RottenTomatoesDatasetHandler,
    }
    MAP_CLASSIFIER = {
        "DummyClassifier": DummyClassifier,
        "GPT2Classifier": GPT2Classifier,
    }
    MAP_ACQUISITION_FUNCTION = {
        "None": None,
        "DummyAF": DummyAF,
        "RandomAF": RandomAF,
        "MaxUncertaintyAF": MaxUncertaintyAF,
    }
    MAP_SAMPLE_GENERATOR = {
        "DummySampleGenerator": DummySampleGenerator,
        "PlainGPT2SampleGenerator": PlainGPT2SampleGenerator,
        "PoolSampleGenerator": PoolSampleGenerator,
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

        # Save the current version of the classifier and dataset, including
        # the new samples awaiting labels from the human
        self._save(samples)

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

    def _save(self, unlabelled_samples: list = []):
        """Save the current classifier and dataset"""
        self.classifier.save()
        self.data_handler.save(unlabelled_samples)

    @classmethod
    def make_experiment(
        cls,
        parameters: Parameters,
        run_id: str,
        is_running_pytests: bool = False,
    ):
        """Get experiment instances to feed into the constructor

        Default setup expects Rotten Tomatoes dataset, and uses a classifier built
        on GPT-2, the HuggingFaceDataHandler, a GPT-2-based sentence generator that
        produces real sentences and filters using a maximum uncertainty acquisition
        function, and the Command Line Interface.

        Parameters
        ----------
        parameters : Parameters
            The dictionary of parameters for the present experiment
        run_id : str
            The ID of the current run
        is_running_pytests: bool, default=False
            If true, wandb will be disabled for the test experiments

        Returns
        -------
        experiment_args : dict
            A dictionary of the non-optional arguments for `Experiment`

        Notes
        -----

        By default all parameters are dummy parameters. This means that calling
        this method with a plain `Parameters` instance will create a dummy
        experiment.

        Example
        -------
        Make a dummy experiment
        >>> parameters = Parameters()
        >>> args = Experiment.make_experiment(parameters, "dummy")
        >>> experiment = Experiment(**args)
        """

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

        # Set up the dataset_container
        dc_class = cls.MAP_DATASET_CONTAINER[parameters["dataset_name"]]
        dataset_container = dc_class(parameters)

        # Set up the classifier
        classifier_name = parameters["classifier"]
        classifier = cls.MAP_CLASSIFIER[classifier_name](
            parameters, dataset_container, wandb_run
        )

        # Set up the data handler
        data_handler = DataHandler(parameters, dataset_container, classifier, wandb_run)

        # Set up the acquisition function (could be None)
        af_name = parameters["acquisition_function"]
        af_class = cls.MAP_ACQUISITION_FUNCTION[af_name]
        if af_class is None:
            acquisition_function = None
        elif af_name == "MaxUncertaintyAF":
            acquisition_function = af_class(parameters, classifier)
        else:
            acquisition_function = af_class(parameters)

        # Set up the sample generator
        sg_name = parameters["sample_generator"]
        if sg_name == "PoolSampleGenerator":
            sample_generator = cls.MAP_SAMPLE_GENERATOR[sg_name](
                parameters, acquisition_function, dataset_container
            )
        else:
            sample_generator = cls.MAP_SAMPLE_GENERATOR[sg_name](
                parameters, acquisition_function=acquisition_function
            )

        # Set up the interface
        if sg_name == "PoolSampleGenerator":
            interface = PoolSimulatorInterface(dataset_container, wandb_run)
        elif parameters["full_loop"]:
            interface = CLIInterface(dataset_container, wandb_run)
        else:
            interface = CLIBrokenLoopInterface(dataset_container, wandb_run)

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
