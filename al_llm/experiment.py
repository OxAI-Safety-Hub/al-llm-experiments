from typing import Union

import torch

import datasets

import wandb

from al_llm.data_handler import DataHandler, DummyDataHandler, HuggingFaceDataHandler
from al_llm.classifier import Classifier, DummyClassifier, GPT2Classifier
from al_llm.sample_generator import (
    PlainGPT2SampleGenerator,
    SampleGenerator,
    DummySampleGenerator,
)
from al_llm.acquisition_function import DummyAcquisitionFunction
from al_llm.interface import Interface, CLIInterface
from al_llm.parameters import Parameters


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
        data_handler: DataHandler,
        categories: dict,
        classifier: Classifier,
        sample_generator: SampleGenerator,
        interface: Interface,
        parameters: Parameters,
        already_finetuned: bool = False,
    ):

        # Set the instance attributes
        self.data_handler = data_handler
        self.categories = categories
        self.classifier = classifier
        self.sample_generator = sample_generator
        self.interface = interface
        self.already_finetuned = already_finetuned
        self.parameters = parameters

        # initialise weights and biases
        wandb.init(
            project="Labs_Project_Experiments",
            entity="oxai-safety-labs-active-learning",
        )

    def run(self):
        """Run the experiment"""

        # Start the interface
        self.interface.begin(parameters=self.parameters)

        # Fine-tune the classifier on the dataset, if necessary
        if not self.already_finetuned:
            self._train_afresh()

        for round in range(self.parameters["num_iterations"]):

            # Generate some new samples to query
            samples = self.sample_generator.generate()

            # Get the labels from the human
            labels = self.interface.prompt(samples)

            # Update the dataset
            dataset_samples = self.data_handler.new_labelled(samples, labels)

            # Fine-tune, resetting if necessary
            if self.parameters["refresh_every"] == 1 or (
                (round + 1) % self.parameters["refresh_every"] == 0 and round != 0
            ):
                self._train_afresh()
            else:
                self._train_update(dataset_samples)

        # End the interface
        self.interface.end()

    def _train_afresh(self):
        """Fine-tune the classifier from scratch"""
        self.interface.train_afresh()
        self.classifier.train_afresh(
            self.data_handler.tokenized_train,
        )

    def _train_update(
        self, dataset_samples: Union[datasets.Dataset, torch.utils.data.Dataset]
    ):
        """Fine-tune the classifier with new datapoints, without resetting"""
        self.interface.train_update()
        self.classifier.train_update(
            dataset_samples,
        )

    @classmethod
    def make_dummy_experiment(cls):
        """Get dummy instances to feed into the constructor

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
        categories = {0: "Valid sentence", 1: "Invalid sentence"}
        classifier = DummyClassifier(parameters)
        data_handler = DummyDataHandler(classifier, parameters)
        classifier.attach_data_handler(data_handler)
        acquisition_function = DummyAcquisitionFunction(parameters)
        sample_generator = DummySampleGenerator(
            parameters, acquisition_function=acquisition_function
        )
        interface = CLIInterface(categories)

        dummy_args = {
            "data_handler": data_handler,
            "categories": categories,
            "classifier": classifier,
            "sample_generator": sample_generator,
            "interface": interface,
            "parameters": parameters,
        }

        return dummy_args

    @classmethod
    def make_experiment(cls, dataset_name: str):
        """Get experiment instances to feed into the constructor

        Default setup expects Rotten Tomatoes dataset, and uses a classifier built
        on GPT-2, the HuggingFaceDataHandler, a GPT-2-based sentence generator that
        just produces real sentences, and the Command Line Interface. Also sets
        `dev_mode` parameter to `True`, which reduces size of datasets provided to
        the classifier.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset this experiment should use

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
        categories = {0: "Negative sentence", 1: "Positive sentence"}
        classifier = GPT2Classifier(parameters)
        data_handler = HuggingFaceDataHandler(dataset_name, classifier, parameters)
        classifier.attach_data_handler(data_handler)
        sample_generator = PlainGPT2SampleGenerator(parameters)
        interface = CLIInterface(categories)

        experiment_args = {
            "data_handler": data_handler,
            "categories": categories,
            "classifier": classifier,
            "sample_generator": sample_generator,
            "interface": interface,
            "parameters": parameters,
        }

        return experiment_args
