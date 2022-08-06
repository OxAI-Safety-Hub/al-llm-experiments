from typing import Union

import torch

import datasets

from al_llm.data_handler import DataHandler, DummyDataHandler, HuggingFaceDataHandler
from al_llm.classifier import Classifier, DummyClassifier, GPT2Classifier
from al_llm.sample_generator import (
    PlainGPT2SampleGenerator,
    SampleGenerator,
    DummySampleGenerator,
)
from al_llm.interface import Interface, CLIInterface


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
    parameters : dict
        A dictionary of parameters to identify this experiment
    already_finetuned : bool, default=False
        Is the classifier already fine-tuned on the dataset?

    Attributes
    ----------
    parameters : dict
        A dictionary of the parameters for this experiment. It has the
        following structure.
        num_iterations : int, default=5
            The number of iterations over which to run the active learning
        refresh_every : int, default=2
            How often to retrain the classifier from scratch
        batch_size : int, default=8
            Batch size of the dataloaders which the classifier trains from
        num_epochs : int, default=3
            The number of epochs that are run between each active learning loop
        num_samples : int, default=5
            Number of samples generated by `sample_generator.generate()`, this is
            the number of samples which will be sent to the human every iteration
        learning_rate : float, default=5e-5
            Initial learning rate for training
    """

    default_parameters = {
        "num_iterations": 5,
        "refresh_every": 2,
        "batch_size": 8,
        "num_epochs": 3,
        "num_samples": 5,
        "learning_rate": 5e-5,
        "dev_mode": False,
    }

    def __init__(
        self,
        data_handler: DataHandler,
        categories: dict,
        classifier: Classifier,
        sample_generator: SampleGenerator,
        interface: Interface,
        parameters: dict,
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

        # Set default values for some parameters
        for name, value in self.default_parameters.items():
            if name not in self.parameters:
                self.parameters[name] = value

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
            self.data_handler.tokenized_validation,
        )

    def _train_update(
        self, dataset_samples: Union[datasets.Dataset, torch.utils.data.Dataset]
    ):
        """Fine-tune the classifier with new datapoints, without resetting"""
        self.interface.train_update()
        self.classifier.train_update(
            dataset_samples,
            self.data_handler.tokenized_validation,
        )

    @classmethod
    def make_dummy_experiment(self):
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

        parameters = {"dev_mode": True}
        categories = {"valid": "Valid sentence", "invalid": "Invalid sentence"}
        classifier = DummyClassifier(parameters)
        data_handler = DummyDataHandler(classifier, parameters)
        sample_generator = DummySampleGenerator(parameters)
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
    def make_experiment(self, dataset_name: str):
        """Get experiment instances to feed into the constructor

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

        parameters = {"dev_mode": True}
        categories = {"neg": "Negative sentence", "pos": "Positive sentence"}
        classifier = GPT2Classifier(parameters)
        data_handler = HuggingFaceDataHandler(dataset_name, classifier, parameters)
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
