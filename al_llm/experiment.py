from typing import Union

import torch

import datasets

from al_llm.data_handler import DataHandler, DummyDataHandler
from al_llm.classifier import Classifier, DummyClassifier
from al_llm.sample_generator import SampleGenerator, DummySampleGenerator
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
    """

    default_parameters = {"num_iterations": 5, "refresh_every": 2}

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
            samples_dataset = self.data_handler.new_labelled(samples, labels)

            # Fine-tune, resetting if necessary
            if (round + 1) % self.parameters["refresh_every"] == 0 and round != 0:
                self._train_afresh()
            else:
                self._train_update(samples_dataset)

        # End the interface
        self.interface.end()

    def _train_afresh(self):
        """Fine-tune the classifier from scratch"""
        self.interface.train_afresh()
        self.classifier.train_afresh(self.data_handler.tokenized_dataset)

    def _train_update(
        self, samples_dataset: Union[datasets.Dataset, torch.utils.data.Dataset]
    ):
        """Fine-tune the classifier with new datapoints, without resetting"""
        self.interface.train_update()
        self.classifier.train_update(samples_dataset)

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

        parameters = {"is_dummy": True}
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
