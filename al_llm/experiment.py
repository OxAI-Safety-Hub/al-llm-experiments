from al_llm.data_handler import DummyDataHandler
from al_llm.classifier import DummyClassifier
from al_llm.sample_generator import DummySampleGenerator
from al_llm.interface import CLIInterface


class Experiment:
    """The experiment runner

    Parameters
    ----------
    data_handler : dataset.Dataset
        The starting dataset of labelled samples
    categories : dict
        A dictionary of categories used by the classifier. The keys are the
        names of the categories as understood by the model, and the values
        are the human-readable names.
    classifier : classifier.Classifier
        The classifier instance to use.
    sample_generator : sample_generator.SampleGenerator
        The generator which produces samples for labelling
    interface : interface.Interface
        The interface instance to use
    already_finetuned : bool, default=False
        Is the classifier already fine-tuned on the dataset?
    parameters : dict, optional
        A dictionary of parameters to identify this experiment
    """

    def __init__(
        self,
        data_handler,
        categories,
        classifier,
        sample_generator,
        interface,
        already_finetuned=False,
        parameters=None,
    ):

        # Set the instance attributes
        self.data_handler = data_handler
        self.categories = categories
        self.classifier = classifier
        self.sample_generator = sample_generator
        self.interface = interface
        self.already_finetuned = already_finetuned
        self.parameters = parameters

    def run(self, num_rounds=5, refresh_every=2):
        """Run the experiment

        Parameters
        ----------
        num_rounds : int, default=100
            The number of rounds to run the experiment for.
        refresh_every : int, default=10
            How often to retrain the classifier from scratch
        """

        # Start the interface
        self.interface.begin(parameters=self.parameters)

        # Fine-tune the classifier on the dataset, if necessary
        if not self.already_finetuned:
            self._train_afresh()

        for round in range(num_rounds):

            # Generate some new samples to query
            samples = self.sample_generator.generate()

            # Get the labels from the human
            labels = self.interface.prompt(samples)

            # Update the dataset
            samples_dataset = self.data_handler.new_labelled(samples, labels)

            # Fine-tune, resetting if necessary
            if (round + 1) % refresh_every == 0 and round != 0:
                self._train_afresh()
            else:
                self._train_update(samples_dataset)

        # End the interface
        self.interface.end()

    def _train_afresh(self):
        """Fine-tune the classifier from scratch"""
        self.interface.train_afresh()
        self.classifier.train_afresh(self.data_handler.tokenized_dataset)

    def _train_update(self, samples_dataset):
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

        categories = {"valid": "Valid sentence", "invalid": "Invalid sentence"}
        classifier = DummyClassifier()
        data_handler = DummyDataHandler(classifier)
        sample_generator = DummySampleGenerator()
        interface = CLIInterface(categories)
        parameters = {"is_dummy": True}

        dummy_args = {
            "data_handler": data_handler,
            "categories": categories,
            "classifier": classifier,
            "sample_generator": sample_generator,
            "interface": interface,
            "parameters": parameters,
        }

        return dummy_args