# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod


class Classifier(ABC):
    """Base classifier class"""

    @abstractmethod
    def train_afresh(self, dataset):
        """Reset the classifier and fine-tune it anew on tokenised data

        Parameters
        ----------
        dataset : dataset.Dataset or torch.utils.data.Dataset
            The dataset with which to fine-tune
        """
        pass

    @abstractmethod
    def train_update(self, dataset):
        """Fine-tune the classifier on more data tokenised, without resetting

        Parameters
        ----------
        dataset : dataset.Dataset or torch.utils.data.Dataset
            The extra tokenised datapoints with which to fine-tune
        """
        pass

    @abstractmethod
    def tokenize(self, text):
        """Tokenise a string for this classifier

        Parameters
        ----------
        text : str
            The string to tokenize
        """
        return None


class DummyClassifier(Classifier):
    """Dummy classifier, which does nothing"""

    def train_afresh(self, data):
        pass

    def train_update(self, data):
        pass

    def tokenize(self, string):
        return [0]
