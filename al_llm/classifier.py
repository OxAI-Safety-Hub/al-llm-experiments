# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Union, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import datasets


class Classifier(ABC):
    """Base classifier class

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters

    @abstractmethod
    def train_afresh(self, dataset: Union[datasets.Dataset, torch.utils.data.Dataset]):
        """Reset the classifier and fine-tune it anew on tokenised data

        Parameters
        ----------
        dataset : dataset.Dataset or torch.utils.data.Dataset
            The dataset with which to fine-tune
        """
        pass

    @abstractmethod
    def train_update(self, dataset: Union[datasets.Dataset, torch.utils.data.Dataset]):
        """Fine-tune the classifier on more data tokenised, without resetting

        Parameters
        ----------
        dataset : dataset.Dataset or torch.utils.data.Dataset
            The extra tokenised datapoints with which to fine-tune
        """
        pass

    @abstractmethod
    def tokenize(self, text: str):
        """Tokenise a string for this classifier

        Parameters
        ----------
        text : str
            The string to tokenize
        """
        return None


class DummyClassifier(Classifier):
    """Dummy classifier, which does nothing"""

    def train_afresh(self, data: Any):
        pass

    def train_update(self, data: Any):
        pass

    def tokenize(self, string: str):
        return [0]


class GPT2Classifier(Classifier):
    """GPT-2 classifier"""

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        # load model and tokenizer and create an optimizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "gpt2", num_labels=2
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        # set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

    def train_afresh(self, data: Any):
        # reload fresh model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "gpt2", num_labels=2
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # create a dataloader for the train dataset
        train_dataloader = DataLoader(
            data, shuffle=True, batch_size=self.parameters["batch_size"]
        )

        for epoch in range(self.parameters["num_epochs"]):
            print("running epoch " + str(epoch + 1))
            self.__train_loop(train_dataloader)
            self.__eval_loop()

    def __train_loop(self, train_dataloader):
        self.model.train()
        for batch in train_dataloader:
            # move batch data to same device as the model
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

    def __eval_loop(self):
        pass

    def train_update(self, data: Any):
        pass

    def tokenize(self, string: str):
        return self.tokenizer(string, return_tensors="pt")
