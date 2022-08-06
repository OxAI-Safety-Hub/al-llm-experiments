# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Union, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
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
    def train_afresh(
        self,
        dataset_train: Union[datasets.Dataset, torch.utils.data.Dataset],
        dataset_validation: Union[datasets.Dataset, torch.utils.data.Dataset],
    ):
        """Reset the classifier and fine-tune it anew on tokenised data

        Parameters
        ----------
        dataset_train : dataset.Dataset or torch.utils.data.Dataset
            The dataset with which to fine-tune
        dataset_validation : dataset.Dataset or torch.utils.data.Dataset
            The dataset with which to check performance
        """
        pass

    @abstractmethod
    def train_update(
        self,
        dataset_train: Union[datasets.Dataset, torch.utils.data.Dataset],
        dataset_validation: Union[datasets.Dataset, torch.utils.data.Dataset],
    ):
        """Fine-tune the classifier on more data tokenized, without resetting

        Parameters
        ----------
        dataset_train : dataset.Dataset or torch.utils.data.Dataset
            The extra tokenized datapoints with which to fine-tune
        dataset_validation : dataset.Dataset or torch.utils.data.Dataset
            The dataset with which to check performance
        """
        pass

    @abstractmethod
    def tokenize(self, text: Union[str, list]) -> torch.Tensor:
        """Tokenize a string or batch of strings for this classifier

        Parameters
        ----------
        text : str or list
            The string or batch of strings to be tokenized

        Returns
        -------
        tokenized : torch.Tensor
            The result of tokenizing `text`
        """
        return None


class DummyClassifier(Classifier):
    """Dummy classifier, which does nothing"""

    def train_afresh(self, dataset_train: Any, dataset_validation: Any):
        pass

    def train_update(self, dataset_train: Any, dataset_validation: Any):
        pass

    def tokenize(self, string: str):
        return [0]


class GPT2Classifier(Classifier):
    """GPT-2 classifier"""

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        # loads the tokenizer that the model will use
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None  # model is not required until a call to `train_afresh``
        # set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def train_afresh(
        self,
        dataset_train: Union[datasets.Dataset, torch.utils.data.Dataset],
        dataset_validation: Union[datasets.Dataset, torch.utils.data.Dataset],
    ):
        # load a fresh version of the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilgpt2", num_labels=2
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.to(self.device)
        # create an optimizer for the model
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.parameters["learning_rate"]
        )

        # create a dataloader for the train and validation datasets
        train_dataloader = DataLoader(
            dataset_train, shuffle=True, batch_size=self.parameters["batch_size"]
        )
        eval_dataloader = DataLoader(
            dataset_validation, batch_size=self.parameters["batch_size"]
        )

        # create a learning rate scheduler
        num_training_steps = self.parameters["num_epochs"] * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for epoch in range(self.parameters["num_epochs"]):
            print("running epoch " + str(epoch + 1))
            self.__train_loop(train_dataloader, lr_scheduler)
            self.__eval_loop(eval_dataloader)

    def __train_loop(self, train_dataloader, lr_scheduler):
        self.model.train()
        for batch in train_dataloader:
            # move batch data to same device as the model
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            self.optimizer.step()
            lr_scheduler.step()
            self.optimizer.zero_grad()

    def __eval_loop(self, eval_dataloader):
        metric = datasets.load_metric("accuracy")
        self.model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        print(metric.compute())

    def train_update(
        self,
        dataset_samples: Union[datasets.Dataset, torch.utils.data.Dataset],
        dataset_validation: Union[datasets.Dataset, torch.utils.data.Dataset],
    ):

        # create a dataloader for the small samples dataset
        small_samples_dataloader = DataLoader(
            dataset_samples, shuffle=True, batch_size=self.parameters["batch_size"]
        )
        eval_dataloader = DataLoader(
            dataset_validation, batch_size=self.parameters["batch_size"]
        )

        # create a learning rate scheduler, but for one epoch only
        num_training_steps = len(small_samples_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        # only run one epoch of the train and eval loops
        self.__train_loop(small_samples_dataloader, lr_scheduler)
        self.__eval_loop(eval_dataloader)

    def tokenize(self, string: str):

        return self.tokenizer(string, padding="max_length", truncation=True)
