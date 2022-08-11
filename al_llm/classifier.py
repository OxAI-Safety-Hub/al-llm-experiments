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

    Attributes
    ----------
    data_handler : DataHandler
        The DataHandler instance attached to this classifier
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.data_handler = None

    @abstractmethod
    def train_afresh(
        self,
        dataset_train: Union[datasets.Dataset, torch.utils.data.Dataset],
    ):
        """Reset the classifier and fine-tune it anew on tokenised data

        Parameters
        ----------
        dataset_train : dataset.Dataset or torch.utils.data.Dataset
            The dataset with which to fine-tune
        """
        pass

    @abstractmethod
    def train_update(
        self,
        dataset_train: Union[datasets.Dataset, torch.utils.data.Dataset],
    ):
        """Fine-tune the classifier on more data tokenized, without resetting

        Parameters
        ----------
        dataset_train : dataset.Dataset or torch.utils.data.Dataset
            The extra tokenized datapoints with which to fine-tune
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

    def attach_data_handler(self, data_handler):
        """Attach an instance of a DataHandler to this classifier, so
        validation and test datasets are easily accessible

        Parameters
        ----------
        data_handler : DataHandler
            The instance to be attached to this classifier
        """
        self.data_handler = data_handler


class UncertaintyMixin(ABC):
    """A mixin for classifiers which provide a measure of uncertainty"""

    @abstractmethod
    def calculate_uncertainties(self, samples: Union[str, list]) -> Union[float, list]:
        """Compute the uncertainty of a sample or batch of samples

        Uncertainties are floats, whose interpretations depend on the
        classifier

        Parameters
        ----------
        samples : str or list
            The sample or samples for which to calculate the uncertainty

        Returns
        -------
        uncertainties : float or list
            The uncertainties of the samples. Either a float or a list of
            floats, depending on the type of `samples`.
        """
        pass


class DummyClassifier(UncertaintyMixin, Classifier):
    """Dummy classifier, which does nothing"""

    def train_afresh(self, dataset_train: Any):
        pass

    def train_update(self, dataset_train: Any):
        pass

    def tokenize(self, text: Union[str, list]) -> torch.Tensor:
        if isinstance(text, str):
            return torch.zeros(1)
        elif isinstance(text, list):
            return torch.zeros((1, len(text)))
        else:
            raise TypeError(
                f"Parameter `text` should be string or list, got {type(text)}"
            )

    def calculate_uncertainties(self, samples: Union[str, list]) -> Union[float, list]:
        if isinstance(samples, str):
            return 0
        elif isinstance(samples, list):
            return [0] * len(samples)
        else:
            raise TypeError(
                f"Parameter `samples` must be a string or list, got {type(samples)}"
            )


class GPT2Classifier(Classifier):
    """Classifier class based on GPT-2

    A classifier class that uses the GPT-2[1]_ model available on HuggingFace
    as a foundation for training a classifier; implemented by replacing
    the head of pretrained GPT-2 with a classifier.

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters for the present experiment

    Attributes
    ----------
    data_handler : DataHandler
        The DataHandler instance attached to this classifier
    tokenizer : transformers.AutoTokenizer
        The HuggingFace tokenizer associated with this classifier
    model : transformers.AutoModelForSequenceClassification
        The HuggingFace model for this classifier; reset to a new model every
        call of `train_afresh`
    optimizer : torch.optim.AdamW
        The torch optimizer used to update the model's parameters when training
    device : torch.device
        Set to either cuda (if GPU available) or CPU

    Notes
    ----------
    Temporarily using distilled version of GPT-2 (distilgpt2 on HuggingFace) due
    to excessive use of GPU RAM during testing; planning on returning to the
    larger version later.

    References
    ----------
    [1] Radford et al., "Language Models are Unsupervised Multitask Learners", 2019
    """

    def __init__(self, parameters: dict):

        # initialises the parameters in the same way as the base class
        super().__init__(parameters)

        # loads the tokenizer that the model will use
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # model and optimizer not required until a call to `train_afresh`
        self.model = None
        self.optimizer = None

        # set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def train_afresh(
        self,
        dataset_train: Union[datasets.Dataset, torch.utils.data.Dataset],
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
            self.data_handler.dataset_validation,
            batch_size=self.parameters["batch_size"],
        )

        # create a learning rate scheduler
        num_training_steps = self.parameters["num_epochs"] * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.parameters["num_warmup_steps"],
            num_training_steps=num_training_steps,
        )

        for epoch in range(self.parameters["num_epochs"]):
            print("running epoch " + str(epoch + 1))
            self._train_loop(train_dataloader, lr_scheduler)
            self._eval_loop(eval_dataloader)

    def _train_loop(self, train_dataloader, lr_scheduler):
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

    def _eval_loop(self, eval_dataloader):
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
    ):

        # create a dataloader for the small samples dataset
        small_samples_dataloader = DataLoader(
            dataset_samples, shuffle=True, batch_size=self.parameters["batch_size"]
        )
        eval_dataloader = DataLoader(
            self.data_handler.dataset_validation,
            batch_size=self.parameters["batch_size"],
        )

        # create a learning rate scheduler, but for one epoch only
        num_training_steps = len(small_samples_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.parameters["num_warmup_steps"],
            num_training_steps=num_training_steps,
        )

        # only run one epoch of the train and eval loops
        self._train_loop(small_samples_dataloader, lr_scheduler)
        self._eval_loop(eval_dataloader)

    def tokenize(self, string: str):

        return self.tokenizer(string, padding="max_length", truncation=True)
