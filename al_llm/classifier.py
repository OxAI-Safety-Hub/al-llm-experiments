# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Union, Any
import configparser

import torch
import wandb
import tempfile
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
import datasets
from al_llm.parameters import Parameters


# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


class Classifier(ABC):
    """Base classifier class

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run

    Attributes
    ----------
    data_handler : DataHandler
        The DataHandler instance attached to this classifier
    """

    def __init__(self, parameters: Parameters, wandb_run: wandb.sdk.wandb_run.Run):
        self.parameters = parameters
        self.wandb_run = wandb_run
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

    @abstractmethod
    def initialise(self):
        """Initialise the model at the beginning of the experiment"""
        pass

    @abstractmethod
    def save(self):
        """Save the classifier, using the wandb_run"""
        pass

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

    def initialise(self):
        pass

    def save(self):
        pass

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
    parameters : Parameters
        The dictionary of parameters for the present experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run

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

    ARTIFACT_NAME = "gpt2-classifier"

    def __init__(self, parameters: Parameters, wandb_run: wandb.sdk.wandb_run.Run):

        # initialises the parameters in the same way as the base class
        super().__init__(parameters, wandb_run)

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

        # set the End of Sentence token as the token used for padding by the model
        self.model.config.pad_token_id = self.model.config.eos_token_id
        # assign the model to the device (CUDA or GPU)
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

        # for each epoch, run the train and eval loops to train the model
        for epoch in range(self.parameters["num_epochs"]):
            print("running epoch " + str(epoch + 1))
            self._train_loop(train_dataloader, lr_scheduler)
            self._eval_loop(eval_dataloader)

    def train_update(
        self,
        dataset_samples: Union[datasets.Dataset, torch.utils.data.Dataset],
    ):

        # If the model is not already loaded then load it
        if self.model is None:
            self._load_model()

        # create dataloaders for the dataset containing all the samples and labels
        # generated by the active learning loop (i.e. human-labelled generated samples)
        # and for the validation dataset
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

    def initialise(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilgpt2", num_labels=2
        )

    def save(self):
        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # store the model in this directory
            file_path = os.path.join(tmpdirname, "model_home.pt")
            self.model.save_pretrained(file_path)

            # upload this model to weights and biases as an artifact
            artifact = wandb.Artifact(self.ARTIFACT_NAME, type="classifier-model")
            artifact.add_dir(tmpdirname)
            self.wandb_run.log_artifact(artifact)

    def _load_model(self):
        """Load the classifier using the wandb_run"""

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # download the model into this directory from wandb
            artifact_path_components = (
                config["Wandb"]["Entity"],
                config["Wandb"]["Project"],
                self.ArtifactName + ":latest",
            )
            artifact_path = "/".join(artifact_path_components)
            artifact = self.wandb_run.use_artifact(
                artifact_path,
                type="classifier-model",
            )
            artifact.download(tmpdirname)

            # load model from this directory
            file_path = os.path.join(tmpdirname, "model_home.pt")
            self.model = AutoModelForSequenceClassification.from_pretrained(file_path)

    def _train_loop(self, train_dataloader, lr_scheduler):
        """Run a native PyTorch training loop

        The loop takes batches from the `train_dataloader` and for each one passes
        the data through the model, calculates the loss, and runs backpropogation
        to adjust the model's parameters according to the optimizer

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            A DataLoader object containing the dataset we want to train the model on
        lr_scheduler : torch.optim.lr_scheduler
            A scheduler to dynamically change the learning rate over multiple epochs
        """

        # set the model to train mode
        self.model.train()

        # iterate over all the batches in the dataloader
        for batch in train_dataloader:

            # move batch data to same device as the model
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # pass the data to the model and calculate the loss and gradients
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            # run backpropogation and step the scheduler along as well
            self.optimizer.step()
            lr_scheduler.step()
            self.optimizer.zero_grad()

    def _eval_loop(self, eval_dataloader):
        """Run a native PyTorch evaluation loop

        The loop takes batches from the `eval_dataloader` and does a forward pass
        through the model, producing some predictions for each and calculating the
        success of the model according to some metrics

        Parameters
        ----------
        eval_dataloader : torch.utils.data.DataLoader
            A DataLoader object containing the dataset we want to evaluate the model on
        """
        # load in the metric(s) - in this case, just accuracy
        metric = datasets.load_metric("accuracy")

        # set the model to eval mode
        self.model.eval()

        # iterate over all the batches in the dataloader
        for batch in eval_dataloader:

            # move batch data to same device as the model
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # pass the data through the model without tracking computations
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits

            # work out the model's predictions for the data using argmax on the logits
            predictions = torch.argmax(logits, dim=-1)

            # give the predictions to the metric(s)
            metric.add_batch(
                predictions=predictions,
                references=batch[config["Data Handling"]["LabelColumnName"]],
            )

        # finally, print the metric's result
        print(metric.compute())

    def tokenize(self, string: str):
        return self.tokenizer(string, padding="max_length", truncation=True)
