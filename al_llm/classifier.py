# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Union, Any
import configparser
import tempfile
import os
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
import datasets
import evaluate

import wandb

from tqdm import tqdm

from al_llm.parameters import Parameters
from al_llm.dataset_container import DatasetContainer


# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")


class Classifier(ABC):
    """Base classifier class

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for the present experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    """

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):
        self.parameters = parameters
        self.dataset_container = dataset_container
        self.wandb_run = wandb_run

    @abstractmethod
    def train_afresh(
        self,
        tokenized_train: Union[datasets.Dataset, torch.utils.data.Dataset],
        iteration: int,
    ):
        """Reset the classifier and fine-tune it anew on tokenised data

        Parameters
        ----------
        tokenized_train : dataset.Dataset or torch.utils.data.Dataset
            The dataset with which to fine-tune
        iteration : int
            The index of the current iteration of the AL loop
        """
        pass

    @abstractmethod
    def train_update(
        self,
        tokenized_samples: Union[datasets.Dataset, torch.utils.data.Dataset],
        iteration: int,
    ):
        """Fine-tune the classifier on more data tokenized, without resetting

        Parameters
        ----------
        tokenized_samples : dataset.Dataset or torch.utils.data.Dataset
            The extra tokenized datapoints with which to fine-tune
        iteration : int
            The index of the current iteration of the AL loop
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

    def train_afresh(self, tokenized_train: Any, iteration: int):
        pass

    def train_update(self, tokenized_samples: Any, iteration: int):
        pass

    def tokenize(self, text: Union[str, list]) -> torch.Tensor:
        if isinstance(text, str):
            return {"input_ids": [0], "attention_mask": [0]}
        elif isinstance(text, list):
            return {"input_ids": [0] * len(text), "attention_mask": [0] * len(text)}
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


class HuggingFaceClassifier(UncertaintyMixin, Classifier):
    """A classifier using a Hugging Face model

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The container for the datasets in this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    model_name : str
        The name of the model, as on Hugging Face

    Attributes
    ----------
    tokenizer : transformers.AutoTokenizer
        The HuggingFace tokenizer associated with this classifier
    model : transformers.AutoModelForSequenceClassification
        The HuggingFace model for this classifier; reset to a new model every
        call of `train_afresh`
    optimizer : torch.optim.AdamW
        The torch optimizer used to update the model's parameters when training
    device : torch.device
        Set to either cuda (if GPU available) or CPU
    """

    ARTIFACT_NAME = "classifier"

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        model_name: str,
    ):

        # initialises the parameters in the same way as the base class
        super().__init__(parameters, dataset_container, wandb_run)

        # Set the model name
        self.model_name = model_name

        # loads the tokenizer that the model will use
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set up the Hugging Face metric evaluator
        metrics = config["Wandb"]["EvaluateMetrics"].replace(" ", "").split(",")
        self.evaluator = evaluate.combine(metrics)

        # model not required until a call to `train_afresh`
        self.model = None

        # set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def train_afresh(
        self,
        tokenized_train: Union[datasets.Dataset, torch.utils.data.Dataset],
        iteration: int,
    ):

        # Get a fresh version of the model
        self._load_fresh_model()

        # create a dataloader for the train dataset
        train_dataloader = DataLoader(
            tokenized_train, shuffle=True, batch_size=self.parameters["batch_size"]
        )

        # Run the training loop
        self._train(train_dataloader, self.parameters["num_epochs_afresh"], iteration)

    def train_update(
        self,
        tokenized_samples: Union[datasets.Dataset, torch.utils.data.Dataset],
        iteration: int,
    ):

        # If the model is not already loaded then load it
        if self.model is None:
            self._load_model_from_wandb()

        # Make a smaple loader from the latest batch of labelled samples
        samples_dataloader = DataLoader(
            tokenized_samples, shuffle=True, batch_size=self.parameters["batch_size"]
        )

        # Run the training loop
        self._train(samples_dataloader, self.parameters["num_epochs_update"], iteration)

    def initialise(self):
        self._load_fresh_model()

    def save(self):
        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # store the model in this directory
            file_path = os.path.join(
                tmpdirname, config["Classifier Loading"]["ModelFileName"]
            )
            self.model.save_pretrained(file_path)

            # upload this model to weights and biases as an artifact
            artifact = wandb.Artifact(
                self.ARTIFACT_NAME, type=config["Classifier Loading"]["ClassifierType"]
            )
            artifact.add_dir(tmpdirname)
            self.wandb_run.log_artifact(artifact)

    def _load_fresh_model(self):
        """Load the classifier model afresh"""

        # Delete the old model to free up memory
        del self.model

        # load a fresh version of the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

        # Setup the model
        self._setup_model()

    def _load_model_from_wandb(self):
        """Load the classifier using the wandb_run"""

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # download the model into this directory from wandb
            artifact_path_components = (
                config["Wandb"]["Entity"],
                config["Wandb"]["Project"],
                self.ARTIFACT_NAME + ":latest",
            )
            artifact_path = "/".join(artifact_path_components)
            artifact = self.wandb_run.use_artifact(
                artifact_path,
                type=config["Classifier Loading"]["ClassifierType"],
            )
            artifact.download(tmpdirname)

            # load model from this directory
            file_path = os.path.join(
                tmpdirname, config["Classifier Loading"]["ModelFileName"]
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(file_path)

        # Setup the model
        self._setup_model()

    def _setup_model(self):
        """Perform some intial setup on the classifier model"""

        # set the End of Sentence token as the token used for padding by the model
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # assign the model to the device (CUDA or GPU)
        self.model.to(self.device)

    def _train(self, train_dataloader: DataLoader, num_epochs: int, iteration: int):

        # create an optimizer for the model
        optimizer = AdamW(self.model.parameters(), lr=self.parameters["learning_rate"])

        # The eval dataloader
        eval_dataloader = DataLoader(
            self.dataset_container.tokenized_validation,
            batch_size=self.parameters["batch_size"],
        )

        # create a learning rate scheduler
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=self.parameters["num_warmup_steps"],
            num_training_steps=num_training_steps,
        )

        # for each epoch, run the train and eval loops to train the model
        for epoch in range(num_epochs):

            # Output the current epoch
            print()
            print(f"--- Epoch: {epoch+1} ---")

            # Run the training and evaluation loops, obtaining the metrics
            print("- Running train loop")
            train_metrics = self._train_epoch(train_dataloader, optimizer, lr_scheduler)
            print(
                f"Train loss: {train_metrics['loss']:.8}; "
                f"train accuracy: {train_metrics['accuracy']:.6%}"
            )
            print("- Running eval loop")
            eval_metrics = self._eval_epoch(eval_dataloader)
            print(
                f"Eval loss: {eval_metrics['loss']:.8}; "
                f"eval accuracy: {eval_metrics['accuracy']:.6%}"
            )

            # Record the metrics with W&B
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "iteration": iteration,
                    "train": train_metrics,
                    "eval": eval_metrics,
                }
            )

    def _train_epoch(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
    ) -> dict:
        """Run a native PyTorch training loop for one epoch

        The loop takes batches from the `train_dataloader` and for each one passes
        the data through the model, calculates the loss, and runs backpropogation
        to adjust the model's parameters according to the optimizer.

        The metrics and loss are computed while running the training loop, and
        returned as a dictionary.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            A DataLoader object containing the dataset we want to train the model on
        optimizer : torch.optim.Optimizer
            The pytorch optimizer to use
        lr_scheduler : torch.optim._LRScheduler
            A scheduler to dynamically change the learning rate over multiple epochs

        Returns
        -------
        train_metrics : dict
            A dictionary of metrics for the training loop. Consists of the
            metrics computed by `self.evaluator`, plus "loss".
        """

        # set the model to train mode
        self.model.train()

        # Recording the training loss by accumulating across batches
        train_loss = 0

        # iterate over all the batches in the dataloader
        for batch in tqdm(train_dataloader):

            # move batch data to same device as the model
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # pass the data to the model
            outputs = self.model(**batch)

            # work out the model's predictions for the data using argmax on
            # the logits
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Use these outputs to calculate metrics
            self.evaluator.add_batch(
                predictions=predictions,
                references=batch[config["Data Handling"]["LabelColumnName"]],
            )

            # Compute the loss
            loss = outputs.loss

            # Add it to the accumulated loss
            train_loss += loss.item() * len(batch)

            # Perform back-propagation
            loss.backward()

            # run and optimisation step and move the lr scheduler forward
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Compute all the metrics for this epoch
        train_metrics = self.evaluator.compute()

        # Compute the average loss for this epoch
        train_loss /= len(train_dataloader.dataset)
        train_metrics["loss"] = train_loss

        return train_metrics

    def _eval_epoch(self, eval_dataloader: torch.utils.data.DataLoader) -> dict:
        """Run a native PyTorch evaluation loop for one epoch

        The loop takes batches from the `eval_dataloader` and does a forward pass
        through the model, producing some predictions for each and calculating the
        success of the model according to some metrics

        Parameters
        ----------
        eval_dataloader : torch.utils.data.DataLoader
            A DataLoader object containing the dataset we want to evaluate the model on

        Returns
        -------
        eval_metrics : dict
            A dictionary of metrics for the evaluation loop. Consists of the
            metrics computed by `self.evaluator`, plus "loss".
        """

        # set the model to eval mode
        self.model.eval()

        # Recording the evaluation loss by accumulating across batches
        eval_loss = 0

        # iterate over all the batches in the dataloader
        for batch in tqdm(eval_dataloader):

            # move batch data to same device as the model
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # pass the data through the model without tracking computations
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits

            # work out the model's predictions for the data using argmax on the logits
            predictions = torch.argmax(logits, dim=-1)

            # give the predictions to the metric(s)
            self.evaluator.add_batch(
                predictions=predictions,
                references=batch[config["Data Handling"]["LabelColumnName"]],
            )

            # Compute the loss
            loss = outputs.loss

            # Add it to the accumulated loss
            eval_loss += loss.item() * len(batch)

        # Compute all the metrics for this epoch
        eval_metrics = self.evaluator.compute()

        # Compute the average loss for this epoch
        eval_loss /= len(eval_dataloader.dataset)
        eval_metrics["loss"] = eval_loss

        return eval_metrics

    def tokenize(
        self, string: str, padding="max_length", truncation=True, *args, **kwargs
    ):
        return self.tokenizer(
            string, padding=padding, truncation=truncation, *args, **kwargs
        )

    def calculate_uncertainties(self, samples: Union[str, list]) -> Union[float, list]:

        # Turn samples into a list if it isn't already
        if isinstance(samples, str):
            samples = [str]
            return_string = True
        else:
            return_string = False

        # Tokenize the samples, ready for feeding into the model
        tokenized_samples_dict = self.tokenize(samples)
        tokenized_samples = datasets.Dataset.from_dict(tokenized_samples_dict)
        tokenized_samples.set_format("torch", columns=["input_ids", "attention_mask"])

        # Put them in a PyTorch dataloader
        samples_dataloader = DataLoader(
            tokenized_samples, batch_size=self.parameters["batch_size"]
        )

        # set the model to eval mode
        self.model.eval()

        # A list of the uncertainties (entropies) for each element of `samples`
        uncertainties = []

        # Print a message to say what we're doing
        print()
        print("Computing uncertainties...")

        # iterate over all the batches in the dataloader
        for batch in tqdm(samples_dataloader):

            # Move the batch to the appropriate device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():

                # Get the raw model output logits
                outputs = self.model(**batch)
                logits = outputs.logits

                # Compute the class probabilities
                probabilities = F.softmax(logits, dim=1)

                # Compute the entropies per element
                per_class_entropies = torch.special.entr(probabilities)

                # Take the sum over the entropies per class, to yield the
                # total entropy per sample
                sum_entropies = torch.sum(per_class_entropies, dim=-1)

                # Add these to the list of uncertainties
                uncertainties.extend(sum_entropies.tolist())

        if return_string:
            return uncertainties[0]
        else:
            return uncertainties


class TAPTClassifier(HuggingFaceClassifier):
    """Classifier class based on a TAPTed HuggingFace model

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The container for the datasets in this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    model_name : str
        The name of the model, as on Hugging Face

    Attributes
    ----------
    tokenizer : transformers.AutoTokenizer
        The HuggingFace tokenizer associated with this classifier
    model : transformers.AutoModelForSequenceClassification
        The HuggingFace model for this classifier; reset to a new model every
        call of `train_afresh`
    optimizer : torch.optim.AdamW
        The torch optimizer used to update the model's parameters when training
    device : torch.device
        Set to either cuda (if GPU available) or CPU
    """

    def initialise(self):
        self._load_fresh_model()
        wandb.config.update({"tapt_classifier": self.training_parameters})

    def _load_fresh_model(self):
        """Load the TAPT classifier model afresh"""

        # Delete the old model to free up memory
        del self.model

        # use a temporary directory as an inbetween
        with tempfile.TemporaryDirectory() as tmpdirname:
            # download the model into this directory from wandb
            artifact_name = self.model_name + "---" + self.parameters["dataset_name"]
            artifact_path_components = (
                config["Wandb"]["Entity"],
                config["Wandb"]["Project"],
                artifact_name + ":latest",
            )
            artifact_path = "/".join(artifact_path_components)
            artifact = self.wandb_run.use_artifact(
                artifact_path,
                type=config["TAPT Model Loading"]["TAPTModelType"],
            )
            artifact.download(tmpdirname)

            # load the dictionary containing the parameters
            dict_file_path = os.path.join(
                tmpdirname, config["TAPT Model Loading"]["ParametersFileName"]
            )
            with open(dict_file_path, "rb") as f:
                tapt_parameters_dict = json.load(f)
                self.training_parameters = tapt_parameters_dict

            # load model from this directory
            model_file_path = os.path.join(
                tmpdirname, config["TAPT Model Loading"]["ModelFileName"]
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_file_path, num_labels=2
            )

        # Setup the model
        self._setup_model()


class PlainGPT2Classifier(HuggingFaceClassifier):
    """Classifier class based on GPT-2

    A classifier class that uses the GPT-2[1]_ model available on HuggingFace
    as a foundation for training a classifier; implemented by replacing
    the head of pretrained GPT-2 with a classifier.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The container for the datasets in this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run

    Attributes
    ----------
    tokenizer : transformers.AutoTokenizer
        The HuggingFace tokenizer associated with this classifier
    model : transformers.AutoModelForSequenceClassification
        The HuggingFace model for this classifier; reset to a new model every
        call of `train_afresh`
    optimizer : torch.optim.AdamW
        The torch optimizer used to update the model's parameters when training
    device : torch.device
        Set to either cuda (if GPU available) or CPU

    References
    ----------
    [1] Radford et al., "Language Models are Unsupervised Multitask Learners", 2019
    """

    MODEL_NAME = "gpt2"
    ARTIFACT_NAME = "gtp2-classifier"

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):
        super().__init__(
            parameters, dataset_container, wandb_run, model_name=self.MODEL_NAME
        )


class PlainDistilGPT2Classifier(HuggingFaceClassifier):
    """Classifier class based on DistilGPT2 by HuggingFace

    A classifier class that uses the DistilGPT2[1]_ model available on
    HuggingFace as a foundation for training a classifier; implemented by
    replacing the head of pretrained GPT-2 with a classifier.

    DistilGPT2 is trained under the supervision of the smallest GPT-2 model.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The container for the datasets in this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run

    Attributes
    ----------
    tokenizer : transformers.AutoTokenizer
        The HuggingFace tokenizer associated with this classifier
    model : transformers.AutoModelForSequenceClassification
        The HuggingFace model for this classifier; reset to a new model every
        call of `train_afresh`
    optimizer : torch.optim.AdamW
        The torch optimizer used to update the model's parameters when training
    device : torch.device
        Set to either cuda (if GPU available) or CPU

    References
    ----------
    [1] Victor et al., "DistilBERT, a distilled version of BERT: smaller,
    faster, cheaper and lighter", NeurIPS EMC^2 Workshop, 2019
    """

    MODEL_NAME = "distilgpt2"
    ARTIFACT_NAME = "distilgpt2-classifier"

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):
        super().__init__(
            parameters, dataset_container, wandb_run, model_name=self.MODEL_NAME
        )


class TAPTGPT2Classifier(TAPTClassifier):
    """Classifier class based on a TAPTed GPT-2 model

    A classifier class that uses the GPT-2[1]_ model available on HuggingFace
    as a foundation for training a classifier; implemented by replacing
    the head of pretrained GPT-2 with a classifier.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The container for the datasets in this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run

    Attributes
    ----------
    tokenizer : transformers.AutoTokenizer
        The HuggingFace tokenizer associated with this classifier
    model : transformers.AutoModelForSequenceClassification
        The HuggingFace model for this classifier; reset to a new model every
        call of `train_afresh`
    optimizer : torch.optim.AdamW
        The torch optimizer used to update the model's parameters when training
    device : torch.device
        Set to either cuda (if GPU available) or CPU

    References
    ----------
    [1] Radford et al., "Language Models are Unsupervised Multitask Learners", 2019
    """

    MODEL_NAME = "gpt2"
    ARTIFACT_NAME = "gtp2-classifier"

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):
        super().__init__(
            parameters, dataset_container, wandb_run, model_name=self.MODEL_NAME
        )


class TAPTDistilGPT2Classifier(TAPTClassifier):
    """Classifier class based on a TAPTed DistilGPT2

    A classifier class that uses the DistilGPT2[1]_ model available on
    HuggingFace as a foundation for training a classifier; implemented by
    replacing the head of pretrained GPT-2 with a classifier.

    DistilGPT2 is trained under the supervision of the smallest GPT-2 model.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The container for the datasets in this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run

    Attributes
    ----------
    tokenizer : transformers.AutoTokenizer
        The HuggingFace tokenizer associated with this classifier
    model : transformers.AutoModelForSequenceClassification
        The HuggingFace model for this classifier; reset to a new model every
        call of `train_afresh`
    optimizer : torch.optim.AdamW
        The torch optimizer used to update the model's parameters when training
    device : torch.device
        Set to either cuda (if GPU available) or CPU
    """

    MODEL_NAME = "distilgpt2"
    ARTIFACT_NAME = "distilgpt2-classifier"

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):
        super().__init__(
            parameters, dataset_container, wandb_run, model_name=self.MODEL_NAME
        )
