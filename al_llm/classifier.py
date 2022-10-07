# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Union, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
import datasets
import evaluate

import wandb

from tqdm import tqdm

from al_llm.parameters import Parameters
from al_llm.dataset_container import DatasetContainer
from al_llm.utils.artifacts import (
    save_classifier_model,
    load_classifier_model,
    load_tapted_model,
)
from al_llm.utils.models import HuggingFaceClassifierEnsemble
from al_llm.constants import LABEL_COLUMN_NAME


class MetricEvaluator:
    """Storage and maintainence of evalutation metrics

    Determines which metrics to use based on the number of classes
    (categories).

    In the multiclass setting, f1, precision and recall use the weighted
    average, but we also the unweighted versions, using '-unweighted' as a
    suffix.

    Parameters
    ----------
    dataset_container : DatasetContainer
        The dataset container for this experiment. We use it to determine how
        many classes there are.
    """

    UNWEIGHTED_SUFFIX = "-unweighted"

    def __init__(self, dataset_container: DatasetContainer):

        self.num_categories = len(dataset_container.CATEGORIES)

        # The metrics common to all classifiers
        self._metrics = {
            "accuracy": evaluate.load("accuracy"),
            "f1": evaluate.load("f1"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
        }

        # In the multiclass case, also record the unweighted version
        if self.num_categories != 2:
            self._metrics["f1" + self.UNWEIGHTED_SUFFIX] = evaluate.load("f1")
            self._metrics["precision" + self.UNWEIGHTED_SUFFIX] = evaluate.load(
                "precision"
            )
            self._metrics["recall" + self.UNWEIGHTED_SUFFIX] = evaluate.load("recall")

    def add_batch(
        self,
        *,
        predictions: Union[list, torch.Tensor],
        references: Union[list, torch.Tensor],
    ):
        """Add a batch of predictions and references for each metric

        Parameters
        ----------
        predictions : list or torch.Tensor
            The predicted values.
        references: list or torch.Tensor
            The true values
        """
        for metric in self._metrics.values():
            metric.add_batch(predictions=predictions, references=references)

    def compute(
        self,
        *,
        predictions: Optional[Union[list, torch.Tensor]] = None,
        references: Optional[Union[list, torch.Tensor]] = None,
    ):
        """Compute each metric

        Parameters
        ----------
        predictions : list or torch.Tensor, optional
            The predicted values.
        references: list or torch.Tensor, optional
            The true values

        Returns
        -------
        results : dict
            A dictionary of the results of computing each metric
        """

        results = {}

        for name, metric in self._metrics.items():

            # For accuracy, just compute it
            if name == "accuracy":
                computed = metric.compute(
                    predictions=predictions, references=references
                )
                results[name] = next(iter(computed.values()))

            else:

                # The extra arguments passed to `compute`
                compute_args = {}

                # Determine which type of average to use
                if name.endswith(self.UNWEIGHTED_SUFFIX):
                    compute_args["average"] = "macro"
                elif self.num_categories == 2:
                    compute_args["average"] = "binary"
                else:
                    compute_args["average"] = "weighted"

                # For precision and recall we need to set this value to prevent
                # warnings from appearing
                if name.startswith("precision") or name.startswith("recall"):
                    compute_args["zero_division"] = 0

                # Compute the metric using this average
                computed = metric.compute(
                    predictions=predictions, references=references, **compute_args
                )
                results[name] = next(iter(computed.values()))

        return results


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
    def _initialise(self):
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

    @abstractmethod
    def calculate_uncertainties_tokenized(
        self, tokenized_samples: torch.Tensor, print_output=True
    ) -> torch.Tensor:
        """Compute the uncertainty of tokenize samples

        Uncertainties are floats, whose interpretations depend on the
        classifier

        Parameters
        ----------
        tokenized_samples : torch.Tensor of shape (num_samples, num_tokens)
            A tensor of the tokenized samples for which to compute the
            uncertainty values
        print_output : bool, default=True
            Whether to print a message saying that uncertainties are being
            computed, and a progress bar

        Returns
        -------
        uncertainties : torch.Tensor of shape (num_samples)
            The uncertainties of the samples
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

    def _initialise(self):
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

    def calculate_uncertainties_tokenized(
        self, tokenized_samples: torch.Tensor, print_output=True
    ) -> torch.Tensor:
        return torch.zeros(tokenized_samples.shape[0])


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

    MODEL_NAME = ""

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
    ):

        # initialises the parameters in the same way as the base class
        super().__init__(parameters, dataset_container, wandb_run)

        # loads the tokenizer that the model will use
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set up the metric evaluator
        self.evaluator = MetricEvaluator(dataset_container)

        # The models are not required until a call to `train_afresh`
        self._model = None

        # set device
        if torch.cuda.is_available():
            self.device = torch.device(self.parameters["cuda_device"])
        else:
            self.device = torch.device("cpu")

    def train_afresh(
        self,
        tokenized_train: Union[datasets.Dataset, torch.utils.data.Dataset],
        iteration: int,
    ):

        # Get a fresh version of the model
        self._load_fresh_model()

        # Perform any first time setup required
        if iteration == 0:
            self._initialise()

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
        if self._model is None:
            self._load_model_from_wandb()

        # Make a smaple loader from the latest batch of labelled samples
        samples_dataloader = DataLoader(
            tokenized_samples, shuffle=True, batch_size=self.parameters["batch_size"]
        )

        # Run the training loop
        self._train(samples_dataloader, self.parameters["num_epochs_update"], iteration)

    def _initialise(self):
        pass

    def save(self):
        save_classifier_model(self.wandb_run, self._model)

    def _load_fresh_model(self):
        """Load the classifier model afresh"""

        # Delete the old model to free up memory
        del self._model

        models = []

        # Load fresh versions of the model
        for i in self.parameters["num_classifier_models"]:
            models.append(AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME, num_labels=len(self.dataset_container.CATEGORIES)
            ))

        # Create an ensemble of all of these
        self._model = HuggingFaceClassifierEnsemble(models)

        # Setup the model
        self._setup_model()

    def _load_model_from_wandb(self):
        """Load the classifier using the wandb_run"""

        # load and setup the model
        self._model = load_classifier_model(
            self.wandb_run, len(self.dataset_container.CATEGORIES)
        )
        self._setup_model()

    def _setup_model(self):
        """Perform some initial setup on the classifier model"""

        # set the End of Sentence token as the token used for padding by the model
        if isinstance(self._model, HuggingFaceClassifierEnsemble):
            for model in self._model.models:
                model.config.pad_token_id = model.config.eos_token_id
        else:
            self._model.config.pad_token_id = self._model.config.eos_token_id

        # Put the model on the torch device
        model.to(self.device)

    def _train(self, train_dataloader: DataLoader, num_epochs: int, iteration: int):

        # create an optimizer for the model
        optimizer = AdamW(self._model.parameters(), lr=self.parameters["learning_rate"])

        # The eval dataloader
        eval_dataloader = DataLoader(
            self.dataset_container.tokenized_validation,
            batch_size=self.parameters["eval_batch_size"],
        )

        # The test dataloader
        test_dataloader = DataLoader(
            self.dataset_container.tokenized_test,
            batch_size=self.parameters["eval_batch_size"],
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

            # Run the training loop, obtaining the metrics
            print("- Running train loop")
            train_metrics = self._train_epoch(train_dataloader, optimizer, lr_scheduler)
            print(
                f"Train loss: {train_metrics['loss']:.8}; "
                f"train f1: {train_metrics['f1']:.6%}"
            )

            # The results to log to weights and biases for this epoch
            results_to_log = {
                "epoch": epoch,
                "iteration": iteration,
                "train": train_metrics,
            }

            # Run eval and test loops, if required
            for split, dataloader in [
                ("eval", eval_dataloader),
                ("test", test_dataloader),
            ]:

                # If the eval loop should run this epoch, or if it is the last epoch
                run_eval = (
                    self.parameters[f"{split}_every"] > 0
                    and (epoch + 1) % self.parameters[f"{split}_every"] == 0
                )
                run_eval = run_eval or (
                    self.parameters[f"{split}_every"] >= 0 and epoch == num_epochs - 1
                )

                if run_eval:
                    # Run the evaluation loop, obtaining the metrics
                    print(f"- Running {split} loop")
                    eval_metrics = self._eval_epoch(dataloader)
                    print(
                        f"{split.capitalize()} loss: {eval_metrics['loss']:.8}; "
                        f"{split.capitalize()} f1: {eval_metrics['f1']:.6%}"
                    )
                    results_to_log[split] = eval_metrics

            # Record the metrics with W&B
            self.wandb_run.log(results_to_log)

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
        self._model.train()

        # Recording the training loss by accumulating across batches
        train_loss = 0

        # iterate over all the batches in the dataloader
        for batch in tqdm(train_dataloader):

            # move batch data to same device as the model
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # pass the data to the model
            outputs = self._model(**batch)

            # work out the model's predictions for the data using argmax on
            # the logits
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Use these outputs to calculate metrics
            self.evaluator.add_batch(
                predictions=predictions,
                references=batch[LABEL_COLUMN_NAME],
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
        self._model.eval()

        # Recording the evaluation loss by accumulating across batches
        eval_loss = 0

        # iterate over all the batches in the dataloader
        for batch in tqdm(eval_dataloader):

            # move batch data to same device as the model
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # pass the data through the model without tracking computations
            with torch.no_grad():
                outputs = self._model(**batch)
            logits = outputs.logits

            # work out the model's predictions for the data using argmax on the logits
            predictions = torch.argmax(logits, dim=-1)

            # give the predictions to the metric(s)
            self.evaluator.add_batch(
                predictions=predictions,
                references=batch[LABEL_COLUMN_NAME],
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
        elif isinstance(samples, list):
            return_string = False
        else:
            raise TypeError(
                f"Parameter `samples` must be a string or list, got {type(samples)}"
            )

        # Tokenize the samples, ready for feeding into the model
        tokenized_samples_dict = self.tokenize(samples)
        tokenized_samples_dataset = datasets.Dataset.from_dict(tokenized_samples_dict)
        tokenized_samples_dataset.set_format("torch", columns=["input_ids"])
        tokenized_samples = tokenized_samples_dataset["input_ids"]

        # Compute the uncertainties, as a PyTorch tensor
        uncertainties = self.calculate_uncertainties_tokenized(tokenized_samples)

        if return_string:
            return uncertainties.item()
        else:
            return uncertainties.tolist()

    def calculate_uncertainties_tokenized(
        self, tokenized_samples: torch.Tensor, print_output=True
    ) -> torch.Tensor:

        # Get the number of samples
        num_samples = tokenized_samples.shape[0]

        # Store the batch size with a shorter variable name
        batch_size = self.parameters["eval_batch_size"]

        # Make a PyTorch dataloader for the samples
        samples_dataloader = DataLoader(tokenized_samples, batch_size=batch_size)

        # set the model to eval mode
        self._model.eval()

        # A list of the uncertainties (entropies) for each element of `samples`
        uncertainties = torch.zeros(num_samples, device=self.device)

        if print_output:
            # Print a message to say what we're doing
            print()
            print("Computing uncertainties...")

        # Make an iterator, including a progress bar if requested
        iterator = enumerate(samples_dataloader)
        if print_output:
            iterator = tqdm(iterator, total=len(samples_dataloader))

        # iterate over all the batches in the dataloader
        for idx, batch in iterator:

            # Move the batch to the appropriate device
            batch = batch.to(self.device)

            with torch.no_grad():

                # Get the raw model output logits
                outputs = self._model(input_ids=batch)
                logits = outputs.logits

                # Compute the class probabilities
                probabilities = F.softmax(logits, dim=1)

                # Compute the entropies per element
                per_class_entropies = torch.special.entr(probabilities)

                # Take the sum over the entropies per class, to yield the
                # total entropy per sample
                sum_entropies = torch.sum(per_class_entropies, dim=-1)

                # Add these to the list of uncertainties

                uncertainties[idx * batch_size : (idx + 1) * batch_size] = sum_entropies

        return uncertainties

    @property
    def model(self) -> PreTrainedModel:
        """Get the Hugging Face model for this classifier, if it exists"""
        if self._model is not None:
            return self._model
        else:
            raise AttributeError


class TAPTClassifier(HuggingFaceClassifier, ABC):
    """Classifier class based on a TAPTed HuggingFace model

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

    def _initialise(self):
        wandb.config.update({"tapt_classifier": self.training_parameters})

    def _load_fresh_model(self):
        """Load the TAPT classifier model afresh"""

        # Delete the old model to free up memory
        del self._model

        # load model and training args from wandb
        model, training_args = load_tapted_model(
            self.wandb_run,
            self.MODEL_NAME,
            self.parameters["dataset_name"],
            "classifier",
            num_categories=len(self.dataset_container.CATEGORIES),
            tapted_model_version=self.parameters["tapted_model_version"],
        )
        self._model = model
        self.training_parameters = training_args

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
