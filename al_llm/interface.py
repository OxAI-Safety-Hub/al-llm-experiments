# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Any
import textwrap

import torch

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

import wandb

from al_llm.dataset_container import DatasetContainer
from al_llm.parameters import Parameters
from al_llm.data_handler import DataHandler
from al_llm.utils import UnlabelledSamples, PromptOutput


class Interface(ABC):
    """Base interface class

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for this experiment
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

    def begin(self, message: str = None):
        """Initialise the interface, displaying a welcome message

        Parameters
        ----------
        message : str, optional
            The welcome message to display. Defaults to a generic message.
        """
        pass

    def train_afresh(self, message: str = None, iteration=None):
        """Tell the human that we're fine-tuning from scratch

        Parameters
        ----------
        message : str, optional
            The message to display. Defaults to a generic message.
        iteration : int, optional
            The current iteration of the AL loop
        """
        pass

    def train_update(self, message: str = None, iteration=None):
        """Tell the human that we're fine-tuning with new datapoints

        Parameters
        ----------
        message : str, optional
            The message to display. Defaults to a generic message.
        iteration : int, optional
            The current iteration of the AL loop
        """
        pass

    def end(self, message: str = None, results: Any = None):
        """Close the interface, displaying a goodbye message

        Parameters
        ----------
        message : str, optional
            The goodbye message to display. Defaults to a generic message.
        results : any, optional
            Any results that should be displayed to the human.
        """
        pass


class FullLoopInterface(Interface, ABC):
    """Base interface class for running the loop all in one go

    Parameters
    ----------
    dataset_container : DatasetContainer
        The dataset container for this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    """

    @abstractmethod
    def prompt(self, samples: UnlabelledSamples) -> PromptOutput:
        """Prompt the human for labels, ambiguities and skips for the samples

        Parameters
        ----------
        samples : UnlabelledSamples
            A list of samples to query the human

        Returns
        -------
        prompt_output : PromptOutput
            A data class object storing the labels and possibly ambiguities
            and the skip mask.
        """

        return PromptOutput(labels=[])


class BrokenLoopInterface(Interface, ABC):
    """Base interface class for broken loop experiments

    Parameters
    ----------
    dataset_container : DatasetContainer
        The dataset container for this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    """

    pass


class CLIInterfaceMixin:
    """A mixin providing useful methods for CLI interfaces

    Attributes
    ----------
    line_width : int
        The width of the lines to wrap the output.
    """

    def _output(self, text: str):
        """Output something to the CLI"""
        print(text)

    def _input(self, prompt: str) -> str:
        """Get some input from the CLI"""
        return input(prompt)

    def _wrap(self, text: str) -> str:
        """Wrap some text to the line width"""
        return textwrap.fill(text, width=self.line_width)

    def _head_text(
        self, message: str, initial_newline: bool = True, trailing_newline: bool = False
    ) -> str:
        """Generate a message with horizontal rules above and below"""
        text = ""
        if initial_newline:
            text += "\n"
        text += self._horizontal_rule()
        text += message + "\n"
        text += self._horizontal_rule(trailing_newline)
        return text

    def _horizontal_rule(self, trailing_newline: bool = True) -> str:
        """Generate a horizontal rule"""
        text = "-" * self.line_width
        if trailing_newline:
            text += "\n"
        return text


class SimpleCLIInterfaceMixin(CLIInterfaceMixin, ABC):
    """A CLI interface mixin which provides basic CLI outputs"""

    def begin(self, message: str = None):

        # Default message
        if message is None:
            message = "AL LLM"

        # Wrap the message
        text = self._wrap(message)

        # Add the parameters
        if self.parameters is not None:
            text += "\n" + self._horizontal_rule()
            parameter_string = f"Parameters: {self.parameters}"
            text += self._wrap(parameter_string)

        text += "\n" + self._horizontal_rule()
        run_id_string = f"Run ID: {self.wandb_run.id}"
        text += self._wrap(run_id_string)

        # Print the message
        text = self._head_text(text, initial_newline=False)
        self._output(text)

    def train_afresh(self, message: str = None, iteration=None):

        # Default message
        if message is None:
            message = "Fine-tuning from scratch..."

        # Prepend the iteration index
        message = f"[{iteration}] {message}"

        # Wrap the message
        text = self._wrap(message)

        # Print the message
        text = self._head_text(text, initial_newline=False)
        self._output(text)

    def train_update(self, message: str = None, iteration=None):

        # Default message
        if message is None:
            message = "Fine-tuning with new datapoints..."

        # Prepend the iteration index
        message = f"[{iteration}] {message}"

        # Wrap the message
        text = self._wrap(message)

        # Print the message
        text = self._head_text(text, initial_newline=False)
        self._output(text)

    def _output(self, text: str):
        """Output something to the CLI"""
        print(text)


class CLIInterface(CLIInterfaceMixin, FullLoopInterface):
    """A command line interface for obtaining labels

    Enumerates the categories, and asks the human to input the number
    corresponding to the appropriate category for each sample

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    line_width : int
        The width of the lines to wrap the output.
    """

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        *,
        line_width: int = 70,
    ):

        super().__init__(parameters, dataset_container, wandb_run)

        self.line_width = line_width

    def begin(self, message: str = None):

        # Default message
        if message is None:
            message = (
                "Welcome to the large language generative model active "
                "learning experiment!"
            )

        # Wrap the message
        text = self._wrap(message)

        # Add the parameters
        if self.parameters is not None:
            text += "\n" + self._horizontal_rule()
            parameter_string = f"Parameters: {self.parameters}"
            text += self._wrap(parameter_string)

        # Print the message
        text = self._head_text(text, initial_newline=False)
        self._output(text)

    def prompt(self, samples: UnlabelledSamples) -> PromptOutput:
        """Prompt the human for labels, ambiguities and skips for the samples

        Parameters
        ----------
        samples : UnlabelledSamples
            A list of samples to query the human

        Returns
        -------
        prompt_output : PromptOutput
            A data class object storing the labels, ambiguities and the skip
            mask.
        """

        # Create the initial PromptOutput object
        getting_ambiguities = self.parameters["ambiguity_mode"] != "none"
        getting_skips = self.parameters["allow_skipping"]
        ambiguities = [] if getting_ambiguities else None
        skip_mask = [] if getting_skips else None
        prompt_output = PromptOutput(
            labels=[], ambiguities=ambiguities, skip_mask=skip_mask
        )

        # Loop over all the samples for which we need a label
        for i, sample in enumerate(samples):

            # Build first part of the message, consisting of the sample plus
            # the question
            text = "\n"
            text += self._wrap(f"[{i+1}/{len(samples)}]") + "\n"
            text += self._wrap(f"{sample!r}") + "\n"
            text += self._wrap("How would you classify this?") + "\n"

            # Keep track of the option number as we build up the list of
            # options
            option_num = 0

            # Add the category selection
            categories = self.dataset_container.categories
            for cat_human_readable in categories.values():
                text += self._wrap(f"[{option_num}] {cat_human_readable}") + "\n"
                option_num += 1

            # If also checking for ambiguity, add these options
            if getting_ambiguities:
                for cat_human_readable in categories.values():
                    text += (
                        self._wrap(f"[{option_num}] {cat_human_readable} (ambiguous)")
                        + "\n"
                    )
                    option_num += 1

            # Add a skip option if we're using it
            if getting_skips:
                text += self._wrap(f"[{option_num}] Skip") + "\n"
                option_num += 1

            # Print the message
            self._output(text)

            # Build the prompt
            prompt = self._wrap(f"Enter a number (0-{option_num - 1}): ")

            # Keep asking the user for a label until they give a valid one
            valid_label = False
            while not valid_label:
                label_str = self._input(prompt)
                try:
                    label = int(label_str)
                except ValueError:
                    continue
                if label >= 0 and label < option_num:
                    valid_label = True

            # If skipping record this and assign and arbitrary label and
            # ambiguity
            if getting_skips and label == option_num - 1:
                prompt_output.skip_mask.append(1)
                prompt_output.labels.append(list(categories.keys())[0])
                if getting_ambiguities:
                    prompt_output.ambiguities.append(0)

            # Otherwise, append this label with the ambiguity assigned
            else:
                prompt_output.labels.append(
                    list(categories.keys())[label % len(categories)]
                )
                if getting_ambiguities:
                    prompt_output.ambiguities.append(label // len(categories))
                if getting_skips:
                    prompt_output.skip_mask.append(0)

        return prompt_output

    def train_afresh(self, message: str = None, iteration=None):

        # Default message
        if message is None:
            message = "Fine-tuning from scratch. This may take a while..."

        # Prepend the iteration index
        message = f"[{iteration}] {message}"

        # Wrap the message
        text = self._wrap(message)

        # Print the message
        text = self._head_text(text)
        self._output(text)

    def train_update(self, message: str = None, iteration=None):

        # Default message
        if message is None:
            message = "Fine-tuning with new datapoints..."

        # Prepend the iteration index
        message = f"[{iteration}] {message}"

        # Wrap the message
        text = self._wrap(message)

        # Print the message
        text = self._head_text(text)
        self._output(text)

    def end(self, message: str = None, results: Any = None):

        # Default message
        if message is None:
            message = "Thank you for participating!"

        # Wrap the message
        text = self._wrap(message)

        # Add any results
        if results is not None:
            results_string = f"Results: {results}"
            text += "\n" + self._wrap(results_string)

        # Print the message
        text = self._head_text(text)
        self._output(text)


class CLIBrokenLoopInterface(SimpleCLIInterfaceMixin, BrokenLoopInterface):
    """A CLI implementation of an interface for broken loop experiments

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    line_width : int, default=70
        The width of the lines to wrap the output.
    """

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        *,
        line_width: int = 70,
    ):
        super().__init__(parameters, dataset_container, wandb_run)
        self.line_width = line_width


class PoolSimulatorInterface(SimpleCLIInterfaceMixin, Interface):
    """Interface for simulating pool-based active learning

    This interface uses the remainder dataset as a pool of labelled samples.
    To simulate active learning, we pretend that these are unlabelled, select
    from them, and then simulate labelling them by taking the labels we
    actually have for them.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    line_width : int, default=70
        The width of the lines to wrap the output.
    """

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        *,
        line_width: int = 70,
    ):
        super().__init__(parameters, dataset_container, wandb_run)
        self.line_width = line_width

    def prompt(self, samples: UnlabelledSamples) -> PromptOutput:
        """Prompt the data for labels for the samples

        Parameters
        ----------
        samples : UnlabelledSamples
            A list of samples to query the data

        Returns
        -------
        prompt_output : PromptOutput
            A data class object storing the labels
        """

        return PromptOutput(labels=samples.suggested_labels)


class AutomaticLabellerInterface(SimpleCLIInterfaceMixin, Interface):
    """Interface for obtaining labels from a pretrained model on Hugging Face

    Loads a Hugging Face classifier model from the online repository by its
    name, and uses it as an oracle to provide labels for the sentences.

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    model_name : str
        The name of the Hugging Face model to load
    line_width : int, default=70
        The width of the lines to wrap the output.

    Attributes
    ----------
    pipeline : transformers.Pipeline
        The Hugging Face pipeline used for text classification
    """

    # For each model we use, a mapping translating labels understood by the
    # automatic labeller to the labels given by the dataset.
    # This is only necessary for models which output labels which are different
    # from those of the dataset.
    CLASS_LABEL_MAPPING = {
        "textattack/roberta-base-rotten-tomatoes": {
            "LABEL_0": "neg",
            "LABEL_1": "pos",
        }
    }

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        model_name: str,
        *,
        line_width: int = 70,
    ):
        super().__init__(parameters, dataset_container, wandb_run)
        self.model_name = model_name
        self.line_width = line_width

        # Set the device
        if torch.cuda.is_available():
            device = torch.device(self.parameters["cuda_device"])
        else:
            device = torch.device("cpu")

        # Make the classification pipeline
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )

    def prompt(self, samples: UnlabelledSamples) -> PromptOutput:
        """Prompt the machine for labels for the samples

        Parameters
        ----------
        samples : UnlabelledSamples
            A list of samples to query the machine

        Returns
        -------
        prompt_output : PromptOutput
            A data class object storing the labels
        """

        # Make a dataset out of the samples
        # Somewhat roundabout way of making a simple `Dataset` where
        # `__getitem__` returns the values of `samples`
        samples_dataset = KeyDataset(
            datasets.Dataset.from_dict({"text": samples}), "text"
        )

        # Announce what we're doing
        print()
        print("Getting labels from the automatic labeller...")

        # Get the outputs of the model
        outputs = self.pipeline(
            samples_dataset, batch_size=self.parameters["eval_batch_size"]
        )

        # Get the actual labels from these
        labels = [self._map_class_label(output["label"]) for output in outputs]

        return PromptOutput(labels=labels)

    def _map_class_label(self, label: str) -> str:
        """Map a label given by the automatic labeller to a dataset label

        If we haven't explicitly specified a mapping for the current model,
        returns the label unchanged. In this case we simply assume that the
        model outputs the correct label name (as it really ought to).
        """

        if self.model_name in self.CLASS_LABEL_MAPPING:
            return self.CLASS_LABEL_MAPPING[self.model_name][label]
        else:
            return label


class ReplayInterface(SimpleCLIInterfaceMixin, Interface):
    """Interface for replaying a run, using already obtained labels

    Parameters
    ----------
    parameters : Parameters
        The dictionary of parameters for the present experiment
    dataset_container : DatasetContainer
        The dataset container for this experiment
    wandb_run : wandb.sdk.wandb_run.Run
        The current wandb run
    data_handler : DataHandler
        The data handler for this experiment
    line_width : int, default=70
        The width of the lines to wrap the output.
    """

    def __init__(
        self,
        parameters: Parameters,
        dataset_container: DatasetContainer,
        wandb_run: wandb.sdk.wandb_run.Run,
        data_handler: DataHandler,
        *,
        line_width: int = 70,
    ):
        super().__init__(parameters, dataset_container, wandb_run)
        self.data_handler = data_handler
        self.line_width = line_width

        # The current index in the replay dataset extension
        self._iteration = 0

    def prompt(self, samples: UnlabelledSamples) -> PromptOutput:

        # Announce what we're doing
        print()
        print("Getting labels from the replayed run...")

        # Get the next batch of samples from the replayed run
        prompt_output = self.data_handler.get_replay_prompt_output(self._iteration)

        # Update the index
        self._iteration += 1

        return prompt_output
