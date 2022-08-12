# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod
from typing import Any

import textwrap
from al_llm.parameters import Parameters


class Interface(ABC):
    """Base interface class

    Parameters
    ----------
    categories : dict
        A dictionary of categories used by the classifier. The keys are the
        names of the categories as understood by the model, and the values
        are the human-readable names.
    run_id : str
        The ID of the current run
    """

    def __init__(self, categories: dict, run_id: str):
        self.categories = categories
        self.run_id = run_id

    def begin(self, message: str = None, parameters: Parameters = None):
        """Initialise the interface, displaying a welcome message

        Parameters
        ----------
        message : str, optional
            The welcome message to display. Defaults to a generic message.
        parameters : Parameters, optional
            The parameters used in this experiment
        """
        pass

    def train_afresh(self, message: str = None):
        """Tell the human that we're fine-tuning from scratch

        Parameters
        ----------
        message : str, optional
            The message to display. Defaults to a generic message.
        """
        pass

    def train_update(self, message: str = None):
        """Tell the human that we're fine-tuning with new datapoints

        Parameters
        ----------
        message : str, optional
            The message to display. Defaults to a generic message.
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
    categories : dict
        A dictionary of categories used by the classifier. The keys are the
        names of the categories as understood by the model, and the values
        are the human-readable names.
    run_id : str
        The ID of the current run
    """

    @abstractmethod
    def prompt(self, samples: list) -> list:
        """Prompt the human for labels for the samples

        Parameters
        ----------
        samples : list
            A list of samples to query the human

        Returns
        -------
        labels : list
            A list of labels, one for each element in `samples`
        """

        return []


class BrokenLoopInterface(Interface, ABC):
    """Base interface class for broken loop experiments

    Parameters
    ----------
    categories : dict
        A dictionary of categories used by the classifier. The keys are the
        names of the categories as understood by the model, and the values
        are the human-readable names.
    run_id : str
        The ID of the current run
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


class CLIInterface(CLIInterfaceMixin, FullLoopInterface):
    """A command line interface for obtaining labels

    Enumerates the categories, and asks the human to input the number
    corresponding to the appropriate category for each sample

    Parameters
    ----------
    categories : dict
        A dictionary of categories used by the classifier. The keys are the
        names of the categories as understood by the model, and the values
        are the human-readable names.
    run_id : str
        The ID of the current run
    line_width : int
        The width of the lines to wrap the output.
    """

    def __init__(self, categories: dict, run_id: str, *, line_width: int = 70):

        super().__init__(categories, run_id)

        self.line_width = line_width

        self._categories_list = [(k, v) for k, v in self.categories.items()]
        self._num_categories = len(self._categories_list)

    def begin(self, message: str = None, parameters: Parameters = None):

        # Default message
        if message is None:
            message = (
                "Welcome to the large language generative model active "
                "learning experiment!"
            )

        # Wrap the message
        text = self._wrap(message)

        # Add the parameters
        if parameters is not None:
            text += "\n" + self._horizontal_rule()
            parameter_string = f"Parameters: {parameters}"
            text += self._wrap(parameter_string)

        # Print the message
        text = self._head_text(text, initial_newline=False)
        self._output(text)

    def prompt(self, samples: list) -> list:

        super().prompt(samples)

        labels = []

        # Loop over all the samples for which we need a label
        for sample in samples:

            # Build the message with the sample plus the category selection
            text = "\n"
            text += self._wrap(f"{sample!r}") + "\n"
            text += self._wrap("How would you classify this?") + "\n"
            for i, (cat_name, cat_human) in enumerate(self._categories_list):
                text += self._wrap(f"[{i}] {cat_human}") + "\n"

            # Print the message
            self._output(text)

            # Keep asking the user for a label until they give a valid one
            prompt = self._wrap(f"Enter a number (0-{self._num_categories-1}):")
            valid_label = False
            while not valid_label:
                label_str = self._input(prompt)
                try:
                    label = int(label_str)
                except ValueError:
                    continue
                if label >= 0 and label < self._num_categories:
                    valid_label = True

            # Append this label
            labels.append(self._categories_list[label])

        return labels

    def train_afresh(self, message: str = None):

        # Default message
        if message is None:
            message = "Fine-tuning from scratch. This may take a while..."

        # Wrap the message
        text = self._wrap(message)

        # Print the message
        text = self._head_text(text)
        self._output(text)

    def train_update(self, message: str = None):

        # Default message
        if message is None:
            message = "Fine-tuning with new datapoints..."

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


class CLIBrokenLoopInterface(CLIInterfaceMixin, BrokenLoopInterface):
    """A CLI implementation of an interface for broken loop experiments

    Parameters
    ----------
    categories : dict
        A dictionary of categories used by the classifier. The keys are the
        names of the categories as understood by the model, and the values
        are the human-readable names.
    run_id : str
        The ID of the current run
    line_width : int
        The width of the lines to wrap the output.
    """

    def __init__(self, categories: dict, run_id: str, *, line_width: int = 70):
        super().__init__(categories, run_id)
        self.line_width = line_width

    def begin(self, message: str = None, parameters: Parameters = None):

        # Default message
        if message is None:
            message = "AL LLM"

        # Wrap the message
        text = self._wrap(message)

        # Add the parameters
        if parameters is not None:
            parameter_string = f"Parameters: {parameters}"
            text += "\n" + self._wrap(parameter_string)

        run_id_string = f"Parameters: {self.run_id}"
        text += "\n" + self._wrap(run_id_string)

        # Print the message
        self._output(text)

    def train_afresh(self, message: str = None):

        # Default message
        if message is None:
            message = "Fine-tuning from scratch..."

        # Wrap the message
        text = self._wrap(message)

        # Print the message
        self._output(text)

    def train_update(self, message: str = None):

        # Default message
        if message is None:
            message = "Fine-tuning with new datapoints..."

        # Wrap the message
        text = self._wrap(message)

        # Print the message
        self._output(text)

    def _output(self, text: str):
        """Output something to the CLI"""
        print(text)
