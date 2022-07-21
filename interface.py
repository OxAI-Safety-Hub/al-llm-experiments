# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

class BaseInterface(ABC):
    """Base interface class
    
    Parameters
    ----------
    categories : dict
        A dictionary of categories used by the classifier. The keys are the
        names of the categories as understood by the model, and the values
        are the human-readable names.    
    """

    def __init__(self, categories):
        self.categories = categories


    def begin(self):
        """Initialise the interface"""
        pass


    @abstractmethod
    def prompt(self, samples):
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

        # Make sure that we have a list of samples
        if not isinstance(samples, list):
            raise TypeError("Parameter `samples` must be a list")

        return []


    def train_afresh(self):
        """Tell the human that we're fine-tuning from scratch"""
        pass


    def train_update(self):
        """Tell the human that we're fine-tuning with new datapoints"""
        pass


    def end(self):
        """Close the interface at the end of the experiment"""
        pass


class CLIInterface(BaseInterface):
    """A command line interface for obtaining labels

    Enumerates the categories, and asks the human to input the number
    corresponding to the appropriate category for each sample
    
    Parameters
    ----------
    categories : dict
        A dictionary of categories used by the classifier. The keys are the
        names of the categories as understood by the model, and the values
        are the human-readable names.
    """


    def __init__(self, categories):
        super().__init__(categories)
        self._categories_list = [(k, v) for k,v in self.categories.items()]
        self._num_categories = len(self._categories_list)


    def begin(self, message=None, parameters=None):
        """Initialise the interface, displaying a welcome message
        
        Parameters
        ----------
        message : str, optional
            The welcome message to display. Defaults to a generic message.
        parameters : dict, optional
            The parameters used in this experiment
        """

        # Default message
        if message is None:
            message = (
                "Welcome to the large language generative model active " 
                "learning experiment!"
            )
        
        # Compose the message and display it in a box
        text = message
        if parameters is not None:
            text += f"\nParameters: {parameters}"
        self._head_text(text, initial_gap=False)


    def prompt(self, samples):

        super().prompt(samples)

        labels = []

        # Loop over all the samples for which we need a label
        for sample in samples:

            # Print the sample, plus the category selection
            print()
            print(f"{sample!r}")
            print("How would you classify this?")
            for i, (cat_name, cat_human) in enumerate(self._categories_list):
                print(f"[{i}] {cat_human}")

            # Keep asking the user for a label until they give a valid one
            prompt = f"Enter a number (0-{self._num_categories-1}):"
            valid_label = False
            while not valid_label:
                label_str = input(prompt)
                try:
                    label =  int(label_str)
                except ValueError:
                    continue
                if label >= 0 and label < self._num_categories:
                    valid_label = True
            
            # Append this label
            labels.append(self._categories_list[label])

        return labels


    def train_afresh(self, message=None):
        """Tell the human that we're fine-tuning from scratch
        
        Parameters
        ----------
        message : str, optional
            The message to display. Defaults to a generic message.
        """

        # Default message
        if message is None:
            message = "Fine-tuning from scratch. This may take a while..."

        # Print the message
        self._head_text(message)


    def train_afresh(self, message=None):
        """Tell the human that we're fine-tuning from scratch
        
        Parameters
        ----------
        message : str, optional
            The message to display. Defaults to a generic message.
        """

        # Default message
        if message is None:
            message = "Fine-tuning with new datapoints..."

        # Print the message
        self._head_text(message)


    def end(self, message=None, results=None):
        """Initialise the interface, displaying a goodbye message
        
        Parameters
        ----------
        message : str, optional
            The goodbye message to display. Defaults to a generic message.
        results : any, optional
            Any results that should be displayed to the human.
        """

        # Default message
        if message is None:
            message = "Thank you for participating!"
        
        # Compose the message and display it in a box
        text = message
        if results is not None:
            text += f"\Results: {results}"
        self._head_text(text)


    def _head_text(self, text, initial_gap=True):
        # Display a message with horizontal rules above and below
        if initial_gap:
            print()
        print("------------------------------------------------------------")
        print(text)
        print("------------------------------------------------------------")