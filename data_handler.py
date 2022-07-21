# The python abc module for making abstract base classes
# https://docs.python.org/3.10/library/abc.html
from abc import ABC, abstractmethod

import torch
from torch.utils.data import TensorDataset

class DataHandler(ABC):
    """Base class for loading and processing the data"""


    def __init__(self):
        self.dataset = None
        self.tokenized = None
        self.tokenizer = None

    def register_tokenizer(self, tokenizer):
        """Register a tokeniser for the data
        
        Parameters
        ----------
        tokenizer : function
            A function which takes a string or a batch of strings and
            tokenises them into PyTorch tensors
        """
        self.tokenizer = tokenizer

    
    def _tokenize(self, text):
        """Tokenize a string or batch of strings
        
        Parameters
        ----------
        text : str or list
            The string or batch of strings to be tokenised
        
        Returns
        -------
        tokenized : torch.Tensor
            The result of tokenizing `text`
        """

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not registered")

        return self.tokenizer(text)

    
    @abstractmethod
    def new_labelled(self, samples, labels):
        """Add new labelled samples, returning a PyTorch dataset for them
        
        Parameters
        ----------
        samples : list
            The list of sample strings
        labels : list
            Labels for the samples

        Returns
        -------
        samples_dataset : torch.utils.data.Dataset
            A PyTorch dataset consisting of the newly added tokenized samples
            and their labels, ready for fine-tuning
        """
        return None


class DummyDataHandler(DataHandler):
    
    def new_labelled(self, samples, labels):
        return TensorDataset(torch.rand(100, 100))