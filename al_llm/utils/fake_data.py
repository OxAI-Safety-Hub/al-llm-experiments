from random import Random

from typing import Optional


class FakeSentenceGenerator:
    """Generator for sequences of English words

    The number of words in the sentences is governed by a log normal
    distribution

    Parameters
    ----------
    seed : int, optional
        The random seed to use
    sentence_len_mu : float, default=1.2
        The mu parameter for the log normal distribution for the number of
        words in a sentence
    sentence_len_sigma : float, default=0.8
        The mu parameter for the log normal distribution for the number of
        words in a sentence
    """

    WORD_LIST_FILENAME = "al_llm/utils/word_list.txt"

    def __init__(
        self,
        seed: Optional[int] = None,
        sentence_len_mu: float = 1.2,
        sentence_len_sigma: float = 0.8,
    ):

        # Load the words from the word list file. The file size should be less
        # than 10000 bytes
        with open(self.WORD_LIST_FILENAME, "r") as f:
            self.word_list = f.read(10000).split("\n")

        # Instantiate the random number generator
        self._number_generator = Random(seed)

        self.sentence_len_mu = sentence_len_mu
        self.sentence_len_sigma = sentence_len_sigma

    def generate(self, num_sentences: int, allow_repeats=False) -> list:
        """Generate a list of fake sentences

        Parameters
        ----------
        num_sentences : int
            The number of sentences to generate
        allow_repeats : bool, default=False
            Whether to allow repeats of the same sentences

        Returns
        -------
        sentences : list
            A list of fake sentences
        """

        sentences = []
        for i in range(num_sentences):

            # The length of the sentence
            num_words = int(
                self._number_generator.lognormvariate(
                    self.sentence_len_mu, self.sentence_len_sigma
                )
            )
            num_words = max(1, num_words)

            sentence = None

            while sentence is None:

                # Pick the words from the word list
                sentence_words = self._number_generator.choices(
                    self.word_list, k=num_words
                )

                # Turn into a capitalised sentence
                sentence = " ".join(sentence_words)
                sentence = sentence.capitalize()

                # If we're not allowing repeats, make sure this sentence hasn't
                # already appeared
                if not allow_repeats and sentence in sentences:
                    sentence = None

            sentences.append(sentence)

        return sentences


class FakeLabelGenerator:
    """Generator for fake class labels

    Parameters
    ----------
    seed : int, optional
        The random seed to use
    categories : list
        The list of possible class labels
    """

    def __init__(self, categories: list, seed: Optional[int] = None):

        # Instantiate the random number generator
        self._number_generator = Random(seed)

        self.categories = categories

    def generate(self, num_labels: int) -> list:
        """Generate a list of fake class labels

        Parameters
        ----------
        num_labels : int
            The number of labels to generate

        Returns
        -------
        sentences : list
            A list of fake sentences
        """
        return self._number_generator.choices(self.categories, k=num_labels)
